"""Real Phase B example: row-sum via *dynamic*-scheduled megakernel.

Same workload as ``row_sum_megakernel.py`` (paper Fig. 3) but compiled
with the **dynamic** scheduling transformation (Algorithm 2).

Differences from the static version:

    * Bodies are *pure compute* -- no ``tl.atomic_add`` / no while-loop
      spin-wait.  The emitter inserts notify-and-push automatically
      based on the IR's out_edges, and consumer tasks are only pushed
      once their event counter triggers, so the wait disappears.
    * Event-tensor topology is *grouped*: one event per consumer task
      with ``wait_count = J`` (producers per consumer), instead of one
      event per producer with ``wait_count = 1``.  This is the natural
      shape for the dynamic scheduler -- a single counter decrement per
      producer; the last decrement triggers the consumer push.
    * No precomputed per-SM queue.  The scheduler is a global ready
      queue with atomic head/tail; SMs work-steal.

Validation: build the IR with policy='dynamic', lower via
:func:`lower_megakernel_dynamic`, import the emitted source, launch on
the GPU, compare to ``a.sum(dim=-1)``.  Numerical match is required.
"""

from __future__ import annotations

import importlib.util
import linecache
import os
import tempfile
from dataclasses import dataclass

import torch
from xdsl.dialects.builtin import (
    ArrayAttr,
    IntegerAttr,
    IntegerType,
    ModuleOp,
    StringAttr,
    SymbolRefAttr,
)
from xdsl.ir import Block, Region

from compgen.ir.event.attrs import EventCoordAttr, EventTensorTypeAttr
from compgen.ir.event.ops import CallDeviceOp, EventTensorOp, GraphOp
from compgen.ir.tile.lower_megakernel_dynamic import (
    DynamicDeviceFunctionSpec,
    DynamicMegakernelLoweringResult,
    DynamicMegakernelLoweringSpec,
    lower_megakernel_dynamic,
)


# ---------------------------------------------------------------------------
# Pure-compute Triton bodies (no event ops; emitter inserts them)
# ---------------------------------------------------------------------------

# partial_sum(task_id, A_ptr, B_ptr, C_ptr, N_ROW_BLOCKS, J, K, BLOCK_M, BLOCK_K)
#   k       = task_id
#   i, j    = k // J, k % J
_PARTIAL_SUM_BODY = """
i = task_id // J
j = task_id %  J

row_offsets = i * BLOCK_M + tl.arange(0, BLOCK_M)
col_offsets = j * BLOCK_K + tl.arange(0, BLOCK_K)

a_ptrs = A_ptr + row_offsets[:, None] * K + col_offsets[None, :]
a = tl.load(a_ptrs)
partial = tl.sum(a, axis=1)

b_offsets = j * (N_ROW_BLOCKS * BLOCK_M) + i * BLOCK_M + tl.arange(0, BLOCK_M)
tl.store(B_ptr + b_offsets, partial)
"""

# final_sum(task_id, A_ptr, B_ptr, C_ptr, N_ROW_BLOCKS, J, K, BLOCK_M, BLOCK_K)
#   i = task_id; pushed only after every partial_sum[i*J + j] has
#   decremented E[i] -- so we can read B with no wait loop.
_FINAL_SUM_BODY = """
i = task_id
row_offsets = i * BLOCK_M + tl.arange(0, BLOCK_M)
acc = tl.zeros((BLOCK_M,), dtype=tl.float32)
for j in tl.static_range(0, J):
    b_offsets = j * (N_ROW_BLOCKS * BLOCK_M) + i * BLOCK_M + tl.arange(0, BLOCK_M)
    acc += tl.load(B_ptr + b_offsets)
tl.store(C_ptr + row_offsets, acc)
"""


# ---------------------------------------------------------------------------
# IR construction (dynamic policy, grouped events)
# ---------------------------------------------------------------------------


def build_dynamic_event_graph(n_row_blocks: int, j_chunks: int) -> tuple[ModuleOp, GraphOp]:
    """Build the dynamic-policy event.graph for the row-sum workload.

    Layout:
        E:           shape = (n_row_blocks,), wait_count = j_chunks
        partial_sum: tile_num = (n_row_blocks * j_chunks,)
                     task k has out_edge E[k // j_chunks]
        final_sum:   tile_num = (n_row_blocks,)
                     task i has in_edge  E[i]
    """
    block = Block()
    block.add_op(
        EventTensorOp.create(
            properties={
                "sym_name": StringAttr("E"),
                "event_type": EventTensorTypeAttr([n_row_blocks]),
                "wait_count": IntegerAttr(j_chunks, IntegerType(64)),
            },
        ),
    )

    n_partials = n_row_blocks * j_chunks

    # per-producer out_edges: each partial_sum[k] decrements E[k // j_chunks]
    out_edges_per_task = [
        EventCoordAttr("E", [str(k // j_chunks)], 1) for k in range(n_partials)
    ]
    block.add_op(
        CallDeviceOp.create(
            properties={
                "device_func": SymbolRefAttr("partial_sum"),
                "task_shape": ArrayAttr([IntegerAttr(n_partials, IntegerType(64))]),
                "out_edges": ArrayAttr(out_edges_per_task),
            },
        ),
    )

    # per-consumer in_edges: final_sum[i] waits on E[i]
    in_edges_per_task = [EventCoordAttr("E", [str(i)], 1) for i in range(n_row_blocks)]
    block.add_op(
        CallDeviceOp.create(
            properties={
                "device_func": SymbolRefAttr("final_sum"),
                "task_shape": ArrayAttr([IntegerAttr(n_row_blocks, IntegerType(64))]),
                "in_edges": ArrayAttr(in_edges_per_task),
            },
        ),
    )

    sm_count = max(1, min(n_partials, 8))
    graph = GraphOp(
        sym_name="row_sum_dyn",
        policy="dynamic",
        sm_count=sm_count,
        body=Region([block]),
    )
    mod = ModuleOp([])
    mod.body.block.add_op(graph)
    return mod, graph


# ---------------------------------------------------------------------------
# Compile + run (mirrors the static example's structure)
# ---------------------------------------------------------------------------


@dataclass
class CompiledDynamicMegakernel:
    kernel_name: str
    kernel_source: str
    kernel_callable: object
    lowering: DynamicMegakernelLoweringResult
    n_row_blocks: int
    j_chunks: int
    block_m: int
    block_k: int
    sm_count: int


def compile_dynamic_megakernel(
    n_row_blocks: int = 8,
    j_chunks: int = 4,
    block_m: int = 32,
    block_k: int = 32,
) -> CompiledDynamicMegakernel:
    mod, graph = build_dynamic_event_graph(n_row_blocks, j_chunks)
    spec = DynamicMegakernelLoweringSpec(
        data_pointers=("A_ptr", "B_ptr", "C_ptr"),
        constexpr_args=("N_ROW_BLOCKS", "J", "K", "BLOCK_M", "BLOCK_K"),
        device_functions=(
            DynamicDeviceFunctionSpec(name="partial_sum", body_source=_PARTIAL_SUM_BODY),
            DynamicDeviceFunctionSpec(name="final_sum",   body_source=_FINAL_SUM_BODY),
        ),
    )
    lowering = lower_megakernel_dynamic(graph, spec=spec)

    fd, path = tempfile.mkstemp(prefix=f"{lowering.kernel_name}_", suffix=".py")
    with os.fdopen(fd, "w") as f:
        f.write(lowering.kernel_source)
    linecache.checkcache(path)
    module_spec = importlib.util.spec_from_file_location(lowering.kernel_name, path)
    if module_spec is None or module_spec.loader is None:
        raise RuntimeError(f"failed to build importlib spec for {path}")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    kernel_callable = getattr(module, lowering.kernel_name)

    return CompiledDynamicMegakernel(
        kernel_name=lowering.kernel_name,
        kernel_source=lowering.kernel_source,
        kernel_callable=kernel_callable,
        lowering=lowering,
        n_row_blocks=n_row_blocks,
        j_chunks=j_chunks,
        block_m=block_m,
        block_k=block_k,
        sm_count=int(lowering.launch_config["grid"]),
    )


def run_dynamic_megakernel(
    compiled: CompiledDynamicMegakernel,
    a: torch.Tensor,
) -> torch.Tensor:
    if a.dtype != torch.float32:
        raise TypeError(f"expected float32, got {a.dtype}")
    expected_rows = compiled.n_row_blocks * compiled.block_m
    expected_cols = compiled.j_chunks * compiled.block_k
    if a.shape != (expected_rows, expected_cols):
        raise ValueError(
            f"input shape {tuple(a.shape)} != expected "
            f"({expected_rows}, {expected_cols})"
        )
    if not a.is_cuda:
        raise RuntimeError("megakernel requires CUDA")

    device = a.device
    n_events = compiled.n_row_blocks
    n_partials = compiled.n_row_blocks * compiled.j_chunks
    total_tasks = compiled.lowering.total_tasks
    max_queue = max(total_tasks * 2, 8)

    b = torch.zeros(
        (compiled.j_chunks, compiled.n_row_blocks * compiled.block_m),
        dtype=torch.float32, device=device,
    )
    c = torch.zeros((expected_rows,), dtype=torch.float32, device=device)
    e = torch.full((n_events,), compiled.j_chunks, dtype=torch.int32, device=device)

    # Pre-seed the queue with every task with no in_edges (== producers).
    # Also mark those slots as already-valid so consumers (== first SMs to
    # pop) don't spin waiting for a publish that already happened on the host.
    queue_pool  = torch.zeros((max_queue, 2), dtype=torch.int32, device=device)
    queue_valid = torch.zeros((max_queue,),  dtype=torch.int32, device=device)
    for slot, (tid, kind) in enumerate(compiled.lowering.initial_queue):
        queue_pool[slot, 0] = tid
        queue_pool[slot, 1] = kind
        queue_valid[slot]   = 1
    queue_head = torch.zeros((1,), dtype=torch.int32, device=device)
    queue_tail = torch.tensor(
        [len(compiled.lowering.initial_queue)], dtype=torch.int32, device=device
    )

    compiled.kernel_callable[(compiled.sm_count,)](
        # data ptrs
        a, b, c,
        # event ptrs
        e,
        # queue ptrs
        queue_pool, queue_head, queue_tail, queue_valid,
        # constexprs
        compiled.n_row_blocks, compiled.j_chunks,
        a.shape[1], compiled.block_m, compiled.block_k,
        # implicit constexprs
        compiled.sm_count, total_tasks, max_queue,
        num_warps=compiled.lowering.launch_config["num_warps"],
        num_stages=compiled.lowering.launch_config["num_stages"],
    )
    torch.cuda.synchronize()

    if not bool(torch.all(e == 0)):
        raise RuntimeError(f"event counters did not drain: {e.tolist()}")
    if int(queue_head.item()) < total_tasks:
        raise RuntimeError(
            f"queue head ({int(queue_head.item())}) < total_tasks ({total_tasks})"
        )

    return c


def reference(a: torch.Tensor) -> torch.Tensor:
    return a.sum(dim=-1)


__all__ = [
    "CompiledDynamicMegakernel",
    "build_dynamic_event_graph",
    "compile_dynamic_megakernel",
    "reference",
    "run_dynamic_megakernel",
]


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise SystemExit("This example requires a CUDA device.")

    compiled = compile_dynamic_megakernel(n_row_blocks=8, j_chunks=4)
    print(f"Emitted dynamic megakernel: {compiled.kernel_name}")
    print(f"  SM_COUNT={compiled.sm_count}, TOTAL_TASKS={compiled.lowering.total_tasks}")
    print(f"  initial queue size = {len(compiled.lowering.initial_queue)} (== producers)")
    print(f"  consumer table = {compiled.lowering.consumer_table}")
    print(f"  source = {len(compiled.kernel_source)} chars")

    M = compiled.n_row_blocks * compiled.block_m
    K = compiled.j_chunks * compiled.block_k
    a = torch.randn((M, K), dtype=torch.float32, device="cuda")

    got = run_dynamic_megakernel(compiled, a)
    ref = reference(a)
    err = (got - ref).abs().max().item()
    print(f"max |got - ref| = {err:.3e} on shape ({M},{K})")
    assert err < 1e-3, "dynamic megakernel output diverges from torch.sum"
    print("PASS: emitted DYNAMIC megakernel matches torch.sum on real GPU.")
