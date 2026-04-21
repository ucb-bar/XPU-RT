"""Real  example: row-sum via Event Tensor megakernel.

Mirrors **Figure 3 of the Event Tensor Compiler paper** (Jin et al.,
MLSys '26) exactly: two-stage row reduction over a matrix ``A`` of
shape ``(n*32, K)``, split-K with ``J = K // 32`` chunks, coordinated
by a per-(row-block, k-chunk) Event Tensor.

This module is the *real* end-to-end  validation:

    1. Construct an ``event.graph`` with the partial_sum and final_sum
       device functions and the inter-task Event Tensor.
    2. Run :class:`StaticMegakernelSchedule` (Algorithm 1) to assign
       tasks to SMs.
    3. Lower to **persistent Triton source** with REAL device-function
       bodies via :func:`lower_megakernel` + :class:`MegakernelLoweringSpec`.
    4. Compile the emitted source string with ``triton.jit`` and launch
       the resulting persistent kernel on the GPU.
    5. Compare the kernel's output to ``A.sum(dim=-1)`` (PyTorch eager).

There are no hand-written Triton kernels in this example -- every line
of GPU code that runs comes out of the CompGen emitter.
"""

from __future__ import annotations

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
from compgen.ir.payload.passes.megakernel_static_schedule import (
    StaticMegakernelSchedule,
)
from compgen.ir.tile.lower_megakernel import (
    DeviceFunctionSpec,
    MegakernelLoweringResult,
    MegakernelLoweringSpec,
    lower_megakernel,
)


# ---------------------------------------------------------------------------
# Triton bodies for the two device functions (string source -> emitter input)
# ---------------------------------------------------------------------------

# partial_sum(task_id, A_ptr, B_ptr, C_ptr, E_ptr, N_ROW_BLOCKS, J,
#             K, BLOCK_M, BLOCK_K)
#
# task_id ranges over [0, N_ROW_BLOCKS * J).  Decode as:
#     i = task_id // J         -- row-block index
#     j = task_id %  J         -- k-chunk index
#
# Computes:
#     B[j, i*32:(i+1)*32] = sum_{k in [j*32, (j+1)*32)} A[i*32:(i+1)*32, k]
# Then notifies E[i * J + j].
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

# event.notify -- decrement counter at E[i * J + j]
tl.atomic_add(E_ptr + (i * J + j), -1)
"""


# final_sum(task_id, A_ptr, B_ptr, C_ptr, E_ptr, N_ROW_BLOCKS, J,
#           K, BLOCK_M, BLOCK_K)
#
# task_id ranges over [0, N_ROW_BLOCKS).  task_id == i.
#
# Waits on E[i * J + j] for every j in [0, J).
# Reads B[:, i*32:(i+1)*32] (shape (J, 32)) and writes
# C[i*32:(i+1)*32] = sum over j.
_FINAL_SUM_BODY = """
i = task_id

# event.wait on E[i, j] for every k-chunk j
for j in tl.static_range(0, J):
    counter = tl.atomic_or(E_ptr + (i * J + j), 0)
    while counter > 0:
        counter = tl.atomic_or(E_ptr + (i * J + j), 0)

row_offsets = i * BLOCK_M + tl.arange(0, BLOCK_M)
acc = tl.zeros((BLOCK_M,), dtype=tl.float32)
for j in tl.static_range(0, J):
    b_offsets = j * (N_ROW_BLOCKS * BLOCK_M) + i * BLOCK_M + tl.arange(0, BLOCK_M)
    acc += tl.load(B_ptr + b_offsets)
tl.store(C_ptr + row_offsets, acc)
"""


# ---------------------------------------------------------------------------
# IR construction
# ---------------------------------------------------------------------------


def build_event_graph(n_row_blocks: int, j_chunks: int) -> tuple[ModuleOp, GraphOp]:
    """Build the row-sum event.graph (paper Fig. 3).

    Layout:
        partial_sum: tile_num = (n_row_blocks * j_chunks,)
        final_sum:   tile_num = (n_row_blocks,)
        E:           shape    = (n_row_blocks * j_chunks,)
                     wait_count = 1   (one notify per slot)
    Edges:
        partial_sum task k notifies E[k]
        final_sum task i waits on E[i*J .. (i+1)*J)
    """
    block = Block()
    n_events = n_row_blocks * j_chunks
    block.add_op(
        EventTensorOp.create(
            properties={
                "sym_name": StringAttr("E"),
                "event_type": EventTensorTypeAttr([n_events]),
                "wait_count": IntegerAttr(1, IntegerType(64)),
            },
        ),
    )
    block.add_op(
        CallDeviceOp.create(
            properties={
                "device_func": SymbolRefAttr("partial_sum"),
                "task_shape": ArrayAttr([IntegerAttr(n_events, IntegerType(64))]),
                "out_edges": ArrayAttr(
                    [EventCoordAttr("E", [str(k)], 1) for k in range(n_events)],
                ),
            },
        ),
    )
    block.add_op(
        CallDeviceOp.create(
            properties={
                "device_func": SymbolRefAttr("final_sum"),
                "task_shape": ArrayAttr([IntegerAttr(n_row_blocks, IntegerType(64))]),
                "in_edges": ArrayAttr(
                    [EventCoordAttr("E", [str(k)], 1) for k in range(n_events)],
                ),
            },
        ),
    )
    sm_count = max(1, min(n_events, 8))
    graph = GraphOp(
        sym_name="row_sum",
        policy="static",
        sm_count=sm_count,
        body=Region([block]),
    )
    mod = ModuleOp([])
    mod.body.block.add_op(graph)
    return mod, graph


# ---------------------------------------------------------------------------
# Emit + compile + launch
# ---------------------------------------------------------------------------


@dataclass
class CompiledMegakernel:
    """Bundle of artifacts produced by compiling an event.graph end-to-end."""

    kernel_name: str
    kernel_source: str
    kernel_callable: object  # the @triton.jit function (Python object)
    lowering: MegakernelLoweringResult
    n_row_blocks: int
    j_chunks: int
    block_m: int
    block_k: int
    sm_count: int
    max_qlen: int
    device_function_table: dict[int, str]


def compile_megakernel(
    n_row_blocks: int = 4,
    j_chunks: int = 4,
    block_m: int = 32,
    block_k: int = 32,
) -> CompiledMegakernel:
    """End-to-end: IR -> static schedule -> Triton source -> @triton.jit.

    Returns a :class:`CompiledMegakernel` whose ``kernel_callable`` is the
    actual Triton-compiled function ready to launch on the GPU.
    """
    mod, graph = build_event_graph(n_row_blocks, j_chunks)
    StaticMegakernelSchedule().run(mod)

    spec = MegakernelLoweringSpec(
        data_pointers=("A_ptr", "B_ptr", "C_ptr"),
        constexpr_args=("N_ROW_BLOCKS", "J", "K", "BLOCK_M", "BLOCK_K"),
        device_functions=(
            DeviceFunctionSpec(name="partial_sum", body_source=_PARTIAL_SUM_BODY),
            DeviceFunctionSpec(name="final_sum", body_source=_FINAL_SUM_BODY),
        ),
    )
    lowering = lower_megakernel(graph, spec=spec)
    if not lowering.kernel_source:
        raise RuntimeError(f"emitter rejected the graph: {lowering.diagnostics}")

    # Triton's @jit decorator uses inspect.getsource, which requires the
    # function to live in a file linecache can read.  Materialise the
    # emitted source to a real temp file and import it.
    import importlib.util
    import linecache
    import os
    import tempfile

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

    sm_count = int(lowering.launch_config["grid"])
    max_qlen = max(
        (len(q) for q in lowering.task_queue.values()),
        default=1,
    )

    return CompiledMegakernel(
        kernel_name=lowering.kernel_name,
        kernel_source=lowering.kernel_source,
        kernel_callable=kernel_callable,
        lowering=lowering,
        n_row_blocks=n_row_blocks,
        j_chunks=j_chunks,
        block_m=block_m,
        block_k=block_k,
        sm_count=sm_count,
        max_qlen=max_qlen,
        device_function_table=lowering.device_function_table,
    )


def _flatten_queue(
    compiled: CompiledMegakernel,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Flatten the per-SM queue into the (SM_COUNT, MAX_QLEN, 2) int32 layout."""
    queue = torch.zeros(
        (compiled.sm_count, compiled.max_qlen, 2),
        dtype=torch.int32,
        device=device,
    )
    lens = torch.zeros((compiled.sm_count,), dtype=torch.int32, device=device)
    for sm, entries in compiled.lowering.task_queue.items():
        for slot, (tid, kind) in enumerate(entries):
            task_id_int = int(tid.split(":")[1])
            queue[sm, slot, 0] = task_id_int
            queue[sm, slot, 1] = kind
        lens[sm] = len(entries)
    return queue, lens


def run_megakernel(
    compiled: CompiledMegakernel,
    a: torch.Tensor,
) -> torch.Tensor:
    """Launch the emitted megakernel on ``a`` and return the row-sum result.

    ``a`` must have shape ``(n_row_blocks * BLOCK_M, J * BLOCK_K)``.
    """
    if a.dtype != torch.float32:
        raise TypeError(f"expected float32 input, got {a.dtype}")
    expected_rows = compiled.n_row_blocks * compiled.block_m
    expected_cols = compiled.j_chunks * compiled.block_k
    if a.shape != (expected_rows, expected_cols):
        raise ValueError(
            f"input shape {tuple(a.shape)} != expected "
            f"({expected_rows}, {expected_cols})"
        )
    if not a.is_cuda:
        raise RuntimeError("megakernel requires a CUDA tensor")

    device = a.device
    n_events = compiled.n_row_blocks * compiled.j_chunks

    # Workspace and output buffers.
    b = torch.zeros(
        (compiled.j_chunks, compiled.n_row_blocks * compiled.block_m),
        dtype=torch.float32,
        device=device,
    )
    c = torch.zeros((expected_rows,), dtype=torch.float32, device=device)
    e = torch.full((n_events,), 1, dtype=torch.int32, device=device)

    queue, lens = _flatten_queue(compiled, device)

    compiled.kernel_callable[(compiled.sm_count,)](
        # data pointers
        a, b, c,
        # event pointers
        e,
        # queue + lens
        queue, lens,
        # constexpr args (in spec.constexpr_args order)
        compiled.n_row_blocks, compiled.j_chunks,
        a.shape[1], compiled.block_m, compiled.block_k,
        # implicit constexprs
        compiled.sm_count, compiled.max_qlen,
        num_warps=compiled.lowering.launch_config["num_warps"],
        num_stages=compiled.lowering.launch_config["num_stages"],
    )
    torch.cuda.synchronize()

    # Sanity post-condition: every event counter must have drained to zero.
    if not bool(torch.all(e == 0)):
        raise RuntimeError(f"event counters did not drain: {e.tolist()}")

    return c


def reference(a: torch.Tensor) -> torch.Tensor:
    """PyTorch eager reference for the row-sum workload."""
    return a.sum(dim=-1)


__all__ = [
    "CompiledMegakernel",
    "build_event_graph",
    "compile_megakernel",
    "reference",
    "run_megakernel",
]


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise SystemExit("This example requires a CUDA device.")

    compiled = compile_megakernel(n_row_blocks=8, j_chunks=4)
    print(f"Emitted kernel: {compiled.kernel_name} ({len(compiled.kernel_source)} chars)")
    print(f"  grid = SM_COUNT = {compiled.sm_count}, max_qlen = {compiled.max_qlen}")
    print(f"  task table per SM: {compiled.lowering.task_queue}")

    M = compiled.n_row_blocks * compiled.block_m
    K = compiled.j_chunks * compiled.block_k
    a = torch.randn((M, K), dtype=torch.float32, device="cuda")

    got = run_megakernel(compiled, a)
    ref = reference(a)
    err = (got - ref).abs().max().item()
    print(f"max |got - ref| = {err:.3e}  on shape ({M},{K})")
    assert err < 1e-3, "row-sum megakernel does not match PyTorch reference"
    print("PASS: emitted megakernel matches torch.sum on real GPU.")
