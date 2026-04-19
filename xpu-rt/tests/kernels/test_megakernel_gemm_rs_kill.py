"""End-to-end kill test for the ETC megakernel pipeline (Phase A).

Walks every layer of the integration on a synthetic GEMM+ReduceScatter-
style 4-task workload:

    1. Build an event.graph with two CallDeviceOps + one EventTensor.
    2. Run StaticMegakernelSchedule (Algorithm 1 of the ETC paper).
    3. Lower to persistent Triton via lower_megakernel.
    4. Compile a *real, hand-substituted* Triton persistent kernel that
       follows the same task-queue + atomic-event protocol.
    5. Execute against a kernel-by-kernel reference on the local GPU.
    6. Compare outputs via the existing
       :func:`compgen.semantic.verify.harness.verify_callable_against_reference`
       and assert numerical match.

The Triton kernel is hand-written rather than auto-generated from the
emitter's source string -- the emitter's per-device-function bodies are
intentionally left as ``pass`` stubs in Phase A (real bodies come from
``lower_tile_to_triton``).  This kill test exercises the *protocol*
(persistent grid, per-SM queue, atomic event counters) on real hardware,
which is the load-bearing claim of the ETC paper.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
triton = pytest.importorskip("triton")
import triton.language as tl
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
from compgen.ir.tile.lower_megakernel import lower_megakernel
from compgen.semantic.verify.harness import verify_callable_against_reference


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="kill test requires a CUDA device"
)


# ---------------------------------------------------------------------------
# Hand-written persistent megakernel (mirrors lower_megakernel.py output)
# ---------------------------------------------------------------------------


@triton.jit
def _megakernel_partial_then_reduce(
    A_ptr,
    out_ptr,
    E_ptr,
    QUEUE_PTR,         # int32, shape (SM_COUNT, MAX_QLEN, 2)
    QUEUE_LEN_PTR,     # int32, shape (SM_COUNT,)
    N_TILES: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    MAX_QLEN: tl.constexpr,
):
    """Persistent megakernel: 4 partial_sum tasks + 4 final_sum tasks.

    Each program (== one SM) walks its precomputed queue.  Task kind 0
    is partial_sum (writes its tile of out and notifies E[i]); task
    kind 1 is final_sum (waits on E[i] and reads its tile back -- in
    this synthetic kill test, "final_sum" is just a copy that proves
    the wait actually fires).
    """
    sm_id = tl.program_id(0)
    qlen = tl.load(QUEUE_LEN_PTR + sm_id)
    task_idx = 0
    while task_idx < qlen:
        task_id = tl.load(QUEUE_PTR + (sm_id * MAX_QLEN + task_idx) * 2 + 0)
        task_kind = tl.load(QUEUE_PTR + (sm_id * MAX_QLEN + task_idx) * 2 + 1)
        if task_kind == 0:
            # partial_sum -- compute tile sum, store, notify event
            offsets = task_id * TILE_SIZE + tl.arange(0, TILE_SIZE)
            mask = offsets < N_TILES * TILE_SIZE
            x = tl.load(A_ptr + offsets, mask=mask, other=0.0)
            tl.store(out_ptr + offsets, x * 2.0, mask=mask)
            tl.atomic_add(E_ptr + task_id, -1)
        else:
            # final_sum -- spin-wait on E[i], then read+rewrite tile
            counter = tl.atomic_or(E_ptr + task_id, 0)
            while counter > 0:
                counter = tl.atomic_or(E_ptr + task_id, 0)
            offsets = task_id * TILE_SIZE + tl.arange(0, TILE_SIZE)
            mask = offsets < N_TILES * TILE_SIZE
            y = tl.load(out_ptr + offsets, mask=mask, other=0.0)
            tl.store(out_ptr + offsets, y + 1.0, mask=mask)
        task_idx += 1


def _flatten_queue(per_sm_order: dict[int, list[str]], task_kinds: dict[str, int], max_qlen: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Flatten the per-SM queue into a (SM_COUNT, MAX_QLEN, 2) int32 tensor + lens."""
    sm_count = max(per_sm_order) + 1 if per_sm_order else 0
    queue = torch.zeros((sm_count, max_qlen, 2), dtype=torch.int32, device="cuda")
    lens = torch.zeros((sm_count,), dtype=torch.int32, device="cuda")
    for sm, tids in per_sm_order.items():
        for slot, tid in enumerate(tids):
            task_id_int = int(tid.split(":")[1])
            queue[sm, slot, 0] = task_id_int
            queue[sm, slot, 1] = task_kinds[tid]
        lens[sm] = len(tids)
    return queue, lens


def _build_event_graph_module(sm_count: int = 4) -> tuple[ModuleOp, GraphOp]:
    block = Block()
    block.add_op(
        EventTensorOp.create(
            properties={
                "sym_name": StringAttr("E"),
                "event_type": EventTensorTypeAttr([4]),
                "wait_count": IntegerAttr(1, IntegerType(64)),
            },
        ),
    )
    block.add_op(
        CallDeviceOp.create(
            properties={
                "device_func": SymbolRefAttr("partial_sum"),
                "task_shape": ArrayAttr([IntegerAttr(4, IntegerType(64))]),
                "out_edges": ArrayAttr(
                    [EventCoordAttr("E", [str(i)], 1) for i in range(4)],
                ),
            },
        ),
    )
    block.add_op(
        CallDeviceOp.create(
            properties={
                "device_func": SymbolRefAttr("final_sum"),
                "task_shape": ArrayAttr([IntegerAttr(4, IntegerType(64))]),
                "in_edges": ArrayAttr(
                    [EventCoordAttr("E", [str(i)], 1) for i in range(4)],
                ),
            },
        ),
    )
    graph = GraphOp(sym_name="mm_rs", policy="static", sm_count=sm_count, body=Region([block]))
    mod = ModuleOp([])
    mod.body.block.add_op(graph)
    return mod, graph


# ---------------------------------------------------------------------------
# Reference (kernel-by-kernel) implementation
# ---------------------------------------------------------------------------


def _reference(a: torch.Tensor) -> torch.Tensor:
    """Two-stage reference matching the megakernel's net effect: out = 2*a + 1."""
    out = a * 2.0
    out = out + 1.0
    return out


# ---------------------------------------------------------------------------
# Kill test
# ---------------------------------------------------------------------------


def test_phase_a_megakernel_pipeline_end_to_end_on_gpu(tmp_path: Path) -> None:
    sm_count = 4
    n_tiles = 4
    tile_size = 32
    n = n_tiles * tile_size

    # 1) Build IR + run static-schedule pass.
    mod, graph = _build_event_graph_module(sm_count=sm_count)
    StaticMegakernelSchedule().run(mod)
    schedule = json.loads(graph.attributes["compgen.static_schedule"].data)
    assert schedule["status"] == "ok", schedule

    # 2) Run the persistent-Triton emitter (validates the IR-level layer).
    lowered = lower_megakernel(graph)
    assert lowered.kernel_name == "megakernel_mm_rs"
    assert "@triton.jit" in lowered.kernel_source
    assert lowered.launch_config["grid"] == sm_count
    assert lowered.event_layout[0]["size"] == n_tiles

    # 3) Flatten the per-SM queue + task-kind table for the Triton kernel.
    task_kinds: dict[str, int] = {}
    for sm_str, q in schedule["per_sm_order"].items():
        for tid in q:
            task_kinds[tid] = 0 if tid.startswith("partial_sum:") else 1
    per_sm_int = {int(sm): tids for sm, tids in schedule["per_sm_order"].items()}
    max_qlen = max((len(q) for q in per_sm_int.values()), default=1)
    queue, lens = _flatten_queue(per_sm_int, task_kinds, max_qlen)

    # 4) Allocate event tensor + I/O on the GPU and seed the wait counts.
    E = torch.full((n_tiles,), 1, dtype=torch.int32, device="cuda")
    a = torch.randn((n,), dtype=torch.float32, device="cuda")
    out = torch.zeros_like(a)

    # 5) Run the megakernel (one program per SM-equivalent).
    def run_megakernel() -> torch.Tensor:
        E.fill_(1)
        out.zero_()
        _megakernel_partial_then_reduce[(sm_count,)](
            a, out, E, queue, lens,
            N_TILES=n_tiles, TILE_SIZE=tile_size, MAX_QLEN=max_qlen,
        )
        torch.cuda.synchronize()
        return out

    # 6) Compare against the kernel-by-kernel reference via the harness.
    run = verify_callable_against_reference(
        name="phase_a_gemm_rs_kill",
        ref_fn=lambda: _reference(a),
        got_fn=run_megakernel,
        out_dir=tmp_path,
        atol=1e-5,
        rtol=1e-5,
    )

    # Assertions: numerical match + report file written.
    assert run.passed, [c for c in run.comparisons if not c.passed]
    report_path = tmp_path / "verification.json"
    assert report_path.exists()
    saved = json.loads(report_path.read_text())
    assert saved["passed"] is True

    # Side proof that every wait actually fired (counters drained to zero).
    assert torch.all(E == 0), f"event counters not drained: {E.tolist()}"
