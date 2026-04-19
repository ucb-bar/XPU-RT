"""Tests for the persistent-Triton megakernel emitter."""

from __future__ import annotations

import pytest
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
    MegakernelLoweringResult,
    lower_megakernel,
)


def _build_gemm_rs(sm_count: int = 4) -> tuple[ModuleOp, GraphOp]:
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
                "out_edges": ArrayAttr([EventCoordAttr("E", [str(i)], 1) for i in range(4)]),
            },
        ),
    )
    block.add_op(
        CallDeviceOp.create(
            properties={
                "device_func": SymbolRefAttr("final_sum"),
                "task_shape": ArrayAttr([IntegerAttr(4, IntegerType(64))]),
                "in_edges": ArrayAttr([EventCoordAttr("E", [str(i)], 1) for i in range(4)]),
            },
        ),
    )
    graph = GraphOp(sym_name="mm_rs", policy="static", sm_count=sm_count, body=Region([block]))
    mod = ModuleOp([])
    mod.body.block.add_op(graph)
    return mod, graph


# ---------------------------------------------------------------------------
# Preconditions
# ---------------------------------------------------------------------------


def test_lowering_requires_static_schedule_annotation() -> None:
    _, graph = _build_gemm_rs()
    with pytest.raises(ValueError, match="static_schedule"):
        lower_megakernel(graph)


def test_lowering_returns_diagnostic_when_schedule_was_rejected() -> None:
    _, graph = _build_gemm_rs()
    graph.attributes["compgen.static_schedule"] = StringAttr(
        '{"status": "rejected", "errors": ["bogus"]}'
    )
    result = lower_megakernel(graph)
    assert result.kernel_source == ""
    assert any("rejected" in d for d in result.diagnostics)


# ---------------------------------------------------------------------------
# End-to-end lowering surface
# ---------------------------------------------------------------------------


def test_lowering_produces_named_persistent_kernel() -> None:
    mod, graph = _build_gemm_rs(sm_count=4)
    StaticMegakernelSchedule().run(mod)
    result = lower_megakernel(graph)
    assert isinstance(result, MegakernelLoweringResult)
    assert result.kernel_name == "megakernel_mm_rs"
    src = result.kernel_source
    assert "@triton.jit" in src
    assert "def megakernel_mm_rs(" in src
    assert "tl.program_id(0)" in src
    assert "while task_idx < qlen" in src


def test_lowering_grid_matches_sm_count() -> None:
    mod, graph = _build_gemm_rs(sm_count=8)
    StaticMegakernelSchedule().run(mod)
    result = lower_megakernel(graph)
    assert result.launch_config["grid"] == 8


def test_lowering_emits_event_pointer_arg_per_event() -> None:
    mod, graph = _build_gemm_rs()
    StaticMegakernelSchedule().run(mod)
    result = lower_megakernel(graph)
    assert "E_ptr" in result.kernel_source
    assert result.event_layout[0]["name"] == "E"
    assert result.event_layout[0]["size"] == 4


def test_lowering_emits_atomic_notify_and_wait_helpers() -> None:
    mod, graph = _build_gemm_rs()
    StaticMegakernelSchedule().run(mod)
    src = lower_megakernel(graph).kernel_source
    assert "_event_notify" in src
    assert "tl.atomic_add" in src
    assert "_event_wait" in src
    assert "tl.atomic_or" in src


def test_lowering_emits_per_device_function_stubs() -> None:
    mod, graph = _build_gemm_rs()
    StaticMegakernelSchedule().run(mod)
    src = lower_megakernel(graph).kernel_source
    assert "_run_partial_sum" in src
    assert "_run_final_sum" in src


def test_lowering_task_queue_partitions_all_tasks_across_sms() -> None:
    mod, graph = _build_gemm_rs(sm_count=4)
    StaticMegakernelSchedule().run(mod)
    result = lower_megakernel(graph)
    flat = [tid for q in result.task_queue.values() for tid, _ in q]
    assert len(flat) == 8
    assert len(set(flat)) == 8  # no duplicates
    # Both functions appear.
    assert any(tid.startswith("partial_sum:") for tid in flat)
    assert any(tid.startswith("final_sum:") for tid in flat)


def test_lowering_kernel_source_is_syntactically_valid_python() -> None:
    """Compile the emitted source as Python (Triton decorators no-op when
    triton isn't actually loading the kernel) -- catches indentation /
    syntax slips early."""
    import ast

    mod, graph = _build_gemm_rs()
    StaticMegakernelSchedule().run(mod)
    src = lower_megakernel(graph).kernel_source
    ast.parse(src)
