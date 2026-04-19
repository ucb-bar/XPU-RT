"""Tests for the MegakernelProvider (Phase A)."""

from __future__ import annotations

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
from compgen.kernels.provider import KernelContract, KernelProvider, SearchBudget
from compgen.kernels.providers.megakernel import MegakernelProvider
from compgen.kernels.selector import KernelStrategy


def _scheduled_graph(sm_count: int = 4) -> GraphOp:
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
    StaticMegakernelSchedule().run(mod)
    return graph


def test_provider_satisfies_kernelprovider_protocol() -> None:
    p = MegakernelProvider()
    assert isinstance(p, KernelProvider)


def test_provider_rejects_when_no_graph_lookup() -> None:
    p = MegakernelProvider()
    assert not p.accepts_contract(KernelContract(region_id="anything"))


def test_provider_accepts_known_region() -> None:
    graph = _scheduled_graph()
    p = MegakernelProvider(graph_lookup=lambda rid: graph if rid == "mm_rs" else None)
    assert p.accepts_contract(KernelContract(region_id="mm_rs"))
    assert not p.accepts_contract(KernelContract(region_id="other"))


def test_provider_returns_triton_source_on_known_region() -> None:
    graph = _scheduled_graph()
    p = MegakernelProvider(graph_lookup=lambda rid: graph if rid == "mm_rs" else None)
    result = p.search(KernelContract(region_id="mm_rs"), SearchBudget())
    assert result.found
    assert result.language == "triton"
    assert "@triton.jit" in result.kernel_code
    assert result.metadata["kernel_name"] == "megakernel_mm_rs"


def test_provider_returns_not_found_on_unknown_region() -> None:
    graph = _scheduled_graph()
    p = MegakernelProvider(graph_lookup=lambda rid: graph if rid == "mm_rs" else None)
    result = p.search(KernelContract(region_id="other"), SearchBudget())
    assert not result.found
    assert "no event.graph" in result.metadata["reason"]


def test_provider_returns_not_found_when_graph_unscheduled() -> None:
    """Lowering an unscheduled graph raises ValueError; provider must
    surface that as found=False rather than crashing."""
    block = Block()
    block.add_op(
        EventTensorOp.create(
            properties={
                "sym_name": StringAttr("E"),
                "event_type": EventTensorTypeAttr([1]),
                "wait_count": IntegerAttr(1, IntegerType(64)),
            },
        ),
    )
    raw_graph = GraphOp(sym_name="g", policy="static", sm_count=1, body=Region([block]))
    p = MegakernelProvider(graph_lookup=lambda _: raw_graph)
    result = p.search(KernelContract(region_id="g"), SearchBudget())
    assert not result.found
    assert "lower_megakernel" in result.metadata["reason"]


def test_provider_exports_megakernel_layout_after_search() -> None:
    graph = _scheduled_graph()
    p = MegakernelProvider(graph_lookup=lambda rid: graph)
    p.search(KernelContract(region_id="mm_rs"), SearchBudget())
    exports = p.export_knowledge()
    assert len(exports) == 1
    assert exports[0].kind == "megakernel_layout"
    assert exports[0].metadata["task_count"] == 8


def test_kernel_strategy_enum_includes_megakernel_variants() -> None:
    assert KernelStrategy.MEGAKERNEL_STATIC.value == "megakernel_static"
    assert KernelStrategy.MEGAKERNEL_DYNAMIC.value == "megakernel_dynamic"
