"""Tests for Recipe IR Family B: Fact/Evidence operations.

Covers BackendAvailableOp, KernelContractOp, TransferCostOp,
LocalMemFitOp, FusibleWithOp, CalibrationOp, ExportIssueOp, GraphBreakOp.
"""

from __future__ import annotations

import io

from compgen.ir.recipe.attrs import CostAttr, DeviceRefAttr
from compgen.ir.recipe.ops_fact import (
    BackendAvailableOp,
    CalibrationOp,
    ExportIssueOp,
    FusibleWithOp,
    GraphBreakOp,
    KernelContractOp,
    LocalMemFitOp,
    TransferCostOp,
)
from xdsl.dialects.builtin import (
    ArrayAttr,
    IntegerAttr,
    IntegerType,
    StringAttr,
    SymbolRefAttr,
)
from xdsl.printer import Printer


def _print_op(op) -> str:
    buf = io.StringIO()
    Printer(stream=buf).print_op(op)
    return buf.getvalue()


# -- BackendAvailableOp -------------------------------------------------------


def test_backend_available_minimal() -> None:
    op = BackendAvailableOp.build(properties={
        "region_ref": SymbolRefAttr("r0"),
        "backend": StringAttr("triton"),
    })
    assert op.backend.data == "triton"
    assert op.confidence is None


def test_backend_available_with_confidence() -> None:
    op = BackendAvailableOp.build(properties={
        "region_ref": SymbolRefAttr("r0"),
        "backend": StringAttr("autocomp"),
        "confidence": StringAttr("high"),
    })
    assert op.confidence.data == "high"


def test_backend_available_name() -> None:
    assert BackendAvailableOp.name == "recipe.fact.backend_available"


# -- KernelContractOp ---------------------------------------------------------


def test_kernel_contract_minimal() -> None:
    op = KernelContractOp.build(properties={
        "region_ref": SymbolRefAttr("r0"),
        "op_name": StringAttr("linalg.matmul"),
    })
    assert op.op_name.data == "linalg.matmul"
    assert op.input_layouts is None
    assert op.estimated_flops is None


def test_kernel_contract_full() -> None:
    op = KernelContractOp.build(properties={
        "region_ref": SymbolRefAttr("r0"),
        "op_name": StringAttr("linalg.matmul"),
        "input_layouts": ArrayAttr([StringAttr("row_major"), StringAttr("col_major")]),
        "output_layouts": ArrayAttr([StringAttr("row_major")]),
        "supported_dtypes": ArrayAttr([StringAttr("f32"), StringAttr("f16")]),
        "estimated_flops": IntegerAttr(2_000_000, IntegerType(64)),
    })
    assert len(op.input_layouts.data) == 2
    assert op.estimated_flops.value.data == 2_000_000


def test_kernel_contract_name() -> None:
    assert KernelContractOp.name == "recipe.fact.kernel_contract"


# -- TransferCostOp -----------------------------------------------------------


def test_transfer_cost_build() -> None:
    cost = CostAttr(500, "measured")
    op = TransferCostOp.build(properties={
        "src_region": SymbolRefAttr("r0"),
        "dst_region": SymbolRefAttr("r1"),
        "cost": cost,
    })
    assert op.cost.value_us.value.data == 500
    assert op.cost.confidence.data == "measured"


def test_transfer_cost_name() -> None:
    assert TransferCostOp.name == "recipe.fact.transfer_cost"


# -- LocalMemFitOp ------------------------------------------------------------


def test_local_mem_fit_build() -> None:
    device = DeviceRefAttr(0, "gpu0")
    op = LocalMemFitOp.build(properties={
        "region_ref": SymbolRefAttr("r0"),
        "device": device,
        "fits": IntegerAttr(1, IntegerType(64)),
    })
    assert op.fits.value.data == 1
    assert op.device.device_name.data == "gpu0"


def test_local_mem_fit_name() -> None:
    assert LocalMemFitOp.name == "recipe.fact.local_mem_fit"


# -- FusibleWithOp ------------------------------------------------------------


def test_fusible_with_minimal() -> None:
    op = FusibleWithOp.build(properties={
        "region_a": SymbolRefAttr("r0"),
        "region_b": SymbolRefAttr("r1"),
    })
    assert op.fusion_kind is None


def test_fusible_with_kind() -> None:
    op = FusibleWithOp.build(properties={
        "region_a": SymbolRefAttr("r0"),
        "region_b": SymbolRefAttr("r1"),
        "fusion_kind": StringAttr("producer_consumer"),
    })
    assert op.fusion_kind.data == "producer_consumer"


def test_fusible_with_name() -> None:
    assert FusibleWithOp.name == "recipe.fact.fusible_with"


# -- CalibrationOp ------------------------------------------------------------


def test_calibration_build() -> None:
    device = DeviceRefAttr(0, "gpu0")
    op = CalibrationOp.build(properties={
        "region_ref": SymbolRefAttr("r0"),
        "measured_latency_us": IntegerAttr(150, IntegerType(64)),
        "device": device,
    })
    assert op.measured_latency_us.value.data == 150


def test_calibration_name() -> None:
    assert CalibrationOp.name == "recipe.fact.calibration"


# -- ExportIssueOp ------------------------------------------------------------


def test_export_issue_build() -> None:
    op = ExportIssueOp.build(properties={
        "description": StringAttr("dynamic batch dim unsupported"),
        "severity": StringAttr("error"),
    })
    assert op.description.data == "dynamic batch dim unsupported"
    assert op.severity.data == "error"


def test_export_issue_name() -> None:
    assert ExportIssueOp.name == "recipe.fact.export_issue"


# -- GraphBreakOp -------------------------------------------------------------


def test_graph_break_build() -> None:
    op = GraphBreakOp.build(properties={
        "location": StringAttr("line 42"),
        "reason": StringAttr("data-dependent control flow"),
    })
    assert op.reason.data == "data-dependent control flow"


def test_graph_break_name() -> None:
    assert GraphBreakOp.name == "recipe.fact.graph_break"


def test_graph_break_printable() -> None:
    op = GraphBreakOp.build(properties={
        "location": StringAttr("line 1"),
        "reason": StringAttr("unsupported op"),
    })
    text = _print_op(op)
    assert "recipe.fact.graph_break" in text
