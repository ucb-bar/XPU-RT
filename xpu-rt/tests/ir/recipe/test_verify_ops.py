"""Tests for Recipe IR Family E: Verification Obligation operations.

Covers RequireDiffTestOp, RequireTranslationValidationOp,
RequireLayoutInvariantOp, RequireMemoryBoundOp, RequireCheckFileOp,
RequireProfileBudgetOp.
"""

from __future__ import annotations

import io

from compgen.ir.recipe.attrs import DeviceRefAttr
from compgen.ir.recipe.ops_verify import (
    RequireCheckFileOp,
    RequireDiffTestOp,
    RequireLayoutInvariantOp,
    RequireMemoryBoundOp,
    RequireProfileBudgetOp,
    RequireTranslationValidationOp,
)
from xdsl.dialects.builtin import IntegerAttr, IntegerType, StringAttr, SymbolRefAttr
from xdsl.printer import Printer


def _i64(val: int) -> IntegerAttr:
    return IntegerAttr(val, IntegerType(64))


def _print_op(op) -> str:
    buf = io.StringIO()
    Printer(stream=buf).print_op(op)
    return buf.getvalue()


# -- RequireDiffTestOp --------------------------------------------------------


def test_diff_test_minimal() -> None:
    op = RequireDiffTestOp.build(properties={
        "region_ref": SymbolRefAttr("r0"),
    })
    assert op.tolerance is None


def test_diff_test_with_tolerance() -> None:
    op = RequireDiffTestOp.build(properties={
        "region_ref": SymbolRefAttr("r0"),
        "tolerance": _i64(4),
    })
    assert op.tolerance.value.data == 4


def test_diff_test_name() -> None:
    assert RequireDiffTestOp.name == "recipe.require_diff_test"


def test_diff_test_verify_ok() -> None:
    op = RequireDiffTestOp.build(properties={
        "region_ref": SymbolRefAttr("r0"),
    })
    op.verify()


# -- RequireTranslationValidationOp -------------------------------------------


def test_tv_minimal() -> None:
    op = RequireTranslationValidationOp.build(properties={
        "region_ref": SymbolRefAttr("r0"),
    })
    assert op.source_op is None
    assert op.target_op is None


def test_tv_full() -> None:
    op = RequireTranslationValidationOp.build(properties={
        "region_ref": SymbolRefAttr("r0"),
        "source_op": StringAttr("linalg.matmul"),
        "target_op": StringAttr("triton_kernel"),
    })
    assert op.source_op.data == "linalg.matmul"
    assert op.target_op.data == "triton_kernel"


def test_tv_name() -> None:
    assert RequireTranslationValidationOp.name == "recipe.require_translation_validation"


# -- RequireLayoutInvariantOp -------------------------------------------------


def test_layout_invariant_build() -> None:
    op = RequireLayoutInvariantOp.build(properties={
        "region_ref": SymbolRefAttr("r0"),
        "expected_layout": StringAttr("NCHW"),
    })
    assert op.expected_layout.data == "NCHW"


def test_layout_invariant_name() -> None:
    assert RequireLayoutInvariantOp.name == "recipe.require_layout_invariant"


def test_layout_invariant_verify_ok() -> None:
    op = RequireLayoutInvariantOp.build(properties={
        "region_ref": SymbolRefAttr("r0"),
        "expected_layout": StringAttr("NHWC"),
    })
    op.verify()


# -- RequireMemoryBoundOp -----------------------------------------------------


def test_memory_bound_minimal() -> None:
    op = RequireMemoryBoundOp.build(properties={
        "region_ref": SymbolRefAttr("r0"),
        "max_bytes": _i64(1_073_741_824),
    })
    assert op.max_bytes.value.data == 1_073_741_824
    assert op.device is None


def test_memory_bound_with_device() -> None:
    device = DeviceRefAttr(0, "gpu0")
    op = RequireMemoryBoundOp.build(properties={
        "region_ref": SymbolRefAttr("r0"),
        "max_bytes": _i64(1024),
        "device": device,
    })
    assert op.device.device_name.data == "gpu0"


def test_memory_bound_name() -> None:
    assert RequireMemoryBoundOp.name == "recipe.require_memory_bound"


# -- RequireCheckFileOp -------------------------------------------------------


def test_check_file_build() -> None:
    op = RequireCheckFileOp.build(properties={
        "check_file_path": StringAttr("tests/checks/matmul.check"),
    })
    assert op.check_file_path.data == "tests/checks/matmul.check"


def test_check_file_name() -> None:
    assert RequireCheckFileOp.name == "recipe.require_check_file"


def test_check_file_printable() -> None:
    op = RequireCheckFileOp.build(properties={
        "check_file_path": StringAttr("foo.check"),
    })
    text = _print_op(op)
    assert "recipe.require_check_file" in text


# -- RequireProfileBudgetOp ---------------------------------------------------


def test_profile_budget_minimal() -> None:
    op = RequireProfileBudgetOp.build(properties={
        "region_ref": SymbolRefAttr("r0"),
        "max_latency_us": _i64(5000),
    })
    assert op.max_latency_us.value.data == 5000
    assert op.device is None


def test_profile_budget_with_device() -> None:
    device = DeviceRefAttr(1, "tpu0")
    op = RequireProfileBudgetOp.build(properties={
        "region_ref": SymbolRefAttr("r0"),
        "max_latency_us": _i64(2000),
        "device": device,
    })
    assert op.device.device_name.data == "tpu0"


def test_profile_budget_name() -> None:
    assert RequireProfileBudgetOp.name == "recipe.require_profile_budget"
