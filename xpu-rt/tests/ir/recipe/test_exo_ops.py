"""Tests for Recipe IR Exo-related operations.

Covers RequestExoKernelOp and SelectExoScheduleLibOp: construction,
verify_(), and lowering to kernel jobs.
"""

from __future__ import annotations

import pytest
from compgen.ir.recipe.attrs import ProvenanceAttr
from compgen.ir.recipe.ops_candidate import RequestExoKernelOp, SelectExoScheduleLibOp
from xdsl.dialects.builtin import IntegerAttr, IntegerType, StringAttr, SymbolRefAttr
from xdsl.utils.exceptions import VerifyException


def _i64(val: int) -> IntegerAttr:
    return IntegerAttr(val, IntegerType(64))


# -- RequestExoKernelOp --------------------------------------------------------


def test_request_exo_kernel_build_minimal() -> None:
    """RequestExoKernelOp can be built with required properties only."""
    op = RequestExoKernelOp.build(
        properties={
            "region_ref": SymbolRefAttr("matmul_0"),
            "search_budget": _i64(10),
        }
    )
    assert op.region_ref.root_reference.data == "matmul_0"
    assert op.search_budget.value.data == 10
    assert op.schedule_lib is None
    assert op.target_kit is None
    assert op.kernel_family is None


def test_request_exo_kernel_build_full() -> None:
    """RequestExoKernelOp can be built with all optional properties."""
    prov = ProvenanceAttr("agent", 2)
    op = RequestExoKernelOp.build(
        properties={
            "region_ref": SymbolRefAttr("conv_1"),
            "search_budget": _i64(20),
            "schedule_lib": StringAttr("x86_avx2"),
            "target_kit": StringAttr("avx2_kit"),
            "kernel_family": StringAttr("conv2d"),
            "provenance": prov,
        }
    )
    assert op.schedule_lib.data == "x86_avx2"
    assert op.target_kit.data == "avx2_kit"
    assert op.kernel_family.data == "conv2d"
    assert op.provenance.source.data == "agent"


def test_request_exo_kernel_verify_ok() -> None:
    """RequestExoKernelOp verifies with positive search budget."""
    op = RequestExoKernelOp.build(
        properties={
            "region_ref": SymbolRefAttr("r0"),
            "search_budget": _i64(5),
        }
    )
    op.verify()  # should not raise


def test_request_exo_kernel_verify_zero_budget_fails() -> None:
    """RequestExoKernelOp rejects zero search budget."""
    op = RequestExoKernelOp.build(
        properties={
            "region_ref": SymbolRefAttr("r0"),
            "search_budget": _i64(0),
        }
    )
    with pytest.raises(VerifyException, match="positive"):
        op.verify()


def test_request_exo_kernel_verify_negative_budget_fails() -> None:
    """RequestExoKernelOp rejects negative search budget."""
    op = RequestExoKernelOp.build(
        properties={
            "region_ref": SymbolRefAttr("r0"),
            "search_budget": _i64(-1),
        }
    )
    with pytest.raises(VerifyException, match="positive"):
        op.verify()


def test_request_exo_kernel_name() -> None:
    assert RequestExoKernelOp.name == "recipe.request_exo_kernel"


# -- SelectExoScheduleLibOp ---------------------------------------------------


def test_select_exo_schedule_lib_build_minimal() -> None:
    """SelectExoScheduleLibOp can be built with required properties only."""
    op = SelectExoScheduleLibOp.build(
        properties={
            "region_ref": SymbolRefAttr("r0"),
            "lib_name": StringAttr("x86_avx512"),
        }
    )
    assert op.lib_name.data == "x86_avx512"
    assert op.version is None


def test_select_exo_schedule_lib_build_with_version() -> None:
    """SelectExoScheduleLibOp can include a version string."""
    op = SelectExoScheduleLibOp.build(
        properties={
            "region_ref": SymbolRefAttr("r0"),
            "lib_name": StringAttr("neon_v8"),
            "version": StringAttr("2.0"),
        }
    )
    assert op.version.data == "2.0"


def test_select_exo_schedule_lib_name() -> None:
    assert SelectExoScheduleLibOp.name == "recipe.select_exo_schedule_lib"
