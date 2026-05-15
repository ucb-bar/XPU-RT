"""Tests for the walk-and-annotate path of 11 upgraded stub passes (stub=False).

Each test builds an empty ModuleOp, runs the pass, and asserts the
count attribute lands on the module with value 0 (no matches on empty
input). Running successfully on a non-matching module exercises the
full walk+annotate path without requiring a synthesized input IR.
"""

from __future__ import annotations

import pytest
from xpu_rt.ir.payload.passes.stubs import (
    FoldTransposesIntoDots,
    FuseDequantMatmul,
    FuseSoftmaxToTriton,
    LowerConvToImg2Col,
    LowerQuantizedConv,
    LowerQuantizedMatmul,
    MatchLibraryCall,
    PlanReduction,
    PropagateTransposes,
    RaiseSpecialOps,
    SetNumericsPolicy,
)
from xdsl.dialects.builtin import ModuleOp

_PASS_COUNT_ATTR: list[tuple[type, str]] = [
    (LowerQuantizedMatmul, "xpu_rt.lower_quantized_matmul.count"),
    (LowerQuantizedConv, "xpu_rt.lower_quantized_conv.count"),
    (PropagateTransposes, "xpu_rt.propagate_transposes.count"),
    (LowerConvToImg2Col, "xpu_rt.lower_conv_to_img2col.count"),
    (RaiseSpecialOps, "xpu_rt.raise_special_ops.count"),
    (MatchLibraryCall, "xpu_rt.match_library_call.count"),
    (SetNumericsPolicy, "xpu_rt.set_numerics_policy.count"),
    (FoldTransposesIntoDots, "xpu_rt.fold_transposes_into_dots.count"),
    (PlanReduction, "xpu_rt.plan_reduction.count"),
    (FuseSoftmaxToTriton, "xpu_rt.fuse_softmax_to_triton.count"),
    (FuseDequantMatmul, "xpu_rt.fuse_dequant_matmul.count"),
]


@pytest.mark.parametrize("cls,count_attr", _PASS_COUNT_ATTR, ids=[c.name for c, _ in _PASS_COUNT_ATTR])
def test_pass_not_stub(cls, count_attr) -> None:
    p = cls()
    assert p.stub is False


@pytest.mark.parametrize("cls,count_attr", _PASS_COUNT_ATTR, ids=[c.name for c, _ in _PASS_COUNT_ATTR])
def test_pass_runs_on_empty_module(cls, count_attr) -> None:
    mod = ModuleOp([])
    p = cls()
    result = p.run(mod)
    assert result is mod
    # Every pass tags the module with its count attribute.
    assert count_attr in mod.attributes
    val = mod.attributes[count_attr]
    # IntegerAttr stores value.data
    assert int(val.value.data) == 0


@pytest.mark.parametrize("cls,_count_attr", _PASS_COUNT_ATTR, ids=[c.name for c, _ in _PASS_COUNT_ATTR])
def test_pass_registers_non_stub_tool(cls, _count_attr) -> None:
    import xpu_rt.ir.payload.passes  # noqa: F401  auto-registration
    from xpu_rt.llm import get_registry

    r = get_registry()
    tool = r.lookup_tool(cls.name, phase=cls.phase)
    assert tool is not None
    assert tool.is_stub is False


def test_match_library_call_accepts_family_list() -> None:
    """Regression: the tool should accept a plain list of family names."""
    mod = ModuleOp([])
    p = MatchLibraryCall()
    # Should not raise; no matmul/conv present so count stays 0.
    p.run(mod, target_capabilities=["gemm", "gemm_int8", "conv2d_nhwc"])
    assert int(mod.attributes["xpu_rt.match_library_call.count"].value.data) == 0


def test_lower_quantized_matmul_skip_policy() -> None:
    mod = ModuleOp([])
    LowerQuantizedMatmul().run(mod, policy="skip")
    # policy=skip is valid; still 0 matches on empty module.
    assert int(mod.attributes["xpu_rt.lower_quantized_matmul.count"].value.data) == 0
