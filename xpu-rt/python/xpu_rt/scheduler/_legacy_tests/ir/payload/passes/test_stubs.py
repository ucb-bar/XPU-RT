"""Tests for the Wave-2/3 ported passes (upgraded to MVP real in ).

These passes historically shipped as scaffolded stubs; the fifth wave
turned them into real MVP annotation passes. This file is kept (rather
than renamed) to preserve git blame context and because the stubs.py
module itself kept its name. The tests here now assert the real
(stub=False) behaviour; the fuller per-pass invariants live in
``test_annotator_pass_coverage.py``.
"""

from __future__ import annotations

import pytest
from compgen.ir.payload.passes.stubs import (
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

_ALL_PASSES = [
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
]


@pytest.mark.parametrize("cls", _ALL_PASSES, ids=[c.name for c in _ALL_PASSES])
def test_runs_on_empty_module(cls) -> None:
    mod = ModuleOp([])
    result = cls().run(mod)
    # MVP passes mutate module attributes (count tags); they return the same
    # ModuleOp instance.
    assert result is mod


@pytest.mark.parametrize("cls", _ALL_PASSES, ids=[c.name for c in _ALL_PASSES])
def test_metadata_populated(cls) -> None:
    p = cls()
    assert p.name
    assert p.wraps_pass
    # Upgrade in wave 5: these are no longer stubs.
    assert p.stub is False


def test_all_passes_register_non_stub() -> None:
    import compgen.ir.payload.passes  # noqa: F401
    from compgen.llm import get_registry

    r = get_registry()
    for cls in _ALL_PASSES:
        tool = r.lookup_tool(cls.name, phase=cls.phase)
        assert tool is not None, f"{cls.name} not registered"
        assert tool.is_stub is False, f"{cls.name} should be non-stub after wave 5"
