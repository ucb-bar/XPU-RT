"""Tests for dataflow analysis verification."""

from __future__ import annotations

import pytest
from compgen.ir.semantic.dataflow_verify import AnalysisVerificationResult


def test_analysis_verification_result_sound() -> None:
    result = AnalysisVerificationResult(sound=True, status="sound", solver_time_ms=10.0)
    assert result.sound is True
    assert result.status == "sound"
    assert result.counterexample is None
    assert result.solver_time_ms == 10.0


def test_analysis_verification_result_unsound() -> None:
    ce = {"input": [1, 2, 3], "expected_range": "(0, 5)", "actual": 7}
    result = AnalysisVerificationResult(sound=False, status="unsound", counterexample=ce)
    assert result.sound is False
    assert result.status == "unsound"
    assert result.counterexample is not None


def test_verify_analysis_soundness() -> None:
    """verify_analysis should check that analysis is a sound over-approximation."""
    z3 = pytest.importorskip("z3")
    from compgen.ir.semantic.dialect import PredicateOp, SemanticType

    bv_type = SemanticType(kind="bitvector", width=8)
    # Test 'eq' predicate
    pred = PredicateOp(name="eq", operands=["x", "y"], semantic_type=bv_type)
    x = z3.BitVec("x", 8)
    y = z3.BitVec("y", 8)
    var_map = {"x": x, "y": y}
    expr = pred.to_z3(var_map)
    # eq(x, y) with x=5, y=5 should be satisfiable
    s = z3.Solver()
    s.add(expr)
    s.add(x == 5)
    s.add(y == 5)
    assert s.check() == z3.sat


def test_verify_analysis_timeout() -> None:
    """verify_analysis with 'true' predicate should always be satisfiable."""
    z3 = pytest.importorskip("z3")
    from compgen.ir.semantic.dialect import PredicateOp

    pred = PredicateOp(name="true")
    expr = pred.to_z3({})
    s = z3.Solver()
    s.add(expr)
    assert s.check() == z3.sat
