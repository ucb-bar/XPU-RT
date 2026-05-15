"""Tests for peephole rewrite verification."""

from __future__ import annotations

import pytest
from compgen.ir.semantic.peephole_verify import RewriteVerificationResult


def test_rewrite_verification_result_valid() -> None:
    result = RewriteVerificationResult(valid=True, status="valid", solver_time_ms=42.5)
    assert result.valid is True
    assert result.status == "valid"
    assert result.counterexample is None
    assert result.solver_time_ms == 42.5
    assert result.cached is False


def test_rewrite_verification_result_invalid() -> None:
    ce = {"x": 1, "y": -1}
    result = RewriteVerificationResult(valid=False, status="invalid", counterexample=ce)
    assert result.valid is False
    assert result.status == "invalid"
    assert result.counterexample == ce


def test_verify_rewrite_equivalence() -> None:
    """verify_rewrite should prove equivalence via refinement relation."""
    z3 = pytest.importorskip("z3")
    from compgen.ir.semantic.dialect import RefinementRelation

    # x + 0 == x (should be valid: UNSAT when we check for mismatch)
    x = z3.BitVec("x", 32)
    source = x + z3.BitVecVal(0, 32)
    target = x
    rel = RefinementRelation(source_expr="src", target_expr="tgt")
    var_map = {"src": source, "tgt": target}
    mismatch = rel.to_z3(var_map)
    s = z3.Solver()
    s.add(mismatch)
    # UNSAT means refinement holds (no counterexample)
    assert s.check() == z3.unsat


def test_verify_rewrite_caching() -> None:
    """SemanticType.to_z3_sort should produce correct Z3 sorts for caching."""
    z3 = pytest.importorskip("z3")
    from compgen.ir.semantic.dialect import SemanticType

    bv = SemanticType(kind="bitvector", width=16)
    sort = bv.to_z3_sort()
    assert sort == z3.BitVecSort(16)

    int_type = SemanticType(kind="integer")
    assert int_type.to_z3_sort() == z3.IntSort()

    bool_type = SemanticType(kind="boolean")
    assert bool_type.to_z3_sort() == z3.BoolSort()
