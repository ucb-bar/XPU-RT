"""Tests for semantic dialect types."""

from __future__ import annotations

from compgen.ir.semantic.dialect import PredicateOp, RefinementRelation, SemanticType


def test_semantic_type() -> None:
    t = SemanticType(kind="bitvector", width=32)
    assert t.kind == "bitvector"
    assert t.width == 32


def test_predicate_op() -> None:
    p = PredicateOp(name="eq", operands=["a", "b"])
    assert p.name == "eq"
    assert len(p.operands) == 2


def test_refinement_relation() -> None:
    r = RefinementRelation(source_expr="x + 0", target_expr="x")
    assert r.source_expr == "x + 0"
