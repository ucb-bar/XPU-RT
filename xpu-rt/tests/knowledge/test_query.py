"""Tests for knowledge query and injection."""

from __future__ import annotations

from compgen.knowledge.base import KnowledgeBase
from compgen.knowledge.inject import (
    inject_for_analysis,
    inject_for_eqsat,
    inject_for_kernel_search,
    inject_knowledge,
)
from compgen.knowledge.query import QueryContext, QueryResult, query_for_op, query_for_stage


def _make_kb() -> KnowledgeBase:
    from compgen.knowledge.anti_patterns import build_default_anti_patterns
    from compgen.knowledge.compiler_heuristics import build_default_compiler_heuristics
    from compgen.knowledge.kernel_wisdom import build_default_kernel_wisdom
    from compgen.knowledge.ops_wisdom import build_default_op_wisdom
    from compgen.knowledge.target_patterns import build_default_target_patterns
    from compgen.knowledge.transform_recipes import build_default_recipes

    return KnowledgeBase(
        op_wisdom=build_default_op_wisdom(),
        target_patterns=build_default_target_patterns(),
        transform_recipes=build_default_recipes(),
        kernel_wisdom=build_default_kernel_wisdom(),
        compiler_heuristics=build_default_compiler_heuristics(),
        anti_patterns=build_default_anti_patterns(),
    )


def test_query_matmul_gpu() -> None:
    kb = _make_kb()
    result = query_for_op(kb, "matmul", "gpu")
    assert not result.is_empty
    assert len(result.op_wisdom) == 1
    assert result.op_wisdom[0].op_family == "matmul"
    assert len(result.recipes) >= 1
    assert len(result.anti_patterns) >= 5


def test_query_for_eqsat_stage() -> None:
    kb = _make_kb()
    result = query_for_stage(kb, "eqsat", "gpu", ["matmul", "elementwise"])
    assert not result.is_empty
    assert len(result.op_wisdom) >= 1


def test_query_empty_op() -> None:
    kb = _make_kb()
    result = query_for_op(kb, "nonexistent_op", "gpu")
    # Should still have target patterns and anti-patterns
    assert len(result.target_patterns) >= 1
    assert len(result.anti_patterns) >= 1


def test_inject_for_analysis() -> None:
    kb = _make_kb()
    text = inject_for_analysis(kb, ["matmul", "relu"], "gpu")
    assert len(text) > 100
    assert "matmul" in text.lower()
    assert "Tiling" in text or "tiling" in text


def test_inject_for_eqsat() -> None:
    kb = _make_kb()
    text = inject_for_eqsat(kb, ["matmul"], "gpu")
    assert len(text) > 50


def test_inject_for_kernel_search() -> None:
    kb = _make_kb()
    text = inject_for_kernel_search(kb, "matmul", "gpu")
    assert len(text) > 50
    assert "CUTLASS" in text or "cutlass" in text or "Triton" in text or "triton" in text


def test_inject_empty_kb() -> None:
    kb = KnowledgeBase()
    text = inject_knowledge(kb, QueryContext(op_families=["matmul"]))
    assert text == ""


def test_query_result_is_empty() -> None:
    r = QueryResult()
    assert r.is_empty


def test_query_cpu_target() -> None:
    kb = _make_kb()
    result = query_for_op(kb, "matmul", "cpu")
    assert not result.is_empty
    # Should have CPU-specific kernel wisdom
    has_cpu = any("onednn" in w.library or "exo" in w.library for w in result.kernel_wisdom)
    assert has_cpu


def test_inject_includes_recipes() -> None:
    kb = _make_kb()
    text = inject_for_analysis(kb, ["matmul"], "gpu")
    assert "Recipe" in text or "recipe" in text or "steps" in text.lower()
