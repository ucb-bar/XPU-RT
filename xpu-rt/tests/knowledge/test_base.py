"""Tests for knowledge base core types and queries."""

from __future__ import annotations

from compgen.knowledge.base import (
    Confidence,
    KnowledgeBase,
    TilingGuidance,
    TransformRecipe,
    TransformStep,
)


def _make_kb() -> KnowledgeBase:
    """Create a populated KB for testing."""
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


def test_kb_has_matmul_wisdom() -> None:
    kb = _make_kb()
    w = kb.query_op("matmul")
    assert w is not None
    assert w.op_family == "matmul"
    assert len(w.tiling_guidance) >= 2  # at least GPU + CPU


def test_kb_has_conv2d_wisdom() -> None:
    kb = _make_kb()
    w = kb.query_op("conv2d")
    assert w is not None


def test_kb_has_attention_wisdom() -> None:
    kb = _make_kb()
    w = kb.query_op("attention")
    assert w is not None


def test_kb_query_gpu_patterns() -> None:
    kb = _make_kb()
    patterns = kb.query_target("gpu")
    assert len(patterns) >= 4  # memory, parallelism, data_movement, instruction


def test_kb_query_cpu_patterns() -> None:
    kb = _make_kb()
    patterns = kb.query_target("cpu")
    assert len(patterns) >= 3


def test_kb_query_recipes_by_op() -> None:
    kb = _make_kb()
    recipes = kb.query_recipes(op_family="matmul")
    assert len(recipes) >= 2  # GPU + CPU at minimum


def test_kb_query_recipes_by_target() -> None:
    kb = _make_kb()
    recipes = kb.query_recipes(target_class="gpu")
    assert len(recipes) >= 3


def test_kb_query_kernel_wisdom_cutlass() -> None:
    kb = _make_kb()
    wisdom = kb.query_kernel_wisdom(library="cutlass")
    assert len(wisdom) >= 5


def test_kb_query_kernel_wisdom_triton() -> None:
    kb = _make_kb()
    wisdom = kb.query_kernel_wisdom(library="triton")
    assert len(wisdom) >= 4


def test_kb_query_compiler_heuristics() -> None:
    kb = _make_kb()
    heuristics = kb.query_heuristics()
    assert len(heuristics) >= 15


def test_kb_query_heuristics_by_compiler() -> None:
    kb = _make_kb()
    tvm = kb.query_heuristics(compiler="tvm")
    assert len(tvm) >= 4


def test_kb_has_anti_patterns() -> None:
    kb = _make_kb()
    anti = kb.query_anti_patterns()
    assert len(anti) >= 10


def test_kb_summary() -> None:
    kb = _make_kb()
    s = kb.summary()
    assert s["op_wisdom"] >= 6
    assert s["target_patterns"] >= 3
    assert s["transform_recipes"] >= 5
    assert s["kernel_wisdom"] >= 20
    assert s["anti_patterns"] >= 10


def test_tiling_guidance_fields() -> None:
    tg = TilingGuidance(
        target_class="gpu",
        tile_sizes=[128, 128, 32],
        rationale="tensor core",
        source="CUTLASS",
        confidence=Confidence.HIGH,
    )
    assert tg.tile_sizes == [128, 128, 32]
    assert tg.confidence == Confidence.HIGH


def test_transform_recipe_fields() -> None:
    recipe = TransformRecipe(
        name="test",
        op_family="matmul",
        target_class="gpu",
        steps=[TransformStep(action="tile", parameters={"sizes": [64, 64]}, rationale="cache")],
        expected_speedup="2x",
        source="test",
        confidence=Confidence.MEDIUM,
    )
    assert len(recipe.steps) == 1
    assert recipe.steps[0].action == "tile"
