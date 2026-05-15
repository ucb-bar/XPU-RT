"""Tests for ``compgen.ir.recipe.execute.RecipeExecutor``.

Locks in the dispatch added so the autonomous compile loop and the MCP
``apply_recipe`` path no longer count MLIR-shaped lowered transform
scripts as ``transforms_failed``.
"""

from __future__ import annotations

from compgen.ir.recipe.execute import RecipeExecutor, _looks_like_mlir_transform
from compgen.ir.recipe.lower import LoweringOutput
from xdsl.dialects.builtin import ModuleOp


def test_looks_like_mlir_transform_recognises_lowered_recipe_text() -> None:
    # Real recipe-lowering output (sliding-tile recipe).
    sample = (
        "// Tile r_4 with sizes [64, 64, 32]\ntransform.structured.tile_using_forall %r_4\n  tile_sizes [64, 64, 32]\n"
    )
    assert _looks_like_mlir_transform(sample)


def test_looks_like_mlir_transform_passes_python_through() -> None:
    py = "from xdsl.pattern_rewriter import RewritePattern\nclass Tile(RewritePattern):\n    pass\n"
    assert not _looks_like_mlir_transform(py)


def test_apply_transforms_routes_mlir_text_to_mutator_accounting() -> None:
    """All-MLIR script set must report ``failed=0`` and ``applied==N``,
    not failed=N with a SyntaxError per script. Pre-fix every recipe op
    counted as a failed transform on real models (TinyLlama: 1231/1231).
    """
    executor = RecipeExecutor()
    lowered = LoweringOutput(
        transform_scripts=[
            "// tile region r_0\ntransform.structured.tile_using_forall %r_0\n",
            "// fuse r_0 r_1\ntransform.structured.fuse %r_0\n",
            'transform.structured.match ops{["linalg.matmul"]} in %m\n',
        ]
    )
    result = executor.execute(ModuleOp([]), lowered)
    assert result.transforms_failed == 0
    assert result.transforms_applied == 3


def test_apply_transforms_still_reports_real_python_syntax_errors() -> None:
    """Non-MLIR scripts that fail Python syntax must still surface as
    ``transforms_failed`` — we narrowed the dispatch, not the diagnostic."""
    executor = RecipeExecutor()
    lowered = LoweringOutput(transform_scripts=["this is not python and not transform-dialect either ::: !"])
    result = executor.execute(ModuleOp([]), lowered)
    assert result.transforms_failed == 1
    assert result.transforms_applied == 0
