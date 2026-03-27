"""Tests for recipe IR lowering."""

from __future__ import annotations

import pytest
from compgen.ir.recipe.lower import LoweringOutput


def test_lowering_output_defaults() -> None:
    output = LoweringOutput()
    assert output.transform_scripts == []
    assert output.kernel_jobs == []
    assert output.plan_fragments == []
    assert output.verification_obligations == []
    assert output.diagnostics == []


def test_lowering_output_with_data() -> None:
    output = LoweringOutput(
        transform_scripts=["script_1"],
        kernel_jobs=[{"backend": "triton"}],
        diagnostics=["warning: fallback used"],
    )
    assert len(output.transform_scripts) == 1
    assert output.kernel_jobs[0]["backend"] == "triton"
    assert len(output.diagnostics) == 1


@pytest.mark.skip(reason="scaffold only -- implementation pending")
def test_lower_recipe_dispatches_ops() -> None:
    """lower_recipe should dispatch each RecipeOp to its handler."""


@pytest.mark.skip(reason="scaffold only -- implementation pending")
def test_lower_recipe_collects_diagnostics() -> None:
    """lower_recipe should surface per-op diagnostics in the output."""
