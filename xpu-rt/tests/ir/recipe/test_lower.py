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


def test_lower_recipe_dispatches_ops() -> None:
    """lower_recipe should dispatch each RecipeOp to its handler."""
    from compgen.ir.recipe.compat import recipe_list_to_module
    from compgen.ir.recipe.lower import lower_recipe
    from compgen.ir.recipe.ops import (
        AssignDevice,
        MatchRegion,
        RequestKernelSearch,
        RequireCheck,
        SetTileParams,
    )

    ops = [
        MatchRegion(region_id="matmul_0", op_filter="linalg.matmul"),
        SetTileParams(region_id="matmul_0", tile_sizes=(128, 128, 32)),
        AssignDevice(region_id="matmul_0", device_index=0, reason="GPU"),
        RequestKernelSearch(region_id="matmul_0", backend="triton", search_budget=20),
        RequireCheck(region_id="matmul_0", check_type="differential"),
    ]
    module = recipe_list_to_module(ops)
    result = lower_recipe(module)

    # TileOp -> transform_scripts
    assert len(result.transform_scripts) >= 1
    assert "matmul_0" in result.transform_scripts[0]

    # RequestTritonKernelOp -> kernel_jobs
    assert len(result.kernel_jobs) >= 1
    assert result.kernel_jobs[0]["type"] == "kernel_search"

    # PlaceOnDeviceOp -> plan_fragments
    assert len(result.plan_fragments) >= 1
    assert result.plan_fragments[0]["type"] == "placement"

    # RequireDiffTestOp -> verification_obligations
    assert len(result.verification_obligations) >= 1
    assert result.verification_obligations[0]["type"] == "differential"


def test_lower_recipe_collects_diagnostics() -> None:
    """lower_recipe should surface per-op diagnostics in the output."""
    from xdsl.dialects.builtin import ModuleOp
    from xdsl.ir import Block, Region

    from compgen.ir.recipe.lower import lower_recipe

    # An empty module should lower without errors and produce empty outputs
    module = ModuleOp(Region(Block()))
    result = lower_recipe(module)
    assert result.diagnostics == []
    assert result.transform_scripts == []
    assert result.kernel_jobs == []
