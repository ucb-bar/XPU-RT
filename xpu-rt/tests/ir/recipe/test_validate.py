"""Tests for Recipe IR validation."""

from __future__ import annotations

import pytest

from compgen.ir.recipe.compat import recipe_list_to_module
from compgen.ir.recipe.ops import AssignDevice, MatchRegion, SetTileParams
from compgen.ir.recipe.validate import validate_recipe, validate_recipe_module


def test_valid_recipe_passes() -> None:
    """A well-formed recipe should pass validation."""
    ops = [
        MatchRegion(region_id="matmul_0", op_filter="linalg.matmul"),
        SetTileParams(region_id="matmul_0", tile_sizes=(128, 128, 32)),
        AssignDevice(region_id="matmul_0", device_index=0, reason="GPU"),
    ]
    module = recipe_list_to_module(ops)
    result = validate_recipe_module(module)
    assert result.valid
    assert result.errors == []


def test_negative_tile_sizes_fail() -> None:
    """Tile sizes must be positive."""
    # Build a module with a negative tile size -- xDSL verify_() on TileOp
    # should catch it and produce a validation error.
    ops = [
        MatchRegion(region_id="r0"),
        SetTileParams(region_id="r0", tile_sizes=(-1, 64)),
    ]
    module = recipe_list_to_module(ops)
    result = validate_recipe_module(module)
    assert not result.valid
    assert len(result.errors) >= 1
    # The error should mention the negative tile size
    messages = " ".join(e.message for e in result.errors)
    assert "tile" in messages.lower() or "positive" in messages.lower() or "Tile" in messages


def test_invalid_device_index_fails() -> None:
    """Device indices must reference valid devices in the profile.

    When PlaceOnDeviceOp references a region symbol that doesn't exist
    in the module, validation should catch the unresolved symbol reference.
    """
    # Create only a PlaceOnDeviceOp (via AssignDevice) with no matching
    # RecipeRegionOp -- the symbol reference will be unresolved.
    ops = [
        AssignDevice(region_id="nonexistent_region", device_index=99, reason="bad"),
    ]
    module = recipe_list_to_module(ops)
    result = validate_recipe_module(module)
    assert not result.valid
    assert len(result.errors) >= 1
    messages = " ".join(e.message for e in result.errors)
    assert "unresolved" in messages.lower() or "symbol" in messages.lower()
