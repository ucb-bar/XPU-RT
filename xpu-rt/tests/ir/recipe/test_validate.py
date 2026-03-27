"""Tests for Recipe IR validation."""

from __future__ import annotations

import pytest


@pytest.mark.skip(reason="scaffold only -- implementation pending")
def test_valid_recipe_passes() -> None:
    """A well-formed recipe should pass validation."""


@pytest.mark.skip(reason="scaffold only -- implementation pending")
def test_negative_tile_sizes_fail() -> None:
    """Tile sizes must be positive."""


@pytest.mark.skip(reason="scaffold only -- implementation pending")
def test_invalid_device_index_fails() -> None:
    """Device indices must reference valid devices in the profile."""
