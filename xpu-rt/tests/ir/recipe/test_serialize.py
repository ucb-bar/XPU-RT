"""Tests for recipe IR serialization."""

from __future__ import annotations

import pytest
from compgen.ir.recipe.serialize import recipe_to_yaml, yaml_to_recipe


def test_recipe_to_yaml_exists() -> None:
    assert callable(recipe_to_yaml)


def test_yaml_to_recipe_exists() -> None:
    assert callable(yaml_to_recipe)


@pytest.mark.skip(reason="scaffold only -- implementation pending")
def test_recipe_round_trip() -> None:
    """recipe_to_yaml -> yaml_to_recipe should be lossless."""


@pytest.mark.skip(reason="scaffold only -- implementation pending")
def test_recipe_to_yaml_deterministic() -> None:
    """Serialization should produce identical YAML for identical inputs."""
