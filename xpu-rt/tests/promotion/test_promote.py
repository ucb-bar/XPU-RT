"""Tests for recipe promotion."""

from __future__ import annotations

from pathlib import Path

from compgen.promotion.promote import RecipeKey, RecipePromoter, promote_recipe
from compgen.runtime.bundle import Bundle


def test_recipe_key_construction() -> None:
    key = RecipeKey(target_hash="abc", model_hash="def", objective_hash="ghi", version=1)
    assert key.key == "abc_def_ghi_v1"


def test_recipe_key_version_increment() -> None:
    key1 = RecipeKey(target_hash="a", model_hash="b", objective_hash="c", version=1)
    key2 = RecipeKey(target_hash="a", model_hash="b", objective_hash="c", version=2)
    assert key1.key != key2.key


def test_promote_success(tmp_path: Path) -> None:
    """A bundle should promote successfully."""
    bundle = Bundle(
        target_profile="cuda-a100",
        model_hash="abc123",
        objective="latency",
        artifacts={"payload": "payload.mlir"},
        creation_timestamp="2025-01-15T12:00:00Z",
    )
    promoter = RecipePromoter(library_path=tmp_path / "library")
    result = promoter.promote(bundle)

    assert result.promoted
    assert result.key is not None
    assert result.recipe_path is not None
    assert result.recipe_path.exists()
    assert (result.recipe_path / "manifest.json").exists()


def test_promote_versioning(tmp_path: Path) -> None:
    """Promoting the same bundle twice should create v1 and v2."""
    bundle = Bundle(
        target_profile="cuda-a100",
        model_hash="abc123",
        objective="latency",
    )
    promoter = RecipePromoter(library_path=tmp_path / "library")

    r1 = promoter.promote(bundle)
    r2 = promoter.promote(bundle)

    assert r1.key.version == 1
    assert r2.key.version == 2
    assert r1.recipe_path != r2.recipe_path


def test_promote_recipe_convenience(tmp_path: Path) -> None:
    """promote_recipe convenience function should work."""
    bundle = Bundle(target_profile="test", model_hash="x")
    result = promote_recipe(bundle, tmp_path / "lib")
    assert result.promoted
