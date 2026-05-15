"""Tests for promotion/cache.py -- recipe cache management."""

from __future__ import annotations

from pathlib import Path

from compgen.promotion.cache import RecipeCache
from compgen.promotion.promote import RecipeKey
from compgen.runtime.bundle import Bundle


def test_recipe_cache_construction(tmp_path: Path) -> None:
    cache = RecipeCache(library_path=tmp_path / "recipes")
    assert cache.library_path == tmp_path / "recipes"


def test_cache_put_and_get(tmp_path: Path) -> None:
    cache = RecipeCache(library_path=tmp_path / "lib")
    key = RecipeKey("tgt", "mdl", "obj", 1)
    bundle = Bundle(
        target_profile="cuda-a100",
        model_hash="abc123",
        objective="latency",
        artifacts={"payload": "payload.mlir"},
    )

    cache.put(key, bundle)
    result = cache.get(key)

    assert result is not None
    assert result.target_profile == "cuda-a100"
    assert result.model_hash == "abc123"


def test_cache_get_missing(tmp_path: Path) -> None:
    cache = RecipeCache(library_path=tmp_path / "lib")
    key = RecipeKey("x", "y", "z", 1)
    assert cache.get(key) is None


def test_cache_invalidate(tmp_path: Path) -> None:
    cache = RecipeCache(library_path=tmp_path / "lib")
    key = RecipeKey("tgt", "mdl", "obj", 1)
    bundle = Bundle(target_profile="test")

    cache.put(key, bundle)
    assert cache.get(key) is not None

    assert cache.invalidate(key, reason="outdated")
    assert cache.get(key) is None


def test_cache_invalidate_missing(tmp_path: Path) -> None:
    cache = RecipeCache(library_path=tmp_path / "lib")
    key = RecipeKey("x", "y", "z", 1)
    assert not cache.invalidate(key)


def test_cache_list_recipes(tmp_path: Path) -> None:
    cache = RecipeCache(library_path=tmp_path / "lib")

    for i in range(3):
        key = RecipeKey("tgt", "mdl", "obj", i + 1)
        cache.put(key, Bundle(target_profile=f"t{i}"))

    recipes = cache.list_recipes()
    assert len(recipes) == 3


def test_cache_list_filter_by_target(tmp_path: Path) -> None:
    cache = RecipeCache(library_path=tmp_path / "lib")
    cache.put(RecipeKey("aaa", "m1", "o1", 1), Bundle())
    cache.put(RecipeKey("bbb", "m2", "o2", 1), Bundle())

    recipes = cache.list_recipes(target_hash="aaa")
    assert len(recipes) == 1
    assert recipes[0].target_hash == "aaa"
