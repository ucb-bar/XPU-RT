"""Tests for recipe IR serialization."""

from __future__ import annotations

from compgen.ir.recipe.serialize import recipe_to_yaml, yaml_to_recipe


def test_recipe_to_yaml_exists() -> None:
    assert callable(recipe_to_yaml)


def test_yaml_to_recipe_exists() -> None:
    assert callable(yaml_to_recipe)


def test_recipe_round_trip() -> None:
    """recipe_to_yaml -> yaml_to_recipe should be lossless."""
    from compgen.ir.recipe.ops import (
        AssignDevice,
        MatchRegion,
    )

    # Use ops whose fields are all plain scalars (no tuples) so that
    # yaml.dump + yaml.safe_load round-trips cleanly.
    ops = [
        MatchRegion(region_id="matmul_0", op_filter="linalg.matmul"),
        AssignDevice(region_id="matmul_0", device_index=0, reason="GPU"),
    ]
    yaml_text = recipe_to_yaml(ops)
    recovered = yaml_to_recipe(yaml_text)

    # recovered is a list of dicts; check structural fidelity
    assert isinstance(recovered, list)
    assert len(recovered) == len(ops)
    # Each entry should have a _type key
    assert recovered[0]["_type"] == "MatchRegion"
    assert recovered[0]["region_id"] == "matmul_0"
    assert recovered[0]["op_filter"] == "linalg.matmul"
    assert recovered[1]["_type"] == "AssignDevice"
    assert recovered[1]["device_index"] == 0
    assert recovered[1]["reason"] == "GPU"


def test_recipe_to_yaml_deterministic() -> None:
    """Serialization should produce identical YAML for identical inputs."""
    from compgen.ir.recipe.ops import (
        AssignDevice,
        MatchRegion,
        SetTileParams,
    )

    ops = [
        MatchRegion(region_id="r0", op_filter="linalg.matmul"),
        SetTileParams(region_id="r0", tile_sizes=(64, 64)),
        AssignDevice(region_id="r0", device_index=0, reason="GPU"),
    ]
    yaml_1 = recipe_to_yaml(ops)
    yaml_2 = recipe_to_yaml(ops)
    assert yaml_1 == yaml_2
