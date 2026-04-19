"""Basic sanity tests for the compgen.schemas package."""

from __future__ import annotations

import pytest

from compgen.schemas import available_schemas, load_schema, schema_path


def test_available_schemas_lists_all_five():
    got = available_schemas()
    assert set(got) == {
        "kernel_contract",
        "recipe_ir",
        "execution_plan",
        "target_resource",
        "region_analysis",
    }


@pytest.mark.parametrize("name", [
    "kernel_contract",
    "recipe_ir",
    "execution_plan",
    "target_resource",
    "region_analysis",
])
def test_load_schema_returns_dict(name: str):
    doc = load_schema(name)
    assert isinstance(doc, dict)
    assert doc


def test_schema_path_points_at_real_file():
    p = schema_path("recipe_ir")
    assert p.exists()
    assert p.suffix == ".yaml"


def test_unknown_schema_raises():
    with pytest.raises(KeyError):
        schema_path("nonexistent")


def test_unknown_version_raises():
    with pytest.raises(ValueError):
        available_schemas(version="v99")
