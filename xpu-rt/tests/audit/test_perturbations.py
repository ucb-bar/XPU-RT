"""Tests for compgen.audit.perturbations."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from compgen.audit.perturbations import (
    change_output_dir,
    corrupt_promotion_library,
    empty_promotion_library,
    rename_regions,
    vary_tile_divisibility,
)


def test_rename_regions_rewrites_jsons(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    f = run_dir / "candidate_actions.json"
    f.write_text(json.dumps({"region_id": "matmul_0",
                             "candidates": [{"region_id": "matmul_0"}]}))
    g = run_dir / "graph.json"
    g.write_text(json.dumps({"regions": ["matmul_0", "matmul_1"]}))

    result = rename_regions(run_dir, {"matmul_0": "renamed_zero"})
    assert result.name == "rename_regions"
    new_f = json.loads(f.read_text())
    assert new_f["region_id"] == "renamed_zero"
    new_g = json.loads(g.read_text())
    assert "renamed_zero" in new_g["regions"]


def test_change_output_dir_moves(tmp_path: Path) -> None:
    src = tmp_path / "old"
    src.mkdir()
    (src / "a.txt").write_text("hello")
    dst = tmp_path / "new"
    result = change_output_dir(src, dst)
    assert result.target == dst
    assert (dst / "a.txt").read_text() == "hello"
    assert not src.exists()


def test_vary_tile_divisibility_annotates_yaml(tmp_path: Path) -> None:
    yaml_path = tmp_path / "model.yaml"
    yaml_path.write_text(
        "schema_version: graphcomp_model_config_v1\nmodel_id: x\n"
    )
    result = vary_tile_divisibility(yaml_path)
    text = yaml_path.read_text()
    assert "perturbation:" in text
    assert "vary_tile_divisibility" in text


def test_vary_tile_divisibility_idempotent(tmp_path: Path) -> None:
    yaml_path = tmp_path / "model.yaml"
    yaml_path.write_text(
        "schema_version: graphcomp_model_config_v1\nperturbation: vary_tile_divisibility\n"
    )
    vary_tile_divisibility(yaml_path)
    text = yaml_path.read_text()
    # Should not duplicate
    assert text.count("perturbation:") == 1


def test_corrupt_promotion_library_zeroes_contract_hash(tmp_path: Path) -> None:
    library = tmp_path / "library"
    sidecar = library / "key123" / "promoted_recipe.json"
    sidecar.parent.mkdir(parents=True)
    sidecar.write_text(json.dumps({
        "key": {"contract_hash": "abcdef0123456789",
                "region_signature": "fedcba9876543210"},
        "recipe_signature": "...",
    }))
    result = corrupt_promotion_library(library)
    assert result.name == "corrupt_promotion_library"
    new_data = json.loads(sidecar.read_text())
    assert new_data["key"]["contract_hash"] == "0" * 16
    # Region signature is untouched
    assert new_data["key"]["region_signature"] == "fedcba9876543210"


def test_corrupt_promotion_library_handles_missing(tmp_path: Path) -> None:
    library = tmp_path / "absent"
    result = corrupt_promotion_library(library)
    assert "missing" in result.before


def test_empty_promotion_library_wipes(tmp_path: Path) -> None:
    library = tmp_path / "library"
    sidecar = library / "key1" / "promoted_recipe.json"
    sidecar.parent.mkdir(parents=True)
    sidecar.write_text("{}")
    result = empty_promotion_library(library)
    assert result.name == "empty_promotion_library"
    assert not library.exists()


def test_empty_promotion_library_idempotent(tmp_path: Path) -> None:
    library = tmp_path / "absent"
    # Should not raise even when library doesn't exist
    result = empty_promotion_library(library)
    assert not library.exists()
