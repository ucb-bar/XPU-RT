"""Tests for compgen.promotion.migrate_sidecars."""

from __future__ import annotations

import json
from pathlib import Path

from compgen.promotion.migrate_sidecars import (
    MigrationReport,
    MigrationResult,
    migrate_library,
    migrate_sidecar,
)


def _write_sidecar(path: Path, body: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(body, indent=2, sort_keys=True))


def _modern_sidecar() -> dict:
    """A sidecar that already carries the cross-link fields."""
    return {
        "schema_version": "promoted_recipe_v1",
        "key": {"target_hash": "t", "model_hash": "m", "objective_hash": "o", "version": 1},
        "recipe": {
            "recipe_id": "recipe_set_tile_params_matmul_0_host_cpu_aabbccdd",
            "evidence_summary": {
                "candidate_kind": "set_tile_params",
                "selected_candidate_id": "tile_M16_N16_K16",
            },
        },
    }


def _legacy_sidecar_with_recipe_id() -> dict:
    """Pre-sidecar where evidence_summary lacks candidate_kind
    but recipe_id encodes it."""
    return {
        "schema_version": "promoted_recipe_v1",
        "key": {"target_hash": "t", "model_hash": "m", "objective_hash": "o", "version": 1},
        "recipe": {
            "recipe_id": "recipe_fuse_producer_consumer_pointwise_0_host_cpu_aabbccdd",
            "recipe_signature": "abcdef0123456789",
            "evidence_summary": {"region_id": "pointwise_0"},
        },
    }


def _legacy_sidecar_unrecoverable() -> dict:
    """Pre-sidecar with no recipe_id and no signature."""
    return {
        "schema_version": "promoted_recipe_v1",
        "key": {"target_hash": "t", "model_hash": "m", "objective_hash": "o", "version": 1},
        "recipe": {
            "evidence_summary": {},
        },
    }


# --------------------------------------------------------------------------- #
# Single-sidecar migration
# --------------------------------------------------------------------------- #


def test_modern_sidecar_already_complete(tmp_path: Path) -> None:
    p = tmp_path / "promoted_recipe.json"
    _write_sidecar(p, _modern_sidecar())
    result = migrate_sidecar(p)
    assert result.already_complete
    assert not result.migrated


def test_legacy_sidecar_inferred_from_recipe_id(tmp_path: Path) -> None:
    p = tmp_path / "promoted_recipe.json"
    _write_sidecar(p, _legacy_sidecar_with_recipe_id())
    result = migrate_sidecar(p)
    assert not result.already_complete
    assert result.migrated
    assert result.inferred_candidate_kind == "fuse_producer_consumer"
    # On-disk file now carries the inferred fields.
    raw = json.loads(p.read_text())
    ev = raw["recipe"]["evidence_summary"]
    assert ev["candidate_kind"] == "fuse_producer_consumer"
    assert ev["selected_candidate_id"] == "abcdef0123456789"


def test_legacy_sidecar_unrecoverable_skipped(tmp_path: Path) -> None:
    p = tmp_path / "promoted_recipe.json"
    _write_sidecar(p, _legacy_sidecar_unrecoverable())
    result = migrate_sidecar(p)
    assert not result.already_complete
    assert not result.migrated
    assert "no inference source" in result.skipped_reason


def test_dry_run_does_not_write(tmp_path: Path) -> None:
    p = tmp_path / "promoted_recipe.json"
    _write_sidecar(p, _legacy_sidecar_with_recipe_id())
    original = p.read_text()
    result = migrate_sidecar(p, dry_run=True)
    assert not result.migrated
    assert result.inferred_candidate_kind == "fuse_producer_consumer"
    # File contents unchanged
    assert p.read_text() == original


def test_missing_sidecar_skipped(tmp_path: Path) -> None:
    result = migrate_sidecar(tmp_path / "does_not_exist.json")
    assert not result.already_complete
    assert not result.migrated
    assert "not found" in result.skipped_reason


def test_unreadable_sidecar_skipped(tmp_path: Path) -> None:
    p = tmp_path / "promoted_recipe.json"
    p.write_text("not json")
    result = migrate_sidecar(p)
    assert not result.migrated
    assert "unreadable" in result.skipped_reason


# --------------------------------------------------------------------------- #
# Library-wide migration
# --------------------------------------------------------------------------- #


def test_migrate_library_aggregates(tmp_path: Path) -> None:
    library = tmp_path / "library"
    # Modern sidecar
    _write_sidecar(
        library / "modern_key" / "promoted_recipe.json", _modern_sidecar(),
    )
    # Legacy sidecar with recipe_id
    _write_sidecar(
        library / "legacy_key" / "promoted_recipe.json",
        _legacy_sidecar_with_recipe_id(),
    )
    # Unrecoverable sidecar
    _write_sidecar(
        library / "unrecoverable_key" / "promoted_recipe.json",
        _legacy_sidecar_unrecoverable(),
    )
    report = migrate_library(library)
    assert isinstance(report, MigrationReport)
    assert len(report.results) == 3
    assert report.already_complete_count == 1
    assert report.migrated_count == 1
    assert report.skipped_count == 1


def test_migrate_library_empty(tmp_path: Path) -> None:
    report = migrate_library(tmp_path / "empty")
    assert report.results == []
    assert report.already_complete_count == 0
    assert report.migrated_count == 0


def test_migrate_library_dry_run_aggregates_inferences(tmp_path: Path) -> None:
    library = tmp_path / "library"
    _write_sidecar(
        library / "k1" / "promoted_recipe.json",
        _legacy_sidecar_with_recipe_id(),
    )
    report = migrate_library(library, dry_run=True)
    assert len(report.results) == 1
    assert not report.results[0].migrated  # dry_run
    assert report.results[0].inferred_candidate_kind == "fuse_producer_consumer"


def test_migration_result_to_dict() -> None:
    r = MigrationResult(
        path=Path("/x/y.json"),
        already_complete=False,
        migrated=True,
        inferred_candidate_kind="set_tile_params",
        inferred_selected_candidate_id="tile_16",
    )
    raw = r.to_dict()
    assert raw["migrated"] is True
    assert raw["inferred_candidate_kind"] == "set_tile_params"
