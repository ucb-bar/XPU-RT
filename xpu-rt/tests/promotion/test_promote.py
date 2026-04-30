"""Tests for recipe promotion."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from compgen.promotion.errors import PromotionBlockedError
from compgen.promotion.promote import RecipeKey, RecipePromoter, promote_recipe
from compgen.runtime.bundle import Bundle


def _make_verified_bundle(tmp_path: Path, **overrides) -> Bundle:
    """Build a Bundle with a real verification_report.json on disk
    that passes the production gate."""
    bundle_root = tmp_path / "bundle_src"
    bundle_root.mkdir(parents=True, exist_ok=True)
    report_path = bundle_root / "verification_report.json"
    report_path.write_text(
        json.dumps(
            {
                "target": "test",
                "bundle_dir": str(bundle_root),
                "passed": True,
                "max_abs_error": 1.0e-6,
                "levels_run": ["structural", "differential"],
                "levels_passed": ["structural", "differential"],
                "details": {
                    "structural": "structural: PASS",
                    "differential": "differential: PASS (32 inputs)",
                },
            }
        )
    )
    defaults = {
        "target_profile": "cuda-a100",
        "model_hash": "abc123",
        "objective": "latency",
        "artifacts": {
            "payload": "payload.mlir",
            "verification_report": "verification_report.json",
        },
        "creation_timestamp": "2025-01-15T12:00:00Z",
        "metadata": {"bundle_root": str(bundle_root)},
    }
    defaults.update(overrides)
    return Bundle(**defaults)


def test_recipe_key_construction() -> None:
    key = RecipeKey(target_hash="abc", model_hash="def", objective_hash="ghi", version=1)
    assert key.key == "abc_def_ghi_v1"


def test_recipe_key_version_increment() -> None:
    key1 = RecipeKey(target_hash="a", model_hash="b", objective_hash="c", version=1)
    key2 = RecipeKey(target_hash="a", model_hash="b", objective_hash="c", version=2)
    assert key1.key != key2.key


def test_promote_success(tmp_path: Path) -> None:
    """A bundle with a passing verification report should promote."""
    bundle = _make_verified_bundle(tmp_path)
    promoter = RecipePromoter(library_path=tmp_path / "library")
    result = promoter.promote(bundle)

    assert result.promoted
    assert result.key is not None
    assert result.recipe_path is not None
    assert result.recipe_path.exists()
    assert (result.recipe_path / "manifest.json").exists()


def test_promote_versioning(tmp_path: Path) -> None:
    """Promoting the same bundle twice should create v1 and v2."""
    bundle = _make_verified_bundle(tmp_path)
    promoter = RecipePromoter(library_path=tmp_path / "library")

    r1 = promoter.promote(bundle)
    r2 = promoter.promote(bundle)

    assert r1.key.version == 1
    assert r2.key.version == 2
    assert r1.recipe_path != r2.recipe_path


def test_promote_recipe_convenience(tmp_path: Path) -> None:
    """promote_recipe convenience function should work."""
    bundle = _make_verified_bundle(tmp_path, target_profile="test", model_hash="x")
    result = promote_recipe(bundle, tmp_path / "lib")
    assert result.promoted


# ---------------------------------------------------------------------------
# Phase-3 verification-gate contract
# ---------------------------------------------------------------------------


def test_promote_rejects_bundle_without_verification_report(tmp_path: Path) -> None:
    """A bundle with no verification_report must be rejected by
    default — the old behavior silently promoted anything."""
    bundle = Bundle(
        target_profile="cuda-a100",
        model_hash="abc",
        objective="latency",
        artifacts={"payload": "payload.mlir"},
    )
    promoter = RecipePromoter(library_path=tmp_path / "library")
    with pytest.raises(PromotionBlockedError) as exc_info:
        promoter.promote(bundle)
    assert any(r.code == "missing_verification_report" for r in exc_info.value.reasons)


def test_promote_rejects_failed_verification(tmp_path: Path) -> None:
    """A verification_report with passed=False must block promotion."""
    bundle_root = tmp_path / "bundle_src"
    bundle_root.mkdir()
    (bundle_root / "verification_report.json").write_text(
        json.dumps(
            {
                "passed": False,
                "max_abs_error": 1.0,
                "levels_run": ["structural", "differential"],
                "levels_passed": ["structural"],
                "details": {
                    "structural": "structural: PASS",
                    "differential": "differential: FAIL — max_abs_error=1e+00",
                },
            }
        )
    )
    bundle = Bundle(
        target_profile="t",
        model_hash="m",
        artifacts={"verification_report": "verification_report.json"},
        metadata={"bundle_root": str(bundle_root)},
    )
    promoter = RecipePromoter(library_path=tmp_path / "lib")
    with pytest.raises(PromotionBlockedError) as exc_info:
        promoter.promote(bundle)
    assert any(r.code == "verification_failed" for r in exc_info.value.reasons)


def test_promote_rejects_skipped_required_level(tmp_path: Path) -> None:
    """Promotion requires structural + differential to actually run."""
    bundle_root = tmp_path / "bundle_src"
    bundle_root.mkdir()
    (bundle_root / "verification_report.json").write_text(
        json.dumps(
            {
                "passed": True,
                "levels_run": ["structural"],  # differential not run
                "levels_passed": ["structural"],
                "details": {"structural": "structural: PASS"},
            }
        )
    )
    bundle = Bundle(
        target_profile="t",
        model_hash="m",
        artifacts={"verification_report": "verification_report.json"},
        metadata={"bundle_root": str(bundle_root)},
    )
    promoter = RecipePromoter(library_path=tmp_path / "lib")
    with pytest.raises(PromotionBlockedError) as exc_info:
        promoter.promote(bundle)
    assert any(r.code == "level_skipped" for r in exc_info.value.reasons)


def test_promote_force_bypasses_gate(tmp_path: Path) -> None:
    """``force=True`` explicitly opts out of the verification gate."""
    bundle = Bundle(
        target_profile="t",
        model_hash="m",
        artifacts={"payload": "payload.mlir"},
    )
    promoter = RecipePromoter(library_path=tmp_path / "lib")
    result = promoter.promote(bundle, force=True)
    assert result.promoted


def test_promote_rejects_unreadable_verification_report(tmp_path: Path) -> None:
    """A mangled verification_report.json is a block reason too."""
    bundle_root = tmp_path / "bundle_src"
    bundle_root.mkdir()
    (bundle_root / "verification_report.json").write_text("not json at all {{{")
    bundle = Bundle(
        target_profile="t",
        model_hash="m",
        artifacts={"verification_report": "verification_report.json"},
        metadata={"bundle_root": str(bundle_root)},
    )
    promoter = RecipePromoter(library_path=tmp_path / "lib")
    with pytest.raises(PromotionBlockedError) as exc_info:
        promoter.promote(bundle)
    assert any(r.code == "verification_report_unreadable" for r in exc_info.value.reasons)
