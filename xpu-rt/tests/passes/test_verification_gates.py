"""Tests for compgen.passes.verification."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from compgen.audit.errors import (
    VerificationGateFailed,
    VerificationGateMissing,
)
from compgen.passes.cards import (
    PassCard, default_registry_root, load_card, resolve_card_path,
)
from compgen.passes.verification import (
    ALLOWED_RUNGS,
    CERT_SCHEMA_VERSION,
    SKIPPABLE_RUNGS,
    VerificationCertificate,
    assert_certificate_artifact_fresh,
    assert_required_rungs_discharged,
    emit_certificate_from_differential_report,
    emit_certificate_from_post_lowering_report,
    emit_skipped_formal_certificate,
    load_certificate,
    write_certificate,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _make_post_lowering_report(run_dir: Path, *, status: str = "pass") -> Path:
    out_dir = run_dir / "03_recipe_planning" / "post_lowering"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "post_lowering_verification_report.json"
    report_path.write_text(json.dumps({
        "schema_version": "post_lowering_verification_report_v1",
        "status": status,
        "model_id": "test_model",
        "checks": [],
    }))
    # Sibling artifact for hashing
    (out_dir / "transformed_payload.mlir").write_text("// transformed payload")
    return report_path


def _make_diff_report(run_dir: Path, *, status: str = "pass") -> Path:
    out_dir = run_dir / "03_recipe_planning" / "differential_verification"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "differential_verification_report.json"
    report_path.write_text(json.dumps({
        "schema_version": "differential_verification_report_v1",
        "status": status,
        "checks": [],
    }))
    # The artifact that diff verifies sits under post_lowering/
    pl = run_dir / "03_recipe_planning" / "post_lowering"
    pl.mkdir(parents=True, exist_ok=True)
    (pl / "transformed_payload.mlir").write_text("// transformed payload")
    return report_path


# --------------------------------------------------------------------------- #
# Schema / round-trip
# --------------------------------------------------------------------------- #


def test_certificate_round_trip(tmp_path: Path) -> None:
    cert = VerificationCertificate(
        schema_version=CERT_SCHEMA_VERSION,
        rung="structural",
        status="pass",
        pass_id="set_tile_params",
        artifact_path="03_recipe_planning/post_lowering/transformed_payload.mlir",
        artifact_hash="abcdef1234567890",
        report_path="03_recipe_planning/post_lowering/post_lowering_verification_report.json",
        generated_at_utc="2026-05-05T00:00:00Z",
    )
    out = write_certificate(cert, run_dir=tmp_path)
    assert out.exists()
    reloaded = load_certificate(tmp_path, "structural")
    assert reloaded == cert


def test_allowed_rungs_match_spec() -> None:
    assert ALLOWED_RUNGS == ("structural", "differential", "formal")


def test_skippable_rungs_today_includes_only_formal() -> None:
    assert SKIPPABLE_RUNGS == frozenset({"formal"})


# --------------------------------------------------------------------------- #
# Emitters
# --------------------------------------------------------------------------- #


def test_emit_post_lowering_certificate_pass(tmp_path: Path) -> None:
    _make_post_lowering_report(tmp_path, status="pass")
    cert = emit_certificate_from_post_lowering_report(
        run_dir=tmp_path, pass_id="set_tile_params",
    )
    assert cert is not None
    assert cert.rung == "structural"
    assert cert.status == "pass"
    assert cert.pass_id == "set_tile_params"
    assert cert.artifact_hash != ""


def test_emit_post_lowering_certificate_fail_propagates(tmp_path: Path) -> None:
    _make_post_lowering_report(tmp_path, status="fail")
    cert = emit_certificate_from_post_lowering_report(run_dir=tmp_path)
    assert cert is not None
    assert cert.status == "fail"


def test_emit_post_lowering_certificate_returns_none_when_report_absent(
    tmp_path: Path,
) -> None:
    cert = emit_certificate_from_post_lowering_report(run_dir=tmp_path)
    assert cert is None


def test_emit_differential_certificate(tmp_path: Path) -> None:
    _make_diff_report(tmp_path, status="pass")
    cert = emit_certificate_from_differential_report(
        run_dir=tmp_path, pass_id="set_tile_params",
    )
    assert cert is not None
    assert cert.rung == "differential"
    assert cert.status == "pass"


def test_emit_skipped_formal_certificate(tmp_path: Path) -> None:
    cert = emit_skipped_formal_certificate(run_dir=tmp_path)
    assert cert.rung == "formal"
    assert cert.status == "skipped"
    assert cert.skipped_reason


# --------------------------------------------------------------------------- #
# Enforcement
# --------------------------------------------------------------------------- #


def _card_with_rungs(rungs: list[str]) -> PassCard:
    return PassCard(
        schema_version="pass_card_v1",
        pass_id="demo",
        display_name="demo",
        level="payload",
        family="tiling",
        reads=("a.json",),
        writes=("b.json",),
        preconditions=("x",),
        invalidates=("payload_summary",),
        preserves_refinement="bit_equality",
        verification=tuple(rungs),
        cost="cheap",
        failure_modes=("x",),
    )


def test_assert_passes_when_all_certs_present_and_passing(tmp_path: Path) -> None:
    _make_post_lowering_report(tmp_path, status="pass")
    _make_diff_report(tmp_path, status="pass")
    emit_certificate_from_post_lowering_report(run_dir=tmp_path)
    emit_certificate_from_differential_report(run_dir=tmp_path)
    card = _card_with_rungs(["structural", "differential"])
    assert_required_rungs_discharged(card, tmp_path)  # no raise


def test_assert_raises_missing_when_no_cert(tmp_path: Path) -> None:
    card = _card_with_rungs(["structural"])
    with pytest.raises(VerificationGateMissing, match="structural"):
        assert_required_rungs_discharged(card, tmp_path)


def test_assert_raises_failed_when_cert_reports_fail(tmp_path: Path) -> None:
    _make_post_lowering_report(tmp_path, status="fail")
    emit_certificate_from_post_lowering_report(run_dir=tmp_path)
    card = _card_with_rungs(["structural"])
    with pytest.raises(VerificationGateFailed, match="status=fail"):
        assert_required_rungs_discharged(card, tmp_path)


def test_skipped_formal_passes_the_gate(tmp_path: Path) -> None:
    """``formal`` is in SKIPPABLE_RUNGS so a skipped cert is accepted."""
    emit_skipped_formal_certificate(run_dir=tmp_path)
    card = _card_with_rungs(["formal"])
    assert_required_rungs_discharged(card, tmp_path)


def test_skipped_structural_does_not_pass_the_gate(tmp_path: Path) -> None:
    """``structural`` is NOT in SKIPPABLE_RUNGS so a skipped cert fails."""
    cert = VerificationCertificate(
        schema_version=CERT_SCHEMA_VERSION,
        rung="structural",
        status="skipped",
        pass_id="demo",
        artifact_path="",
        artifact_hash="",
        report_path="",
        generated_at_utc="2026-05-05T00:00:00Z",
        skipped_reason="dev shortcut",
    )
    write_certificate(cert, run_dir=tmp_path)
    card = _card_with_rungs(["structural"])
    with pytest.raises(VerificationGateFailed, match="skipped"):
        assert_required_rungs_discharged(card, tmp_path)


# --------------------------------------------------------------------------- #
# Artifact freshness
# --------------------------------------------------------------------------- #


def test_artifact_freshness_passes_when_unchanged(tmp_path: Path) -> None:
    _make_post_lowering_report(tmp_path, status="pass")
    cert = emit_certificate_from_post_lowering_report(run_dir=tmp_path)
    assert cert is not None
    assert_certificate_artifact_fresh(cert, tmp_path)  # no raise


def test_artifact_freshness_raises_when_artifact_mutated(tmp_path: Path) -> None:
    _make_post_lowering_report(tmp_path, status="pass")
    cert = emit_certificate_from_post_lowering_report(run_dir=tmp_path)
    assert cert is not None
    # Mutate the artifact AFTER the certificate is emitted
    payload = tmp_path / "03_recipe_planning" / "post_lowering" / "transformed_payload.mlir"
    payload.write_text("// tampered")
    with pytest.raises(VerificationGateFailed, match="mutated since"):
        assert_certificate_artifact_fresh(cert, tmp_path)


def test_artifact_freshness_skips_certificates_without_artifact(tmp_path: Path) -> None:
    cert = emit_skipped_formal_certificate(run_dir=tmp_path)
    # No artifact_path → no-op
    assert_certificate_artifact_fresh(cert, tmp_path)


# --------------------------------------------------------------------------- #
# Real seed-card integration
# --------------------------------------------------------------------------- #


def test_seed_card_set_tile_params_required_rungs() -> None:
    card = load_card(resolve_card_path("set_tile_params"))
    # The card declares structural + differential
    assert "structural" in card.verification
    assert "differential" in card.verification


def test_seed_card_fuse_producer_consumer_required_rungs() -> None:
    card = load_card(resolve_card_path("fuse_producer_consumer"))
    assert "structural" in card.verification
    assert "differential" in card.verification
