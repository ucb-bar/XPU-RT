"""execution-evidence harness tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from compgen.audit.execution_evidence import (
    BLOCKED_PROOF_REASONS,
    EVIDENCE_SCHEMA_VERSION,
    BlockedProof,
    CertificateRecord,
    ExecutionEvidenceError,
    RunReport,
    audit_provider_dir,
    discover_per_provider_dirs,
    record_block,
    record_evidence,
)
from compgen.audit.extension_architecture import (
    check_execution_evidence,
    AuditReport,
    run_audit,
)


def _make_run_report(provider_id: str = "autocomp") -> RunReport:
    return RunReport(
        schema_version=EVIDENCE_SCHEMA_VERSION,
        provider_id=provider_id,
        contract_hash="abc",
        correct=True,
        latency_ms=0.5,
        device="cuda:0",
        max_abs_diff=1e-3,
        max_rel_diff=1e-3,
        samples=20,
    )


def _make_certificate(provider_id: str = "autocomp") -> CertificateRecord:
    return CertificateRecord(
        schema_version=EVIDENCE_SCHEMA_VERSION,
        provider_id=provider_id,
        contract_hash="abc",
        kernel_source_path="placeholder",
        kernel_source_sha256="placeholder",
        verifier_verdict="passed",
    )


# ---------------------------------------------------------------------------
# Schema discipline
# ---------------------------------------------------------------------------


def test_blocked_proof_requires_typed_reason():
    with pytest.raises(ExecutionEvidenceError, match="blocked_reason"):
        BlockedProof(
            schema_version=EVIDENCE_SCHEMA_VERSION,
            provider_id="x",
            status="blocked",
            blocked_reason="hand_wave",
            detail="something",
        )


def test_blocked_proof_requires_detail():
    with pytest.raises(ExecutionEvidenceError, match="detail"):
        BlockedProof(
            schema_version=EVIDENCE_SCHEMA_VERSION,
            provider_id="x",
            status="blocked",
            blocked_reason="env_missing",
            detail="",
        )


def test_blocked_proof_rejects_available_status():
    with pytest.raises(ExecutionEvidenceError, match="status"):
        BlockedProof(
            schema_version=EVIDENCE_SCHEMA_VERSION,
            provider_id="x",
            status="available",
            blocked_reason="env_missing",
            detail="x",
        )


def test_blocked_proof_reasons_enum_complete():
    assert "env_missing" in BLOCKED_PROOF_REASONS
    assert "command_missing" in BLOCKED_PROOF_REASONS
    assert "python_package_missing" in BLOCKED_PROOF_REASONS
    assert "hardware_unavailable" in BLOCKED_PROOF_REASONS
    assert "remote_unreachable" in BLOCKED_PROOF_REASONS


def test_run_report_round_trip():
    rr = _make_run_report()
    body = rr.to_dict()
    restored = RunReport.from_dict(body)
    assert restored == rr


def test_certificate_round_trip():
    cert = _make_certificate()
    body = cert.to_dict()
    restored = CertificateRecord.from_dict(body)
    assert restored == cert


def test_blocked_proof_round_trip():
    proof = BlockedProof(
        schema_version=EVIDENCE_SCHEMA_VERSION,
        provider_id="cuda_tile_ir",
        status="blocked",
        blocked_reason="env_missing",
        detail="CUDA_TILE_ROOT unset",
        missing="CUDA_TILE_ROOT",
    )
    restored = BlockedProof.from_dict(proof.to_dict())
    assert restored == proof


# ---------------------------------------------------------------------------
# record_evidence + record_block
# ---------------------------------------------------------------------------


def test_record_evidence_writes_quartet(tmp_path: Path):
    pp = record_evidence(
        evidence_pack=tmp_path,
        provider_id="autocomp",
        kernel_source="import torch\nclass ModelNew: pass\n",
        language="python",
        run_report=_make_run_report(),
        certificate=_make_certificate(),
    )
    assert pp == tmp_path / "per_provider" / "autocomp"
    assert (pp / "kernel_source.py").is_file()
    assert (pp / "run_report.json").is_file()
    assert (pp / "certificate.json").is_file()


def test_record_evidence_corrects_sha_drift(tmp_path: Path):
    """If the caller passes a stale certificate hash, ``record_evidence``
    must re-issue the certificate with the actual source's sha."""

    src = "import triton\n# different content\n"
    record_evidence(
        evidence_pack=tmp_path,
        provider_id="autocomp",
        kernel_source=src,
        language="triton",
        run_report=_make_run_report(),
        certificate=_make_certificate(),  # stub hash
    )
    cert_path = tmp_path / "per_provider" / "autocomp" / "certificate.json"
    body = json.loads(cert_path.read_text())
    import hashlib
    expected = hashlib.sha256(src.encode("utf-8")).hexdigest()
    assert body["kernel_source_sha256"] == expected


def test_record_block_writes_blocked_proof(tmp_path: Path):
    proof = BlockedProof(
        schema_version=EVIDENCE_SCHEMA_VERSION,
        provider_id="cuda_tile_ir",
        status="blocked",
        blocked_reason="env_missing",
        detail="CUDA_TILE_ROOT not set",
        missing="CUDA_TILE_ROOT",
    )
    pp = record_block(evidence_pack=tmp_path, provider_id="cuda_tile_ir", proof=proof)
    assert (pp / "blocked_proof.json").is_file()
    body = json.loads((pp / "blocked_proof.json").read_text())
    assert body["blocked_reason"] == "env_missing"


def test_record_evidence_with_remote_receipt(tmp_path: Path):
    """When a remote_receipt is supplied, it lands alongside."""

    receipt = {
        "schema_version": "remote_execution_receipt_v1",
        "host": "tpu_v5e_pod_1.example.com",
        "started_utc": "2026-05-12T06:00:00Z",
        "finished_utc": "2026-05-12T06:00:30Z",
    }
    pp = record_evidence(
        evidence_pack=tmp_path,
        provider_id="pallas",
        kernel_source="import jax\n",
        language="python",
        run_report=_make_run_report(provider_id="pallas"),
        certificate=_make_certificate(provider_id="pallas"),
        remote_receipt=receipt,
    )
    assert (pp / "remote_receipt.json").is_file()


# ---------------------------------------------------------------------------
# audit_provider_dir + discover_per_provider_dirs
# ---------------------------------------------------------------------------


def test_audit_recognizes_available_state(tmp_path: Path):
    record_evidence(
        evidence_pack=tmp_path,
        provider_id="autocomp",
        kernel_source="x",
        language="python",
        run_report=_make_run_report(),
        certificate=_make_certificate(),
    )
    state, _ = audit_provider_dir(tmp_path / "per_provider" / "autocomp")
    assert state == "available"


def test_audit_recognizes_blocked_state(tmp_path: Path):
    record_block(
        evidence_pack=tmp_path,
        provider_id="cuda_tile_ir",
        proof=BlockedProof(
            schema_version=EVIDENCE_SCHEMA_VERSION,
            provider_id="cuda_tile_ir",
            status="blocked",
            blocked_reason="env_missing",
            detail="x",
        ),
    )
    state, _ = audit_provider_dir(tmp_path / "per_provider" / "cuda_tile_ir")
    assert state == "blocked"


def test_audit_recognizes_empty_state(tmp_path: Path):
    (tmp_path / "per_provider" / "empty_provider").mkdir(parents=True)
    state, detail = audit_provider_dir(
        tmp_path / "per_provider" / "empty_provider"
    )
    assert state == "empty"
    assert detail


def test_audit_recognizes_malformed_state(tmp_path: Path):
    bad_dir = tmp_path / "per_provider" / "bad_provider"
    bad_dir.mkdir(parents=True)
    (bad_dir / "kernel_source.py").write_text("x")
    (bad_dir / "run_report.json").write_text("not json")
    (bad_dir / "certificate.json").write_text("{}")
    state, detail = audit_provider_dir(bad_dir)
    assert state == "malformed"


def test_discover_per_provider_dirs(tmp_path: Path):
    record_evidence(
        evidence_pack=tmp_path,
        provider_id="autocomp",
        kernel_source="x",
        language="python",
        run_report=_make_run_report(),
        certificate=_make_certificate(),
    )
    record_block(
        evidence_pack=tmp_path,
        provider_id="cuda_tile_ir",
        proof=BlockedProof(
            schema_version=EVIDENCE_SCHEMA_VERSION,
            provider_id="cuda_tile_ir",
            status="blocked",
            blocked_reason="env_missing",
            detail="x",
        ),
    )
    dirs = discover_per_provider_dirs(tmp_path)
    assert len(dirs) == 2
    assert {d.name for d in dirs} == {"autocomp", "cuda_tile_ir"}


# ---------------------------------------------------------------------------
# Audit integration
# ---------------------------------------------------------------------------


def test_run_audit_includes_execution_evidence_check(tmp_path: Path):
    record_evidence(
        evidence_pack=tmp_path,
        provider_id="autocomp",
        kernel_source="x",
        language="python",
        run_report=_make_run_report(),
        certificate=_make_certificate(),
    )
    report = run_audit(evidence_pack=tmp_path)
    assert "execution_evidence" in report.checks_run
    assert report.summary.get("execution_evidence_available", 0) >= 1
    assert report.passed


def test_run_audit_fails_on_malformed_evidence(tmp_path: Path):
    bad_dir = tmp_path / "per_provider" / "rogue"
    bad_dir.mkdir(parents=True)
    (bad_dir / "run_report.json").write_text("not json")
    (bad_dir / "kernel_source.py").write_text("x")
    (bad_dir / "certificate.json").write_text("{}")
    report = run_audit(evidence_pack=tmp_path)
    violations = [
        v for v in report.violations if v.check == "execution_evidence"
    ]
    assert violations, "expected execution_evidence violation"
    assert violations[0].reason == "execution_evidence_malformed"
    assert not report.passed


def test_run_audit_accepts_missing_per_provider_dir():
    """If no per_provider/ exists yet, the check should not violate —
    it only fails on present-but-malformed dirs."""

    report = run_audit(evidence_pack=Path("/nonexistent/path/xyz"))
    violations = [
        v for v in report.violations if v.check == "execution_evidence"
    ]
    assert not violations
