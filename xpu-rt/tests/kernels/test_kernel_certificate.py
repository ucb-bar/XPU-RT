"""M-45 kernel certificate tests.

Layered coverage:
- Schema round-trip (to_dict / from_dict).
- Emit produces a cert at the canonical path keyed by contract_hash.
- validate_certificate returns valid=True on a fresh cert and valid=False
  with typed failure_kind on:
    * artifact_hash drift (kernel.c edited post-cert)
    * verifier_report drift (validation report edited post-cert)
    * missing artifact / missing verifier report
- The cert's paper_claimable bit reflects fallback usage.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from compgen.kernels.kernel_certificate import (
    KernelCertificate,
    certificate_path_for,
    emit_certificate,
    load_certificate,
    validate_certificate,
)


def _setup_run_dir(tmp_path: Path) -> tuple[Path, dict, dict, Path]:
    """Build a minimal run_dir with a request, a sandboxed artifact
    set, and a validation report. Returns
    (run_dir, request_body, response_body, validation_report_path)."""
    run_dir = tmp_path / "run"
    requests = run_dir / "04_kernel_codegen" / "requests"
    artifact_dir = run_dir / "04_kernel_codegen" / "artifacts" / "kspec_test"
    validation_dir = run_dir / "04_kernel_codegen" / "validation"
    contracts = run_dir / "04_kernel_codegen" / "contracts"
    for d in (requests, artifact_dir, validation_dir, contracts):
        d.mkdir(parents=True, exist_ok=True)

    contract_path = contracts / "matmul_0.cafe1234cafe1234.json"
    contract_path.write_text(json.dumps({"hash": "cafe1234cafe1234"}))

    request_body = {
        "task_id": "kspec_test",
        "contract_hash": "cafe1234cafe1234",
        "region_id": "matmul_0",
        "candidate_id": "cand_test",
        "contract_paths": {"full": "04_kernel_codegen/contracts/matmul_0.cafe1234cafe1234.json"},
    }
    (requests / "kspec_test.request.json").write_text(json.dumps(request_body))

    # Synthetic artifacts.
    artifacts = {}
    for name, content in (
        ("kernel_source", "/* synthetic kernel */\n"),
        ("kernel_metadata", "{}\n"),
        ("launch_config", "{}\n"),
        ("provider_claims", "{}\n"),
    ):
        ext = ".c" if name == "kernel_source" else ".json"
        path = artifact_dir / f"{name}{ext}"
        path.write_text(content)
        artifacts[name] = str(path.relative_to(run_dir))

    response_body = {
        "task_id": "kspec_test",
        "contract_hash": "cafe1234cafe1234",
        "artifacts": artifacts,
        "claims": {"backend": "c_reference", "expected_numerics": "bit_equality"},
        "provider": {"kind": "test"},
    }

    validation_report_path = validation_dir / "kspec_test.validation.json"
    validation_report_path.write_text(json.dumps({"overall": "pass"}))

    return run_dir, request_body, response_body, validation_report_path


# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #


class TestSchema:
    def test_to_from_dict_round_trip(self) -> None:
        c = KernelCertificate(
            schema_version="kernel_certificate_v1",
            contract_hash="abc",
            task_id="kspec_x",
            region_id="r",
            candidate_id="c",
            accepted_at_utc="2026-05-07T00:00:00Z",
            artifact_hashes={"kernel_source": "h1"},
            artifact_paths={"kernel_source": "p1"},
            verifier_report_path="vr",
            verifier_report_hash="vh",
            claims={"backend": "c_reference"},
        )
        d = c.to_dict()
        c2 = KernelCertificate.from_dict(d)
        assert c == c2

    def test_paper_claimable_defaults_true(self) -> None:
        c = KernelCertificate(
            schema_version="kernel_certificate_v1",
            contract_hash="x", task_id="t", region_id="r", candidate_id="c",
            accepted_at_utc="t", artifact_hashes={}, artifact_paths={},
            verifier_report_path="", verifier_report_hash="", claims={},
        )
        assert c.paper_claimable is True
        assert c.fallback_used is False


# --------------------------------------------------------------------------- #
# Emit + load
# --------------------------------------------------------------------------- #


class TestEmit:
    def test_emit_writes_to_canonical_path(self, tmp_path: Path) -> None:
        run_dir, req, resp, vr_path = _setup_run_dir(tmp_path)
        cert_path = emit_certificate(
            run_dir=run_dir, request_body=req, response_body=resp,
            verifier_report_path=vr_path,
        )
        expected = certificate_path_for(
            run_dir=run_dir, contract_hash=req["contract_hash"],
        )
        assert cert_path == expected
        assert cert_path.exists()

    def test_emit_records_artifact_hashes(self, tmp_path: Path) -> None:
        run_dir, req, resp, vr_path = _setup_run_dir(tmp_path)
        emit_certificate(
            run_dir=run_dir, request_body=req, response_body=resp,
            verifier_report_path=vr_path,
        )
        cert = load_certificate(
            run_dir=run_dir, contract_hash=req["contract_hash"],
        )
        assert cert is not None
        assert set(cert.artifact_hashes) == {"kernel_source", "kernel_metadata",
                                              "launch_config", "provider_claims"}
        for h in cert.artifact_hashes.values():
            # sha256[:16] = 16 hex chars
            assert len(h) == 16
        assert cert.verifier_report_hash
        assert len(cert.verifier_report_hash) == 16

    def test_emit_with_fallback_marks_non_paper_claimable(self, tmp_path: Path) -> None:
        run_dir, req, resp, vr_path = _setup_run_dir(tmp_path)
        emit_certificate(
            run_dir=run_dir, request_body=req, response_body=resp,
            verifier_report_path=vr_path,
            fallback_used=True, fallback_reason="provider exhausted, fell back to reference",
        )
        cert = load_certificate(
            run_dir=run_dir, contract_hash=req["contract_hash"],
        )
        assert cert.paper_claimable is False
        assert cert.fallback_used is True
        assert "exhausted" in cert.fallback_reason


# --------------------------------------------------------------------------- #
# Negative controls — drift detection (the M-37.13 pattern)
# --------------------------------------------------------------------------- #


class TestValidationNegativeControls:
    def test_fresh_certificate_validates(self, tmp_path: Path) -> None:
        run_dir, req, resp, vr_path = _setup_run_dir(tmp_path)
        emit_certificate(
            run_dir=run_dir, request_body=req, response_body=resp,
            verifier_report_path=vr_path,
        )
        cert = load_certificate(
            run_dir=run_dir, contract_hash=req["contract_hash"],
        )
        result = validate_certificate(run_dir=run_dir, cert=cert)
        assert result.valid is True
        assert result.failure_kind == ""

    def test_artifact_hash_drift_detected(self, tmp_path: Path) -> None:
        run_dir, req, resp, vr_path = _setup_run_dir(tmp_path)
        emit_certificate(
            run_dir=run_dir, request_body=req, response_body=resp,
            verifier_report_path=vr_path,
        )
        # Tamper kernel_source AFTER the cert was emitted.
        kernel_path = run_dir / resp["artifacts"]["kernel_source"]
        kernel_path.write_text("/* TAMPERED — different source */\n")
        cert = load_certificate(
            run_dir=run_dir, contract_hash=req["contract_hash"],
        )
        result = validate_certificate(run_dir=run_dir, cert=cert)
        assert result.valid is False
        assert result.failure_kind == "artifact_hash_drift"
        assert "kernel_source" in result.drifted

    def test_verifier_report_drift_detected(self, tmp_path: Path) -> None:
        run_dir, req, resp, vr_path = _setup_run_dir(tmp_path)
        emit_certificate(
            run_dir=run_dir, request_body=req, response_body=resp,
            verifier_report_path=vr_path,
        )
        # Tamper validation report.
        vr_path.write_text(json.dumps({"overall": "pass", "tampered": True}))
        cert = load_certificate(
            run_dir=run_dir, contract_hash=req["contract_hash"],
        )
        result = validate_certificate(run_dir=run_dir, cert=cert)
        assert result.valid is False
        assert result.failure_kind == "verifier_report_drift"

    def test_missing_artifact_detected(self, tmp_path: Path) -> None:
        run_dir, req, resp, vr_path = _setup_run_dir(tmp_path)
        emit_certificate(
            run_dir=run_dir, request_body=req, response_body=resp,
            verifier_report_path=vr_path,
        )
        kernel_path = run_dir / resp["artifacts"]["kernel_source"]
        kernel_path.unlink()
        cert = load_certificate(
            run_dir=run_dir, contract_hash=req["contract_hash"],
        )
        result = validate_certificate(run_dir=run_dir, cert=cert)
        assert result.valid is False
        assert result.failure_kind == "missing_artifact"

    def test_missing_verifier_report_detected(self, tmp_path: Path) -> None:
        run_dir, req, resp, vr_path = _setup_run_dir(tmp_path)
        emit_certificate(
            run_dir=run_dir, request_body=req, response_body=resp,
            verifier_report_path=vr_path,
        )
        vr_path.unlink()
        cert = load_certificate(
            run_dir=run_dir, contract_hash=req["contract_hash"],
        )
        result = validate_certificate(run_dir=run_dir, cert=cert)
        assert result.valid is False
        assert result.failure_kind == "missing_verifier_report"


# --------------------------------------------------------------------------- #
# Lookup
# --------------------------------------------------------------------------- #


def test_load_certificate_returns_none_for_missing(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    cert = load_certificate(run_dir=run_dir, contract_hash="does_not_exist")
    assert cert is None
