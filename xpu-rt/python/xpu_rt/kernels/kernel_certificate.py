"""Kernel certificate (M-45).

Phase C M-45. The certificate is the single load-bearing artifact M-46
(plan ↔ kernel binding) consults to decide whether a kernel is
trustworthy enough to call. M-43's commit tool emits the certificate
when M-44's contract-driven verifier accepts; M-46+ refuse to bind a
kernel without a matching certificate.

The certificate binds:

::

    contract_hash               (the M-26 / M-41 canonical key)
    task_id                     (links back to 04_kernel_codegen/requests/<task_id>.request.json)
    artifact_hashes             (per-artifact sha256[:16] at certify time)
    verifier_report_hash        (sha256[:16] of the validation report)
    claims                      (the provider's claims block, frozen)
    accepted_at_utc             (timestamp)
    paper_claimable             (true ONLY when no fallback was used)

Certificate validation (``validate_certificate``):

  - re-hashes every kernel artifact and compares to the cert's
    ``artifact_hashes``. Any drift is a typed ``artifact_hash_drift``
    failure (mirrors M-37.13's ``certificate_artifact_hash_changed``
    negative control pattern).
  - re-hashes the validation report and compares to
    ``verifier_report_hash``. Any drift means the verifier report
    was edited post-certify.
  - checks that all referenced files still exist.

The cert lives at ``04_kernel_codegen/certificates/<contract_hash>.json``.
The same contract_hash → same cert path; M-46 looks up by hash.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


_CERT_SCHEMA_VERSION = "kernel_certificate_v1"


# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class KernelCertificate:
    schema_version: str
    contract_hash: str
    task_id: str
    region_id: str
    candidate_id: str
    accepted_at_utc: str

    # Hashes of every artifact the cert vouches for.
    artifact_hashes: dict[str, str]   # name → sha256[:16]
    artifact_paths: dict[str, str]    # name → relative path

    # Validation report this cert is signed against.
    verifier_report_path: str
    verifier_report_hash: str

    # Frozen provider claims at accept time (M-46+ reads these).
    claims: dict[str, Any]

    # Paper-claimable only when no fallback was used.
    paper_claimable: bool = True
    fallback_used: bool = False
    fallback_reason: str = ""

    # Pointers back to the materialised contract + request.
    contract_path: str = ""
    request_path: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "contract_hash": self.contract_hash,
            "task_id": self.task_id,
            "region_id": self.region_id,
            "candidate_id": self.candidate_id,
            "accepted_at_utc": self.accepted_at_utc,
            "artifact_hashes": dict(self.artifact_hashes),
            "artifact_paths": dict(self.artifact_paths),
            "verifier_report_path": self.verifier_report_path,
            "verifier_report_hash": self.verifier_report_hash,
            "claims": dict(self.claims),
            "paper_claimable": self.paper_claimable,
            "fallback_used": self.fallback_used,
            "fallback_reason": self.fallback_reason,
            "contract_path": self.contract_path,
            "request_path": self.request_path,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "KernelCertificate":
        return cls(
            schema_version=str(data.get("schema_version", _CERT_SCHEMA_VERSION)),
            contract_hash=str(data["contract_hash"]),
            task_id=str(data["task_id"]),
            region_id=str(data.get("region_id", "")),
            candidate_id=str(data.get("candidate_id", "")),
            accepted_at_utc=str(data.get("accepted_at_utc", "")),
            artifact_hashes=dict(data.get("artifact_hashes") or {}),
            artifact_paths=dict(data.get("artifact_paths") or {}),
            verifier_report_path=str(data.get("verifier_report_path", "")),
            verifier_report_hash=str(data.get("verifier_report_hash", "")),
            claims=dict(data.get("claims") or {}),
            paper_claimable=bool(data.get("paper_claimable", True)),
            fallback_used=bool(data.get("fallback_used", False)),
            fallback_reason=str(data.get("fallback_reason", "")),
            contract_path=str(data.get("contract_path", "")),
            request_path=str(data.get("request_path", "")),
        )


@dataclass(frozen=True)
class CertificateValidation:
    valid: bool
    failure_kind: str = ""           # "artifact_hash_drift" | "missing_artifact"
                                     # | "verifier_report_drift"
                                     # | "missing_verifier_report"
    failure_summary: str = ""
    drifted: dict[str, str] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _utcnow() -> str:
    return datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _hash_file(path: Path) -> str:
    """sha256[:16] of file contents — same shorthash convention as
    M-37.13 verification certificates."""
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _certificate_dir(run_dir: Path) -> Path:
    return run_dir / "04_kernel_codegen" / "certificates"


def certificate_path_for(*, run_dir: Path, contract_hash: str) -> Path:
    """The single canonical path the certificate lives at, indexed by
    contract_hash (M-46 looks up here)."""
    return _certificate_dir(run_dir) / f"{contract_hash}.json"


# --------------------------------------------------------------------------- #
# Emit
# --------------------------------------------------------------------------- #


def emit_certificate(
    *,
    run_dir: Path,
    request_body: dict[str, Any],
    response_body: dict[str, Any],
    verifier_report_path: Path,
    fallback_used: bool = False,
    fallback_reason: str = "",
) -> Path:
    """Build and persist a ``KernelCertificate`` for an accepted kernel
    response. Returns the cert path.

    The cert is keyed by contract_hash, so re-emission for the same
    contract overwrites — the fresher cert is the source of truth.
    Caller invariant: only call after M-44 verifier returns
    overall=pass.
    """
    run_dir = Path(run_dir).resolve()
    contract_hash = request_body["contract_hash"]
    task_id = request_body["task_id"]

    artifact_hashes: dict[str, str] = {}
    artifact_paths: dict[str, str] = {}
    for name, rel_path in (response_body.get("artifacts") or {}).items():
        full = run_dir / rel_path
        artifact_hashes[name] = _hash_file(full)
        artifact_paths[name] = rel_path

    verifier_report_hash = _hash_file(verifier_report_path)

    cert = KernelCertificate(
        schema_version=_CERT_SCHEMA_VERSION,
        contract_hash=contract_hash,
        task_id=task_id,
        region_id=request_body.get("region_id", ""),
        candidate_id=request_body.get("candidate_id", ""),
        accepted_at_utc=_utcnow(),
        artifact_hashes=artifact_hashes,
        artifact_paths=artifact_paths,
        verifier_report_path=str(verifier_report_path.relative_to(run_dir)),
        verifier_report_hash=verifier_report_hash,
        claims=dict(response_body.get("claims") or {}),
        paper_claimable=not fallback_used,
        fallback_used=fallback_used,
        fallback_reason=fallback_reason,
        contract_path=request_body.get("contract_paths", {}).get("full", ""),
        request_path=str(
            (run_dir / "04_kernel_codegen" / "requests" / f"{task_id}.request.json")
            .relative_to(run_dir)
        ),
    )

    out_dir = _certificate_dir(run_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = certificate_path_for(run_dir=run_dir, contract_hash=contract_hash)
    path.write_text(
        json.dumps(cert.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def load_certificate(
    *, run_dir: Path, contract_hash: str,
) -> KernelCertificate | None:
    path = certificate_path_for(run_dir=run_dir, contract_hash=contract_hash)
    if not path.exists():
        return None
    return KernelCertificate.from_dict(json.loads(path.read_text(encoding="utf-8")))


# --------------------------------------------------------------------------- #
# Validate (re-hash artifacts + verifier report)
# --------------------------------------------------------------------------- #


def validate_certificate(
    *, run_dir: Path, cert: KernelCertificate,
) -> CertificateValidation:
    """Re-hash every artifact + the verifier report and compare to
    the cert's recorded hashes. Catches the M-37.13-style negative
    control: edit the kernel artifact post-certify → cert no longer
    validates → M-46 refuses to bind."""
    run_dir = Path(run_dir).resolve()
    drifted: dict[str, str] = {}

    # Verifier report.
    report_path = run_dir / cert.verifier_report_path
    if not report_path.exists():
        return CertificateValidation(
            valid=False, failure_kind="missing_verifier_report",
            failure_summary=(
                f"verifier report missing at {cert.verifier_report_path}"
            ),
        )
    actual_report_hash = _hash_file(report_path)
    if actual_report_hash != cert.verifier_report_hash:
        return CertificateValidation(
            valid=False, failure_kind="verifier_report_drift",
            failure_summary=(
                f"verifier report hash drift: expected "
                f"{cert.verifier_report_hash}, got {actual_report_hash}"
            ),
            drifted={"verifier_report": actual_report_hash},
        )

    # Each artifact.
    for name, rel_path in cert.artifact_paths.items():
        full = run_dir / rel_path
        if not full.exists():
            return CertificateValidation(
                valid=False, failure_kind="missing_artifact",
                failure_summary=(
                    f"artifact {name!r} missing at {rel_path}"
                ),
            )
        actual = _hash_file(full)
        expected = cert.artifact_hashes.get(name, "")
        if actual != expected:
            drifted[name] = actual

    if drifted:
        return CertificateValidation(
            valid=False, failure_kind="artifact_hash_drift",
            failure_summary=(
                f"{len(drifted)} artifact(s) drifted: "
                + ", ".join(f"{n} expected={cert.artifact_hashes.get(n, '')!r} got={h!r}"
                            for n, h in drifted.items())
            ),
            drifted=drifted,
        )
    return CertificateValidation(valid=True)
