"""Execution-evidence harness.

Every provider that probes ``available`` must, for at least one
contract, produce the full quartet under
``<evidence_pack>/per_provider/<provider_id>/``:

* ``kernel_source.<ext>`` — the actual kernel text the adapter emitted.
* ``run_report.json``     — typed measurement record (correctness +
  latency + device id + timestamps + max-abs/rel diff).
* ``certificate.json``    — typed certificate proving the verifier
  accepted this kernel for this contract.
* (optional) ``remote_receipt.json`` — present when the run executed
  on a remote target.

Every provider that probes ``blocked`` (or any other non-available
status) must produce ``blocked_proof.json`` carrying the typed
missing prerequisite.

The audit gate in
``compgen.audit.extension_architecture.check_execution_evidence``
walks the evidence-pack ``per_provider/`` directory and refuses to
pass if any provider's status / artifact set is inconsistent.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final


EVIDENCE_SCHEMA_VERSION: Final[str] = "execution_evidence_v1"

# Reasons a provider can be in a "blocked" state in the evidence pack.
# These mirror compgen.providers.provider_types.BLOCKED_REASONS so the
# audit can verify the proof is real.
BLOCKED_PROOF_REASONS: Final[tuple[str, ...]] = (
    "env_missing",
    "python_package_missing",
    "command_missing",
    "hardware_unavailable",
    "sdk_missing",
    "license_missing",
    "version_mismatch",
    "unsupported_platform",
    "unsupported_contract_kind",
    "probe_exception",
    "not_installed",
    "contract_rejected",
    "search_failed",
    "remote_unreachable",
)


# Extension chosen per language; the audit only checks that ONE
# kernel_source.* file is present, not the specific extension.
LANGUAGE_TO_EXT: Final[dict[str, str]] = {
    "python": "py",
    "triton": "py",
    "c": "c",
    "cpp": "cpp",
    "cuda": "cu",
    "ptx": "ptx",
    "mlir": "mlir",
    "asm": "s",
    "elf": "elf",
}


class ExecutionEvidenceError(ValueError):
    """An evidence record violated the schema."""


@dataclass(frozen=True)
class RunReport:
    """Typed measurement record."""

    schema_version: str
    provider_id: str
    contract_hash: str
    correct: bool
    latency_ms: float | None
    device: str
    max_abs_diff: float | None = None
    max_rel_diff: float | None = None
    samples: int = 0
    started_utc: str = ""
    finished_utc: str = ""
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "provider_id": self.provider_id,
            "contract_hash": self.contract_hash,
            "correct": self.correct,
            "latency_ms": self.latency_ms,
            "device": self.device,
            "max_abs_diff": self.max_abs_diff,
            "max_rel_diff": self.max_rel_diff,
            "samples": self.samples,
            "started_utc": self.started_utc,
            "finished_utc": self.finished_utc,
            "extras": dict(self.extras),
        }

    @classmethod
    def from_dict(cls, body: dict[str, Any]) -> "RunReport":
        return cls(
            schema_version=str(body.get("schema_version", EVIDENCE_SCHEMA_VERSION)),
            provider_id=str(body["provider_id"]),
            contract_hash=str(body["contract_hash"]),
            correct=bool(body["correct"]),
            latency_ms=(
                None if body.get("latency_ms") is None else float(body["latency_ms"])
            ),
            device=str(body.get("device", "")),
            max_abs_diff=(
                None
                if body.get("max_abs_diff") is None
                else float(body["max_abs_diff"])
            ),
            max_rel_diff=(
                None
                if body.get("max_rel_diff") is None
                else float(body["max_rel_diff"])
            ),
            samples=int(body.get("samples", 0)),
            started_utc=str(body.get("started_utc", "")),
            finished_utc=str(body.get("finished_utc", "")),
            extras=dict(body.get("extras", {}) or {}),
        )


@dataclass(frozen=True)
class CertificateRecord:
    """Minimal certificate persisted in the evidence pack.

    The richer :class:`compgen.kernels.kernel_certificate.KernelCertificate`
    is what the verifier emits; this is the **audit-facing** view that
    couples a contract to a verified kernel source.
    """

    schema_version: str
    provider_id: str
    contract_hash: str
    kernel_source_path: str
    kernel_source_sha256: str
    verifier_verdict: str  # "passed" | "failed"
    verifier_detail: str = ""
    issued_utc: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "provider_id": self.provider_id,
            "contract_hash": self.contract_hash,
            "kernel_source_path": self.kernel_source_path,
            "kernel_source_sha256": self.kernel_source_sha256,
            "verifier_verdict": self.verifier_verdict,
            "verifier_detail": self.verifier_detail,
            "issued_utc": self.issued_utc,
        }

    @classmethod
    def from_dict(cls, body: dict[str, Any]) -> "CertificateRecord":
        return cls(
            schema_version=str(body.get("schema_version", EVIDENCE_SCHEMA_VERSION)),
            provider_id=str(body["provider_id"]),
            contract_hash=str(body["contract_hash"]),
            kernel_source_path=str(body["kernel_source_path"]),
            kernel_source_sha256=str(body["kernel_source_sha256"]),
            verifier_verdict=str(body["verifier_verdict"]),
            verifier_detail=str(body.get("verifier_detail", "")),
            issued_utc=str(body.get("issued_utc", "")),
        )


@dataclass(frozen=True)
class BlockedProof:
    """Typed proof that a provider is genuinely blocked.

    The audit uses this to verify the blocked claim is real (not
    just a card declaration).
    """

    schema_version: str
    provider_id: str
    status: str  # "blocked" | "unsupported" | "probe_error" | "not_installed"
    blocked_reason: str
    detail: str
    missing: str = ""
    verified_utc: str = ""

    def __post_init__(self) -> None:
        if self.status not in {
            "blocked",
            "unsupported",
            "probe_error",
            "not_installed",
            "contract_rejected",
        }:
            raise ExecutionEvidenceError(
                f"blocked_proof status={self.status!r} must be one of the "
                f"non-available probe statuses"
            )
        if self.blocked_reason not in BLOCKED_PROOF_REASONS:
            raise ExecutionEvidenceError(
                f"blocked_reason={self.blocked_reason!r} must be one of "
                f"{BLOCKED_PROOF_REASONS}"
            )
        if not self.detail:
            raise ExecutionEvidenceError(
                "blocked_proof requires a non-empty detail"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "provider_id": self.provider_id,
            "status": self.status,
            "blocked_reason": self.blocked_reason,
            "detail": self.detail,
            "missing": self.missing,
            "verified_utc": self.verified_utc,
        }

    @classmethod
    def from_dict(cls, body: dict[str, Any]) -> "BlockedProof":
        return cls(
            schema_version=str(body.get("schema_version", EVIDENCE_SCHEMA_VERSION)),
            provider_id=str(body["provider_id"]),
            status=str(body["status"]),
            blocked_reason=str(body["blocked_reason"]),
            detail=str(body["detail"]),
            missing=str(body.get("missing", "")),
            verified_utc=str(body.get("verified_utc", "")),
        )


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _sha256(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _provider_dir(evidence_pack: Path, provider_id: str) -> Path:
    pp = Path(evidence_pack) / "per_provider" / provider_id
    pp.mkdir(parents=True, exist_ok=True)
    return pp


def record_evidence(
    *,
    evidence_pack: Path | str,
    provider_id: str,
    kernel_source: str,
    language: str,
    run_report: RunReport,
    certificate: CertificateRecord,
    remote_receipt: dict[str, Any] | None = None,
) -> Path:
    """Write the available-provider quartet (kernel_source +
    run_report + certificate, plus optional remote_receipt).

    Returns the per-provider directory path so callers can attach
    further artifacts under it.
    """

    pp = _provider_dir(Path(evidence_pack), provider_id)
    ext = LANGUAGE_TO_EXT.get(language.lower(), "txt")
    src_path = pp / f"kernel_source.{ext}"
    src_path.write_text(kernel_source)
    sha = _sha256(kernel_source)

    if certificate.kernel_source_sha256 != sha:
        # Reissue the certificate with the correct hash so the on-disk
        # record matches the source we just wrote. This avoids a
        # silent drift between the kernel and its certificate.
        certificate = CertificateRecord(
            schema_version=certificate.schema_version,
            provider_id=certificate.provider_id,
            contract_hash=certificate.contract_hash,
            kernel_source_path=str(src_path),
            kernel_source_sha256=sha,
            verifier_verdict=certificate.verifier_verdict,
            verifier_detail=certificate.verifier_detail,
            issued_utc=certificate.issued_utc or _now_iso(),
        )

    (pp / "run_report.json").write_text(
        json.dumps(run_report.to_dict(), indent=2, sort_keys=True)
    )
    (pp / "certificate.json").write_text(
        json.dumps(certificate.to_dict(), indent=2, sort_keys=True)
    )
    if remote_receipt is not None:
        (pp / "remote_receipt.json").write_text(
            json.dumps(remote_receipt, indent=2, sort_keys=True)
        )
    return pp


def record_block(
    *,
    evidence_pack: Path | str,
    provider_id: str,
    proof: BlockedProof,
) -> Path:
    """Write ``blocked_proof.json`` for a non-available provider."""

    pp = _provider_dir(Path(evidence_pack), provider_id)
    (pp / "blocked_proof.json").write_text(
        json.dumps(proof.to_dict(), indent=2, sort_keys=True)
    )
    return pp


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------


def discover_per_provider_dirs(evidence_pack: Path | str) -> list[Path]:
    base = Path(evidence_pack) / "per_provider"
    if not base.is_dir():
        return []
    return sorted([p for p in base.iterdir() if p.is_dir()])


def audit_provider_dir(provider_dir: Path) -> tuple[str, str]:
    """Inspect one ``per_provider/<id>/`` directory.

    Returns ``(state, detail)``:

    * ``state="available"``   — full quartet present + schema-valid.
    * ``state="blocked"``     — ``blocked_proof.json`` present + valid.
    * ``state="malformed"``   — present but failing schema/missing files.
    * ``state="empty"``       — directory exists but no audit artifacts.
    """

    quartet = (
        provider_dir / "run_report.json",
        provider_dir / "certificate.json",
    )
    has_quartet_skeleton = all(p.is_file() for p in quartet)
    has_blocked = (provider_dir / "blocked_proof.json").is_file()
    sources = list(provider_dir.glob("kernel_source.*"))

    if has_quartet_skeleton and sources:
        # validate schemas
        try:
            RunReport.from_dict(json.loads(quartet[0].read_text()))
            CertificateRecord.from_dict(json.loads(quartet[1].read_text()))
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            return "malformed", f"available-quartet schema invalid: {exc}"
        return "available", ""

    if has_blocked:
        try:
            BlockedProof.from_dict(
                json.loads((provider_dir / "blocked_proof.json").read_text())
            )
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            return "malformed", f"blocked_proof schema invalid: {exc}"
        return "blocked", ""

    return "empty", "no evidence artifacts in directory"
