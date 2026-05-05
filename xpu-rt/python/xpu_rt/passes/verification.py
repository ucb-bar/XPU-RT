"""Verification certificate gate (M-33.2).

Each pass card declares a list of ``verification`` rungs that must be
discharged before downstream consumers can read the pass's outputs.
Today CompGen emits two real verification reports:

- ``03_recipe_planning/post_lowering/post_lowering_verification_report.json``
  (structural / contract verification)
- ``03_recipe_planning/differential_verification/differential_verification_report.json``
  (differential numerical verification)

M-33 wraps each into a :class:`VerificationCertificate` JSON sidecar
co-located with the report. The certificate carries:

- ``rung``           — ``structural`` / ``differential`` / ``formal``
- ``status``         — ``pass`` / ``fail`` / ``skipped``
- ``artifact_path``  — relative path to the underlying artifact this
                       certificate vouches for (e.g. transformed
                       payload MLIR)
- ``artifact_hash``  — sha256[:16] of that artifact at certificate-emit
                       time. A later upstream change makes the
                       certificate stale (hash mismatch).
- ``report_path``    — relative path to the underlying full report
- ``generated_at_utc``

Two enforcement entry points:

- :func:`assert_required_rungs_discharged(card, run_dir)` — for the
  validator side. Raises :class:`VerificationGateMissing` when a rung
  has no certificate, :class:`VerificationGateFailed` when a
  certificate exists but reports a non-pass status, and treats
  ``skipped`` as a pass when (and only when) the rung is allowed to
  be skipped (e.g. ``formal`` today).
- :func:`emit_certificate_from_post_lowering_report` /
  :func:`emit_certificate_from_differential_report` — wrap-and-write
  helpers called from ``run.py`` after each emitter completes.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from compgen.audit.errors import (
    VerificationGateFailed,
    VerificationGateMissing,
)
from compgen.passes.cards import PassCard

CERT_SCHEMA_VERSION = "verification_certificate_v1"

ALLOWED_RUNGS: tuple[str, ...] = ("structural", "differential", "formal")
ALLOWED_STATUSES: tuple[str, ...] = ("pass", "fail", "skipped")

# Rungs that may legitimately be ``skipped`` without failing the gate.
# ``formal`` has no live emitter today — pass cards that declare it get
# a skipped-with-typed-reason certificate. M-33's honest-residual list
# names this as the residual to close when semantic-IR formal
# verification ships.
SKIPPABLE_RUNGS: frozenset[str] = frozenset({"formal"})


def _utc_now() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256_short(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


def _hash_path(path: Path) -> str:
    if not path.exists():
        return ""
    if path.is_file():
        return _sha256_short(path.read_bytes())
    h = hashlib.sha256()
    for sub in sorted(path.rglob("*")):
        if not sub.is_file():
            continue
        rel = sub.relative_to(path).as_posix()
        h.update(rel.encode("utf-8"))
        h.update(b"\x00")
        h.update(sub.read_bytes())
        h.update(b"\x01")
    return h.hexdigest()[:16]


@dataclass(frozen=True)
class VerificationCertificate:
    """One rung-level verification certificate."""

    schema_version: str
    rung: str
    status: str  # pass | fail | skipped
    pass_id: str  # the pass this certificate vouches for ("" if global)
    artifact_path: str  # relative to run_dir
    artifact_hash: str  # sha256[:16] at emit time
    report_path: str  # relative path to underlying report
    generated_at_utc: str
    skipped_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "rung": self.rung,
            "status": self.status,
            "pass_id": self.pass_id,
            "artifact_path": self.artifact_path,
            "artifact_hash": self.artifact_hash,
            "report_path": self.report_path,
            "generated_at_utc": self.generated_at_utc,
            "skipped_reason": self.skipped_reason,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VerificationCertificate:
        return cls(
            schema_version=str(data.get("schema_version", CERT_SCHEMA_VERSION)),
            rung=str(data["rung"]),
            status=str(data["status"]),
            pass_id=str(data.get("pass_id", "")),
            artifact_path=str(data.get("artifact_path", "")),
            artifact_hash=str(data.get("artifact_hash", "")),
            report_path=str(data.get("report_path", "")),
            generated_at_utc=str(data.get("generated_at_utc", "")),
            skipped_reason=str(data.get("skipped_reason", "")),
        )


# --------------------------------------------------------------------------- #
# Locations
# --------------------------------------------------------------------------- #


def _certificate_path_for(run_dir: Path, rung: str) -> Path:
    """Canonical certificate location per rung.

    M-33 colocates each certificate with the report it wraps:
      structural   → 03_recipe_planning/post_lowering/verification_certificate.json
      differential → 03_recipe_planning/differential_verification/verification_certificate.json
      formal       → 03_recipe_planning/formal_verification/verification_certificate.json
    """
    base = run_dir / "03_recipe_planning"
    if rung == "structural":
        return base / "post_lowering" / "verification_certificate.json"
    if rung == "differential":
        return base / "differential_verification" / "verification_certificate.json"
    if rung == "formal":
        return base / "formal_verification" / "verification_certificate.json"
    raise ValueError(f"unknown verification rung: {rung!r}")


def load_certificate(
    run_dir: Path, rung: str,
) -> VerificationCertificate | None:
    """Return the certificate for ``rung`` or None if not on disk."""
    path = _certificate_path_for(run_dir, rung)
    if not path.exists():
        return None
    return VerificationCertificate.from_dict(json.loads(path.read_text()))


def write_certificate(
    cert: VerificationCertificate, *, run_dir: Path,
) -> Path:
    out = _certificate_path_for(run_dir, cert.rung)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(cert.to_dict(), indent=2, sort_keys=True) + "\n")
    return out


# --------------------------------------------------------------------------- #
# Emitters: wrap existing reports
# --------------------------------------------------------------------------- #


def emit_certificate_from_post_lowering_report(
    *,
    run_dir: Path,
    pass_id: str = "",
) -> VerificationCertificate | None:
    """Read post_lowering_verification_report.json and emit a certificate.

    Returns None if the underlying report is not present (the upstream
    stage never ran). The caller decides whether absence is acceptable
    (often it is — the report is only emitted when a real transform
    landed under post_lowering).
    """
    report_path = (
        run_dir
        / "03_recipe_planning"
        / "post_lowering"
        / "post_lowering_verification_report.json"
    )
    if not report_path.exists():
        return None
    report = json.loads(report_path.read_text())
    transformed_payload = (
        run_dir
        / "03_recipe_planning"
        / "post_lowering"
        / "transformed_payload.mlir"
    )
    cert = VerificationCertificate(
        schema_version=CERT_SCHEMA_VERSION,
        rung="structural",
        status=str(report.get("status", "fail")),
        pass_id=pass_id,
        artifact_path=transformed_payload.relative_to(run_dir).as_posix(),
        artifact_hash=_hash_path(transformed_payload),
        report_path=report_path.relative_to(run_dir).as_posix(),
        generated_at_utc=_utc_now(),
    )
    write_certificate(cert, run_dir=run_dir)
    return cert


def emit_certificate_from_differential_report(
    *,
    run_dir: Path,
    pass_id: str = "",
) -> VerificationCertificate | None:
    """Read differential_verification_report.json and emit a certificate."""
    report_path = (
        run_dir
        / "03_recipe_planning"
        / "differential_verification"
        / "differential_verification_report.json"
    )
    if not report_path.exists():
        return None
    report = json.loads(report_path.read_text())
    # The artifact this certificate vouches for is the same transformed
    # payload as structural — they verify the same payload from
    # different angles.
    transformed_payload = (
        run_dir
        / "03_recipe_planning"
        / "post_lowering"
        / "transformed_payload.mlir"
    )
    cert = VerificationCertificate(
        schema_version=CERT_SCHEMA_VERSION,
        rung="differential",
        status=str(report.get("status", "fail")),
        pass_id=pass_id,
        artifact_path=transformed_payload.relative_to(run_dir).as_posix()
        if transformed_payload.exists() else "",
        artifact_hash=_hash_path(transformed_payload)
        if transformed_payload.exists() else "",
        report_path=report_path.relative_to(run_dir).as_posix(),
        generated_at_utc=_utc_now(),
    )
    write_certificate(cert, run_dir=run_dir)
    return cert


def emit_skipped_formal_certificate(
    *, run_dir: Path, pass_id: str = "",
    reason: str = "formal verification has no live emitter (M-33 honest residual)",
) -> VerificationCertificate:
    """Emit a typed ``skipped`` certificate for the formal rung."""
    cert = VerificationCertificate(
        schema_version=CERT_SCHEMA_VERSION,
        rung="formal",
        status="skipped",
        pass_id=pass_id,
        artifact_path="",
        artifact_hash="",
        report_path="",
        generated_at_utc=_utc_now(),
        skipped_reason=reason,
    )
    write_certificate(cert, run_dir=run_dir)
    return cert


# --------------------------------------------------------------------------- #
# Enforcement
# --------------------------------------------------------------------------- #


def assert_required_rungs_discharged(
    card: PassCard,
    run_dir: Path,
) -> None:
    """Verify every rung in ``card.verification`` has a passing certificate.

    Raises :class:`VerificationGateMissing` if any required rung has no
    certificate on disk. Raises :class:`VerificationGateFailed` if any
    certificate reports ``status == "fail"``. ``skipped`` is accepted
    only for rungs in :data:`SKIPPABLE_RUNGS` (today: ``formal``).
    """
    missing: list[str] = []
    failed: list[tuple[str, str]] = []  # (rung, reason)
    for rung in card.verification:
        if rung not in ALLOWED_RUNGS:
            # The pass-card schema validator should already have caught
            # this; treat it as missing here.
            missing.append(rung)
            continue
        cert = load_certificate(run_dir, rung)
        if cert is None:
            missing.append(rung)
            continue
        if cert.status == "pass":
            continue
        if cert.status == "skipped" and rung in SKIPPABLE_RUNGS:
            continue
        if cert.status == "fail":
            failed.append((rung, "status=fail"))
        elif cert.status == "skipped":
            failed.append((rung, f"skipped (reason: {cert.skipped_reason})"))
        else:
            failed.append((rung, f"unknown status {cert.status!r}"))

    if missing:
        raise VerificationGateMissing(
            f"pass {card.pass_id!r} requires rungs {list(card.verification)}; "
            f"missing certificates: {missing}"
        )
    if failed:
        details = ", ".join(f"{r}: {why}" for r, why in failed)
        raise VerificationGateFailed(
            f"pass {card.pass_id!r}: verification failed for rungs: {details}"
        )


def assert_certificate_artifact_fresh(
    cert: VerificationCertificate,
    run_dir: Path,
) -> None:
    """Verify the certificate's artifact_hash still matches on-disk content.

    Used by consumers to detect "the report was emitted, but the
    artifact has since been mutated" — the certificate is stale.
    """
    if not cert.artifact_path:
        return  # Skipped certificates have no artifact to track.
    abs_path = run_dir / cert.artifact_path
    actual = _hash_path(abs_path)
    if actual != cert.artifact_hash:
        raise VerificationGateFailed(
            f"certificate for rung={cert.rung!r} pass={cert.pass_id!r} "
            f"references artifact {cert.artifact_path} with "
            f"hash={cert.artifact_hash!r}, but on-disk hash is "
            f"{actual!r}: artifact has been mutated since cert emission"
        )
