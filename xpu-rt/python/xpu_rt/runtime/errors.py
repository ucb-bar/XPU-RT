"""Typed runtime errors surfaced by the compile + bundle pipeline.

Centralising errors here keeps the raise sites honest: every failure
that a caller needs to distinguish has its own class, and every class
carries the structured payload the caller needs to recover or diagnose.

Why a separate module: ``api.py`` and ``bundle_emit.py`` and the
providers all need to catch / raise these, and importing across those
modules without a circular-import trap requires a neutral home.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ArtifactStatus:
    """Status of a single bundle artifact after emission.

    ``status`` is one of:
      - ``"ok"``      — artifact written successfully; ``path`` is set.
      - ``"failed"``  — emission raised; ``error`` carries the reason.
      - ``"skipped"`` — upstream data not available (e.g. no ``analysis``
                        passed, so ``gap_analysis.json`` has nothing to
                        emit). ``reason`` explains why.

    ``skipped`` is not failure — it's an honest "this contract slot is
    empty because the compile didn't produce the inputs". ``failed``
    means we tried and blew up; that gets aggregated into
    :class:`BundleEmissionError`.
    """

    name: str
    status: str  # "ok" | "failed" | "skipped"
    path: str | None = None
    error: str | None = None
    reason: str | None = None

    def as_dict(self) -> dict[str, str | None]:
        return {
            "status": self.status,
            "path": self.path,
            "error": self.error,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class BundleEmissionReport:
    """Aggregate result of :func:`emit_extended_artefacts`.

    Holds one :class:`ArtifactStatus` per known artifact slot, plus the
    bundle directory for downstream callers that want to pin a path.
    Keeping the report explicit lets callers:

    - Serialise it into ``manifest.json::extended_artifacts`` so users
      see which contract slots actually emitted.
    - Publish per-artifact trace events for observability.
    - Detect at least one ``failed`` artifact and raise
      :class:`BundleEmissionError` before returning a bundle that
      silently lost data.
    """

    bundle_dir: Path
    statuses: tuple[ArtifactStatus, ...] = field(default_factory=tuple)

    @property
    def failed(self) -> tuple[ArtifactStatus, ...]:
        return tuple(s for s in self.statuses if s.status == "failed")

    @property
    def ok(self) -> tuple[ArtifactStatus, ...]:
        return tuple(s for s in self.statuses if s.status == "ok")

    @property
    def skipped(self) -> tuple[ArtifactStatus, ...]:
        return tuple(s for s in self.statuses if s.status == "skipped")

    def to_manifest_block(self) -> dict[str, dict[str, str | None]]:
        return {s.name: s.as_dict() for s in self.statuses}


class BundleEmissionError(RuntimeError):
    """One or more extended artifacts failed to emit.

    Carries the full :class:`BundleEmissionReport` so callers can see
    which artifacts were ok, which were skipped (acceptable), and which
    failed (not acceptable). The string form lists failed artifacts +
    their errors for human inspection.
    """

    def __init__(self, report: BundleEmissionReport) -> None:
        self.report = report
        parts = [f"{s.name}: {s.error}" for s in report.failed]
        super().__init__(
            f"bundle emission failed for {len(report.failed)} artifact(s) in {report.bundle_dir}: " + "; ".join(parts)
        )


class AdapterUnavailableError(RuntimeError):
    """A requested runtime adapter is not available on this build.

    Replaces the older "silent stub" pattern where an adapter's
    ``dispatch`` / ``replay`` raised a bare ``NotImplementedError``.
    Carries the adapter name + why so callers can log or fall back.
    """

    def __init__(self, adapter_name: str, reason: str) -> None:
        self.adapter_name = adapter_name
        self.reason = reason
        super().__init__(f"runtime adapter {adapter_name!r} unavailable: {reason}")


class UnsupportedTopologyError(NotImplementedError):
    """Model topology is out of scope for a topology-specialized lowerer.

    Raised by path-specific lowerers (``runtime.embedded.cnn_lowering``
    etc.) when the caller's model doesn't match the shape fingerprint
    the lowerer was designed for. This is a *scope boundary*, not a
    bug — generic models fall back to the FX / payload-IR path. The
    name signals intent so callers don't conflate it with "we forgot
    to implement this".
    """


class SymbolicShapeUnsupportedError(NotImplementedError):
    """Runtime path hit symbolic / data-dependent shapes it can't handle yet.

    Raised by ``compgen.ir.event.lower`` when an Event Tensor / task
    grid has ``-1`` dims or the graph contains ``UpdateOp`` /
    ``TriggerOp`` / ``MaterializeViewOp`` — paper Fig. 4 / Fig. 5
    extensions from Jin et al., MLSys '26. Marked as a ROADMAP
    boundary in ``01_v1_honest_state.md``; static (compile-time
    shape) graphs lower cleanly. The error message points to the
    paper section so downstream code can triage without re-reading
    the lowerer.
    """


__all__ = [
    "AdapterUnavailableError",
    "ArtifactStatus",
    "BundleEmissionError",
    "BundleEmissionReport",
    "SymbolicShapeUnsupportedError",
    "UnsupportedTopologyError",
]
