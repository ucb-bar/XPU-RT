"""Typed errors for the promotion pipeline.

Phase 3 production-hardening: promotion must never silently accept an
unverified bundle. Callers that hit
:class:`PromotionBlockedError` get a structured reason so they can
either fix the bundle or opt out with ``force=True`` (explicit).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class PromotionBlockReason:
    """One reason a bundle can't be promoted.

    ``code`` is one of:

    - ``"missing_verification_report"`` — bundle has no
      ``verification_report.json``.
    - ``"verification_report_unreadable"`` — file exists but can't be
      parsed.
    - ``"verification_failed"`` — ladder ran, at least one level did
      not pass.
    - ``"level_skipped"`` — ladder passed but a required level was
      SKIPPED; production promotion requires all levels exercised.
    - ``"level_failed_strict"`` — details dict reports a FAIL on a
      level that claims to pass at the top level.
    """

    code: str
    detail: str
    path: str | None = None


class PromotionBlockedError(RuntimeError):
    """Raised when :class:`RecipePromoter.promote` refuses a bundle.

    Carries the list of :class:`PromotionBlockReason` entries plus the
    bundle path so callers can diagnose without re-running the
    verification ladder.
    """

    def __init__(
        self,
        reasons: list[PromotionBlockReason],
        bundle_root: Path | None = None,
    ) -> None:
        self.reasons = list(reasons)
        self.bundle_root = bundle_root
        detail = "; ".join(f"[{r.code}] {r.detail}" for r in self.reasons)
        super().__init__(f"promotion blocked ({len(self.reasons)} reason(s)): {detail}")


@dataclass(frozen=True)
class VerificationGateResult:
    """Inspection outcome for a bundle's verification report."""

    passed: bool
    reasons: list[PromotionBlockReason] = field(default_factory=list)
    report: dict | None = None


__all__ = [
    "PromotionBlockReason",
    "PromotionBlockedError",
    "VerificationGateResult",
]
