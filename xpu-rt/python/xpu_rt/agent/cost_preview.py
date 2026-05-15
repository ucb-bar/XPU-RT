"""Pre-commit cost preview (P2.2).

Converts the agent loop from **edit → compile → bench → throw away**
to **predict → edit → verify → bench-the-survivors**: before any
Recipe-IR edit is applied, the agent gets a typed
:class:`CostPreview` saying whether the edit is *legal*, what its
predicted *delta_static* against the analytical roofline is,
and which other candidates *dominate* it.

The preview is *deterministic* — same inputs → same outputs. The
LLM never reads cost as a single scalar; it reads the typed payload
and chooses among non-dominated survivors.

Wire-in to :mod:`compgen.graph_compilation.agent_decision`
(``propose_recipe_edit``) is a follow-up; this module is testable in
isolation against a candidate-list shape.

Hard rules:

* ``delta_surrogate`` starts as ``None`` (honestly absent) until the
  P2.6 online surrogate trains and exposes its delta. Pretending the
  surrogate is available before P2.6 is forbidden.
* ``legality`` is a closed enum (``ok | blocked | unknown``); a
  candidate with ``legality != "ok"`` is never used to dominate
  another candidate.
* ``dominated_by`` is filled in a single pass over the candidate
  list. The dominance relation is *strict*: candidate ``b`` dominates
  candidate ``a`` iff ``b.legality == "ok"`` AND
  ``b.delta_static < a.delta_static`` (lower cost is better) AND
  ``b.legality`` is at least as strong as ``a.legality``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Final

LEGALITY_VALUES: Final[tuple[str, ...]] = ("ok", "blocked", "unknown")


class CostPreviewError(ValueError):
    """Bad input passed to :func:`compute_cost_previews`."""


@dataclass(frozen=True)
class CandidateInput:
    """Minimal candidate shape the preview consumes.

    The full candidate dataclass
    :class:`compgen.agent.suggest._candidate.ProposalCandidate` carries
    more (rationale, members, …); the preview only needs id + cost
    facts + legality.
    """

    candidate_id: str
    delta_static: float
    legality: str = "ok"
    detail: str = ""

    def __post_init__(self) -> None:
        if not self.candidate_id:
            raise CostPreviewError("candidate_id must be a non-empty string")
        if self.legality not in LEGALITY_VALUES:
            raise CostPreviewError(
                f"candidate {self.candidate_id!r} legality={self.legality!r} "
                f"must be one of {LEGALITY_VALUES}"
            )


@dataclass(frozen=True)
class CostPreview:
    """Typed preview per candidate.

    * ``delta_static`` — change in static cost score from the
      analytical roofline. Lower is better; sign is preserved so the
      LLM can spot regressions.
    * ``delta_surrogate`` — learned-model prediction. ``None`` until
      P2.6 is wired.
    * ``confidence`` — in [0, 1], reflecting how similar this edit is
      to historical bench data. ``None`` when no surrogate exists.
    * ``legality`` — closed-enum legality of the edit.
    * ``dominated_by`` — list of candidate ids that strictly dominate
      this one (lower static cost AND legal).
    """

    candidate_id: str
    delta_static: float
    delta_surrogate: float | None
    confidence: float | None
    legality: str
    dominated_by: tuple[str, ...] = field(default_factory=tuple)
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "delta_static": self.delta_static,
            "delta_surrogate": self.delta_surrogate,
            "confidence": self.confidence,
            "legality": self.legality,
            "dominated_by": list(self.dominated_by),
            "detail": self.detail,
        }

    @property
    def is_survivor(self) -> bool:
        """A *survivor* is a legal, non-dominated candidate worth benching."""

        return self.legality == "ok" and not self.dominated_by


def compute_cost_previews(
    candidates: list[CandidateInput],
    *,
    surrogate_deltas: dict[str, float] | None = None,
    confidence_by_id: dict[str, float] | None = None,
) -> list[CostPreview]:
    """Build the full list of cost previews in one deterministic pass.

    Parameters
    ----------
    candidates
        Closed candidate set — the LLM cannot add to it.
    surrogate_deltas
        Optional mapping ``candidate_id -> delta_surrogate`` from the
        P2.6 online surrogate. Passing ``None`` (the default) is the
        pre-P2.6 honest baseline: ``CostPreview.delta_surrogate`` is
        ``None`` for every row.
    confidence_by_id
        Optional mapping ``candidate_id -> confidence in [0, 1]``.
        Same lifecycle as ``surrogate_deltas``.
    """

    if not isinstance(candidates, list):
        raise CostPreviewError("candidates must be a list")
    ids = [c.candidate_id for c in candidates]
    if len(set(ids)) != len(ids):
        raise CostPreviewError("candidate_id must be unique within the candidate set")

    # Index legal candidates by static cost so dominance is O(N).
    legal_sorted = sorted(
        (c for c in candidates if c.legality == "ok"),
        key=lambda c: c.delta_static,
    )

    surrogate_deltas = surrogate_deltas or {}
    confidence_by_id = confidence_by_id or {}

    out: list[CostPreview] = []
    for cand in candidates:
        dominators: list[str] = []
        if cand.legality == "ok":
            # Any LEGAL candidate with strictly lower static cost
            # dominates this one. Tied costs do NOT dominate
            # (we never silently prefer one tied alternative).
            for other in legal_sorted:
                if other.candidate_id == cand.candidate_id:
                    break
                if other.delta_static < cand.delta_static:
                    dominators.append(other.candidate_id)
        # blocked/unknown candidates: never dominated (they need to
        # surface their typed status to the LLM intact); never
        # dominator either (already excluded from legal_sorted).
        out.append(
            CostPreview(
                candidate_id=cand.candidate_id,
                delta_static=cand.delta_static,
                delta_surrogate=surrogate_deltas.get(cand.candidate_id),
                confidence=confidence_by_id.get(cand.candidate_id),
                legality=cand.legality,
                dominated_by=tuple(dominators),
                detail=cand.detail,
            )
        )
    return out


def survivors(previews: list[CostPreview]) -> list[CostPreview]:
    """Return the previews worth benching — legal AND non-dominated."""

    return [p for p in previews if p.is_survivor]


__all__ = [
    "LEGALITY_VALUES",
    "CandidateInput",
    "CostPreview",
    "CostPreviewError",
    "compute_cost_previews",
    "survivors",
]
