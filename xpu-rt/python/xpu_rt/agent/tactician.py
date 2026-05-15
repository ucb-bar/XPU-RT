"""Tactician — picks one Recipe-IR edit per region (P2.5).

The Tactician operates *inside* one region under a fixed Plan rung.
It reads:

* the region's :class:`compgen.agent.plan.RegionPlan` (current
  tactic + ladder);
* the precomputed candidate set;
* the per-candidate :class:`compgen.agent.cost_preview.CostPreview`.

It returns the candidate id of the next edit to apply, plus a typed
``next_action`` (``apply | escalate | exhausted``) the orchestrator
acts on.

Hard rules:

1. The Tactician never invents an edit — it picks from the input
   candidate list only (the headline P3.0 forbidden-action
   invariant).
2. The Tactician never reasons about correctness — it picks among
   *non-dominated, legal* candidates only; rejected candidates flow
   back through :func:`compgen.agent.plan.replan_on_reject`.
3. The Tactician's primary path delegates to a deterministic picker
   (lowest static cost among survivors); a live-LLM-driven path
   plugs in through P3.3's ``rank_candidates`` once the LLM provider
   is wired.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from compgen.agent.cost_preview import CostPreview, survivors
from compgen.agent.plan import RegionPlan

TACTICIAN_ACTIONS: Final[tuple[str, ...]] = ("apply", "escalate", "exhausted")


class TacticianError(ValueError):
    """A Tactician contract violation (e.g. region exhausted)."""


@dataclass(frozen=True)
class TacticianDecision:
    """Typed Tactician output."""

    next_action: str
    chosen_candidate_id: str | None
    reason: str

    def __post_init__(self) -> None:
        if self.next_action not in TACTICIAN_ACTIONS:
            raise TacticianError(
                f"next_action={self.next_action!r} must be one of {TACTICIAN_ACTIONS}"
            )

    def to_dict(self) -> dict:
        return {
            "next_action": self.next_action,
            "chosen_candidate_id": self.chosen_candidate_id,
            "reason": self.reason,
        }


def pick_edit(
    region_plan: RegionPlan,
    *,
    cost_previews: list[CostPreview],
) -> TacticianDecision:
    """Pick the next candidate to apply within ``region_plan``.

    Decision flow:

    1. If the region's rung is :data:`compgen.agent.plan.EXHAUSTED_TACTIC`,
       the Tactician returns ``next_action=exhausted`` — the
       orchestrator must escalate (Strategist re-plans or human
       takes over).
    2. Otherwise, filter to legal + non-dominated candidates
       (:func:`compgen.agent.cost_preview.survivors`).
    3. If there are no survivors, ``next_action=escalate`` — every
       candidate is dominated or blocked; the Strategist must
       widen the candidate set.
    4. Otherwise, pick the survivor with the lowest static cost.
       Tied costs break on lexicographic candidate_id (deterministic).
    """

    if region_plan.is_exhausted:
        return TacticianDecision(
            next_action="exhausted",
            chosen_candidate_id=None,
            reason=f"region {region_plan.region_id!r} fallback ladder exhausted",
        )

    surv = survivors(cost_previews)
    if not surv:
        return TacticianDecision(
            next_action="escalate",
            chosen_candidate_id=None,
            reason="no legal non-dominated candidates in input set",
        )

    surv.sort(key=lambda p: (p.delta_static, p.candidate_id))
    chosen = surv[0]
    return TacticianDecision(
        next_action="apply",
        chosen_candidate_id=chosen.candidate_id,
        reason=(
            f"picked lowest-cost survivor for tactic={region_plan.tactic!r}; "
            f"delta_static={chosen.delta_static}"
        ),
    )


__all__ = [
    "TACTICIAN_ACTIONS",
    "TacticianDecision",
    "TacticianError",
    "pick_edit",
]
