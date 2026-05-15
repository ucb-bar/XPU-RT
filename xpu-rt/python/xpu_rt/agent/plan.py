"""Plan dataclass + replan-on-reject (P2.4).

The Strategist owns a :class:`Plan` whose region-level rungs are the
fallback ladders the Tactician walks. On rejection, the typed
:class:`compgen.agent.counterexample.Counterexample.rejection_class`
decides what happens:

* ``tactic_fatal`` — the Strategist drops the current rung from the
  region's fallback ladder. Replan emits a new :class:`Plan` whose
  region's ``tactic`` is the next rung.
* ``tactic_recoverable`` — no replan. The Tactician retries on the
  same rung with the remediation hint.
* ``surprising`` — the Strategist escalates (currently: drops the
  rung *and* marks the region with ``escalated=True`` so a human / a
  larger model can take over).

This module is pure-function and standalone — wire-in to the
LLMDriver session loop is a follow-up.

Hard rules:

* :class:`Plan` is frozen. A "replan" produces a *new* Plan with
  ``plan_version`` incremented; the old plan is preserved for audit.
* ``global_objective`` is a closed enum: any free-form objective is
  rejected at construction.
* A region whose fallback_ladder runs out (the last rung also gets
  rejected) is marked with the dedicated rung ``"_exhausted"``;
  callers MUST check :attr:`RegionPlan.is_exhausted` before issuing
  another Tactician step.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Final

GLOBAL_OBJECTIVES: Final[tuple[str, ...]] = (
    "minimize_p50_latency",
    "minimize_p99_latency",
    "maximize_throughput",
    "minimize_memory",
    "minimize_compile_time",
    "correctness_only",
)

EXHAUSTED_TACTIC: Final[str] = "_exhausted"


class PlanError(ValueError):
    """The Plan / replan layer rejected a request."""


@dataclass(frozen=True)
class Budget:
    """Per-session compile + LLM budget."""

    compile_seconds: float
    llm_dollars: float

    def __post_init__(self) -> None:
        if self.compile_seconds < 0:
            raise PlanError("compile_seconds must be >= 0")
        if self.llm_dollars < 0:
            raise PlanError("llm_dollars must be >= 0")

    def to_dict(self) -> dict[str, float]:
        return {"compile_seconds": self.compile_seconds, "llm_dollars": self.llm_dollars}


@dataclass(frozen=True)
class RegionPlan:
    """One region's slice of the Plan."""

    region_id: str
    tactic: str
    fallback_ladder: tuple[str, ...]
    escalated: bool = False

    def __post_init__(self) -> None:
        if not self.region_id:
            raise PlanError("region_id must be a non-empty string")
        if not self.tactic:
            raise PlanError(
                f"region {self.region_id!r}: tactic must be non-empty "
                f"(use {EXHAUSTED_TACTIC!r} for exhausted ladders)"
            )

    @property
    def is_exhausted(self) -> bool:
        return self.tactic == EXHAUSTED_TACTIC

    def to_dict(self) -> dict[str, Any]:
        return {
            "region_id": self.region_id,
            "tactic": self.tactic,
            "fallback_ladder": list(self.fallback_ladder),
            "escalated": self.escalated,
        }


@dataclass(frozen=True)
class Plan:
    """Session-level Plan."""

    session_id: str
    plan_version: int
    global_objective: str
    budget: Budget
    region_partition: tuple[RegionPlan, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.global_objective not in GLOBAL_OBJECTIVES:
            raise PlanError(
                f"global_objective={self.global_objective!r} must be one of {GLOBAL_OBJECTIVES}"
            )
        if self.plan_version < 0:
            raise PlanError("plan_version must be >= 0")
        ids = [r.region_id for r in self.region_partition]
        if len(set(ids)) != len(ids):
            raise PlanError("region_partition contains duplicate region_id values")

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "plan_version": self.plan_version,
            "global_objective": self.global_objective,
            "budget": self.budget.to_dict(),
            "region_partition": [r.to_dict() for r in self.region_partition],
        }

    def get_region(self, region_id: str) -> RegionPlan:
        for r in self.region_partition:
            if r.region_id == region_id:
                return r
        raise PlanError(f"region {region_id!r} not present in plan")


def _next_rung(ladder: tuple[str, ...], current: str) -> str:
    """Pick the rung that follows ``current`` in ``ladder``.

    Returns :data:`EXHAUSTED_TACTIC` when ``current`` is at or past
    the end. The fallback ladder is *ordered* — index in the tuple
    is the priority.
    """

    if current not in ladder:
        return EXHAUSTED_TACTIC
    idx = ladder.index(current)
    if idx + 1 >= len(ladder):
        return EXHAUSTED_TACTIC
    return ladder[idx + 1]


def replan_on_reject(
    plan: Plan,
    *,
    region_id: str,
    rejection_class: str,
) -> Plan:
    """Compute the next Plan in response to a rejection on a region.

    Behaviour by :class:`compgen.agent.counterexample.Counterexample.rejection_class`:

    * ``tactic_fatal`` — drop the current rung, advance to the next.
    * ``tactic_recoverable`` — return the plan unchanged (Tactician
      retries on the same rung with the remediation hint).
    * ``surprising`` — drop the rung AND mark the region as
      ``escalated=True``.

    Raises :class:`PlanError` on an unknown rejection class or unknown
    region id.
    """

    from compgen.agent.counterexample import REJECTION_CLASSES

    if rejection_class not in REJECTION_CLASSES:
        raise PlanError(
            f"rejection_class={rejection_class!r} must be one of {REJECTION_CLASSES}"
        )
    target = plan.get_region(region_id)
    if rejection_class == "tactic_recoverable":
        # No structural change; same plan reused.
        return plan

    new_tactic = _next_rung(target.fallback_ladder, target.tactic)
    escalated = target.escalated or rejection_class == "surprising"
    new_region = RegionPlan(
        region_id=target.region_id,
        tactic=new_tactic,
        fallback_ladder=target.fallback_ladder,
        escalated=escalated,
    )
    new_partition = tuple(
        new_region if r.region_id == region_id else r for r in plan.region_partition
    )
    return replace(plan, plan_version=plan.plan_version + 1, region_partition=new_partition)


__all__ = [
    "EXHAUSTED_TACTIC",
    "GLOBAL_OBJECTIVES",
    "Budget",
    "Plan",
    "PlanError",
    "RegionPlan",
    "replan_on_reject",
]
