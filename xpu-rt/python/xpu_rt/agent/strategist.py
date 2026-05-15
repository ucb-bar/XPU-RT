"""Strategist — emits the session-level :class:`compgen.agent.plan.Plan` (P2.5).

The Strategist is *cheap*: it reads the graph dossier, the target
profile, the recipe-library index, and the perf budget; it emits a
:class:`Plan` with a per-region tactic + fallback ladder. The
Strategist never edits Recipe IR — it only picks tactics.

On rejection, :func:`compgen.agent.plan.replan_on_reject` decides
how to walk the ladder; the Strategist can be re-invoked when the
ladder is exhausted to widen the search (escalation).

Hard rules:

1. The Strategist's output is a :class:`Plan`. It does not produce
   edits, kernels, or verdicts.
2. Every region in the dossier ends up in the Plan with a non-empty
   ``fallback_ladder``. Empty ladders are forbidden — the region
   needs at least one ``naive_sync`` rung as a last resort.
3. The Strategist's *primary* path delegates to a deterministic
   default-tactic table; a live LLM-driven Strategist lands when the
   P3 primitives are wired to a real provider.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Final

from compgen.agent.plan import (
    GLOBAL_OBJECTIVES,
    Budget,
    Plan,
    PlanError,
    RegionPlan,
)

# Default ordered fallback ladder used when no region-specific
# recipe-library hint is available. ``naive_sync`` is the
# *last-resort* rung that any target supports.
DEFAULT_FALLBACK_LADDER: Final[tuple[str, ...]] = (
    "fuse",
    "tile_only",
    "naive_async",
    "naive_sync",
)

# Per-objective default tactic the Strategist prefers when no
# region-specific hint is available. The Tactician walks the
# ladder from this rung downward on rejection.
_OBJECTIVE_DEFAULT_TACTIC: Final[dict[str, str]] = {
    "minimize_p50_latency": "fuse",
    "minimize_p99_latency": "tile_only",
    "maximize_throughput": "naive_async",
    "minimize_memory": "tile_only",
    "minimize_compile_time": "naive_sync",
    "correctness_only": "naive_sync",
}


@dataclass(frozen=True)
class DossierRegion:
    """Minimal region summary the Strategist consumes.

    Mirrors the shape produced by the graph-analysis pipeline; the
    Strategist intentionally does not pull in the full
    ``graph_dossier`` to keep the dependency graph thin.
    """

    region_id: str
    op_family: str = "unknown"
    suggested_tactic: str | None = None
    recipe_library_match: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "region_id": self.region_id,
            "op_family": self.op_family,
            "suggested_tactic": self.suggested_tactic,
            "recipe_library_match": self.recipe_library_match,
        }


@dataclass(frozen=True)
class StrategistInput:
    """The closed contract the Strategist reads."""

    session_id: str
    global_objective: str
    budget: Budget
    regions: tuple[DossierRegion, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.global_objective not in GLOBAL_OBJECTIVES:
            raise PlanError(
                f"global_objective={self.global_objective!r} must be one of {GLOBAL_OBJECTIVES}"
            )


def plan_session(inputs: StrategistInput) -> Plan:
    """Emit the initial :class:`Plan` for a session.

    For each region:

    * ``suggested_tactic`` (from the dossier) wins if present;
    * otherwise the per-objective default tactic is used;
    * the fallback ladder is :data:`DEFAULT_FALLBACK_LADDER`, rooted
      so the picked tactic is the head;
    * regions with no rung options at all get ``("naive_sync",)``.

    The Strategist guarantees every region has a non-empty fallback
    ladder; :func:`compgen.agent.plan.replan_on_reject` walks it.
    """

    region_plans: list[RegionPlan] = []
    default_tactic = _OBJECTIVE_DEFAULT_TACTIC.get(inputs.global_objective, "naive_sync")

    for region in inputs.regions:
        head = region.suggested_tactic or default_tactic
        # Build a ladder rooted at ``head``: keep its position first,
        # drop entries that come before it, append the safety rung.
        ladder = [head]
        for rung in DEFAULT_FALLBACK_LADDER:
            if rung == head:
                continue
            ladder.append(rung)
        # Deduplicate while preserving order.
        seen: set[str] = set()
        deduped: list[str] = []
        for rung in ladder:
            if rung in seen:
                continue
            seen.add(rung)
            deduped.append(rung)
        if not deduped:
            deduped = ["naive_sync"]
        region_plans.append(
            RegionPlan(
                region_id=region.region_id,
                tactic=deduped[0],
                fallback_ladder=tuple(deduped),
            )
        )

    return Plan(
        session_id=inputs.session_id,
        plan_version=0,
        global_objective=inputs.global_objective,
        budget=inputs.budget,
        region_partition=tuple(region_plans),
    )


__all__ = [
    "DEFAULT_FALLBACK_LADDER",
    "DossierRegion",
    "StrategistInput",
    "plan_session",
]
