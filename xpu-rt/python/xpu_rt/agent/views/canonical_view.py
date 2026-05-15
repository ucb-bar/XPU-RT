"""canonical_view — bounded session summary for LLM context (P2.1).

The view collapses a full ``DriverCheckpoint``-shaped state into a
≤1 KB-per-region summary. The LLM never has to see the entire
session — it sees:

* the current Plan rung the Strategist is on (or ``"unplanned"``);
* per region: the open decision sites' ids, the last verdict (if
  any), and the current tactic from the Plan's fallback ladder.

If the canonical view would exceed the byte budget, it raises
:class:`CanonicalViewBudgetError` rather than silently truncating —
the caller is responsible for narrowing the regions or invoking
:func:`compgen.agent.views.focus_chunk.focus_chunk` per region
instead.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Final

CANONICAL_VIEW_BYTE_BUDGET: Final[int] = 4096  # 1 KB × 4 regions worth


class CanonicalViewBudgetError(ValueError):
    """The summary exceeded :data:`CANONICAL_VIEW_BYTE_BUDGET` bytes."""


@dataclass(frozen=True)
class CanonicalRegionRow:
    """One region's slice of the canonical view."""

    region_id: str
    current_tactic: str
    open_decision_site_ids: tuple[str, ...]
    last_verdict: str | None
    last_reason: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "region_id": self.region_id,
            "current_tactic": self.current_tactic,
            "open_decision_site_ids": list(self.open_decision_site_ids),
            "last_verdict": self.last_verdict,
            "last_reason": self.last_reason,
        }


@dataclass(frozen=True)
class CanonicalView:
    """Bounded session summary."""

    session_id: str
    plan_version: int
    global_objective: str
    rows: tuple[CanonicalRegionRow, ...] = field(default_factory=tuple)
    byte_size: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "plan_version": self.plan_version,
            "global_objective": self.global_objective,
            "rows": [r.to_dict() for r in self.rows],
        }

    def to_dict_with_metadata(self) -> dict[str, Any]:
        """Like ``to_dict`` but also reports the cached byte size.

        Kept separate so the canonical-JSON serialisation used to
        compute ``byte_size`` does not itself depend on ``byte_size``
        (a chicken-and-egg loop).
        """

        return {**self.to_dict(), "byte_size": self.byte_size}


def canonical_view(
    session_state: dict[str, Any], *, max_bytes: int = CANONICAL_VIEW_BYTE_BUDGET
) -> CanonicalView:
    """Build the canonical view from a driver-checkpoint-shaped dict.

    Expected ``session_state`` shape (a strict subset of the
    :class:`compgen.agent.llm_driver.DriverCheckpoint` to_dict shape):

    ::

        {
          "session_id": "ses_a3",
          "plan": {"plan_version": 7, "global_objective": "minimize_p50_latency",
                   "region_partition": [{"region_id": "017", "tactic": "fuse",
                                          "fallback_ladder": ["fuse", "tile_only"]}]},
          "regions": [{"region_id": "017", "open_decision_sites": ["site_a"],
                       "last_verdict": "rejected", "last_reason": "scratchpad_overflow"}]
        }

    Missing fields are tolerated: ``plan`` defaults to an unplanned
    session; ``regions`` defaults to empty.
    """

    session_id = str(session_state.get("session_id", ""))
    plan = session_state.get("plan") or {}
    plan_version = int(plan.get("plan_version") or 0)
    global_objective = str(plan.get("global_objective") or "unplanned")

    tactic_by_region: dict[str, str] = {}
    for entry in plan.get("region_partition", []) or []:
        if not isinstance(entry, dict):
            continue
        tactic_by_region[str(entry.get("region_id"))] = str(entry.get("tactic", "unplanned"))

    rows: list[CanonicalRegionRow] = []
    for region in session_state.get("regions", []) or []:
        if not isinstance(region, dict):
            continue
        rid = str(region.get("region_id", ""))
        rows.append(
            CanonicalRegionRow(
                region_id=rid,
                current_tactic=tactic_by_region.get(rid, "unplanned"),
                open_decision_site_ids=tuple(
                    str(s) for s in region.get("open_decision_sites", []) or []
                ),
                last_verdict=(
                    str(region["last_verdict"]) if region.get("last_verdict") else None
                ),
                last_reason=(
                    str(region["last_reason"]) if region.get("last_reason") else None
                ),
            )
        )

    view = CanonicalView(
        session_id=session_id,
        plan_version=plan_version,
        global_objective=global_objective,
        rows=tuple(rows),
        byte_size=0,  # placeholder; will be recomputed below
    )
    serialized = json.dumps(view.to_dict(), sort_keys=True, separators=(",", ":"))
    byte_size = len(serialized.encode("utf-8"))
    if byte_size > max_bytes:
        raise CanonicalViewBudgetError(
            f"canonical view would be {byte_size} bytes (cap={max_bytes}); "
            f"narrow the region set or use focus_chunk per region"
        )
    return CanonicalView(
        session_id=view.session_id,
        plan_version=view.plan_version,
        global_objective=view.global_objective,
        rows=view.rows,
        byte_size=byte_size,
    )


__all__ = [
    "CANONICAL_VIEW_BYTE_BUDGET",
    "CanonicalRegionRow",
    "CanonicalView",
    "CanonicalViewBudgetError",
    "canonical_view",
]
