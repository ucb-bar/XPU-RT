"""diff_since — delta between two driver checkpoints (P2.1).

Surfaces what changed in a session between two snapshots so the
Strategist can replan reactively instead of reasoning from scratch.

Diff entries are typed (``added`` / ``changed`` / ``removed``) and
keyed by an opaque pointer string ``region.<id>.<field>`` so the LLM
can navigate the diff without seeing every untouched field.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final

DIFF_KINDS: Final[tuple[str, ...]] = ("added", "changed", "removed")


@dataclass(frozen=True)
class DiffEntry:
    """One typed change between two snapshots."""

    pointer: str
    kind: str
    before: Any | None
    after: Any | None

    def __post_init__(self) -> None:
        if self.kind not in DIFF_KINDS:
            raise ValueError(f"unknown diff kind {self.kind!r}; must be in {DIFF_KINDS}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "pointer": self.pointer,
            "kind": self.kind,
            "before": self.before,
            "after": self.after,
        }


def _index_regions(state: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Return ``{region_id: region_dict}`` for a session-state shape."""

    out: dict[str, dict[str, Any]] = {}
    for region in state.get("regions", []) or []:
        if not isinstance(region, dict):
            continue
        rid = str(region.get("region_id", ""))
        if rid:
            out[rid] = region
    return out


# Region-level fields the diff inspects. Anything outside this list is
# ignored — the diff is intentionally narrow so the LLM sees only
# decision-relevant changes.
_TRACKED_REGION_FIELDS: Final[tuple[str, ...]] = (
    "current_tactic",
    "last_verdict",
    "last_reason",
    "open_decision_sites",
    "candidate_set",
)


def diff_since(
    before: dict[str, Any],
    after: dict[str, Any],
) -> list[DiffEntry]:
    """Compute a typed diff between two session-state dicts.

    The diff is *region-keyed*: top-level fields like ``session_id``
    are not compared (they are identity, not state).
    """

    out: list[DiffEntry] = []

    # Plan-level: a plan_version change is one entry; a plan
    # disappearance is one entry; otherwise we don't recurse.
    plan_before = before.get("plan") or {}
    plan_after = after.get("plan") or {}
    pv_before = plan_before.get("plan_version")
    pv_after = plan_after.get("plan_version")
    if pv_before != pv_after:
        out.append(
            DiffEntry(
                pointer="plan.plan_version",
                kind="changed" if pv_before is not None and pv_after is not None
                else ("added" if pv_after is not None else "removed"),
                before=pv_before,
                after=pv_after,
            )
        )

    regions_before = _index_regions(before)
    regions_after = _index_regions(after)
    all_ids = sorted(set(regions_before) | set(regions_after))

    for rid in all_ids:
        b = regions_before.get(rid)
        a = regions_after.get(rid)
        if b is None and a is not None:
            out.append(
                DiffEntry(
                    pointer=f"region.{rid}",
                    kind="added",
                    before=None,
                    after=a,
                )
            )
            continue
        if a is None and b is not None:
            out.append(
                DiffEntry(
                    pointer=f"region.{rid}",
                    kind="removed",
                    before=b,
                    after=None,
                )
            )
            continue
        # Both present: compare tracked fields.
        for field_name in _TRACKED_REGION_FIELDS:
            bf = b.get(field_name) if b else None
            af = a.get(field_name) if a else None
            if bf != af:
                out.append(
                    DiffEntry(
                        pointer=f"region.{rid}.{field_name}",
                        kind="changed",
                        before=bf,
                        after=af,
                    )
                )

    return out


__all__ = ["DIFF_KINDS", "DiffEntry", "diff_since"]
