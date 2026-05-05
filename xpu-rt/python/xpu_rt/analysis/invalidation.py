"""Invalidation diff + claim-vs-actual enforcement (M-33.1).

When a pass runs, the on-disk analysis summaries change. The pipeline
captures two :class:`AnalysisIndex` snapshots — one before the pass,
one after — and asks this module:

1. Which summaries actually mutated, appeared, or were removed?
2. Are those mutations covered by the pass's declared ``invalidates``
   list (and the transitive closure in
   :data:`compgen.analysis.checkpoints.KNOWN_SUMMARIES`)?
3. Is there a mutation the pass *did not declare*? That is the
   "lying pass" failure mode — the pass mutated a summary the
   invalidation tracker would not flag stale on the consumer side.

If a mutation is not covered, :func:`assert_invalidations_match_claim`
raises :class:`compgen.audit.errors.UnannouncedInvalidation` (which
:class:`compgen.audit.errors.StaleAnalysisAudit` aliases for backward
compatibility with M-31A.5 negative controls).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from compgen.analysis.checkpoints import (
    KNOWN_SUMMARIES,
    AnalysisIndex,
    AnalysisSummaryError,
    assert_resolvable,
)
from compgen.audit.errors import UnannouncedInvalidation


@dataclass(frozen=True)
class InvalidationDiff:
    """Three-way diff between two analysis-index snapshots.

    All three tuples are sorted for byte-stable serialisation.
    """

    mutated: tuple[str, ...]
    appeared: tuple[str, ...]
    removed: tuple[str, ...]

    @property
    def is_empty(self) -> bool:
        return not (self.mutated or self.appeared or self.removed)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mutated": list(self.mutated),
            "appeared": list(self.appeared),
            "removed": list(self.removed),
        }


def compute_invalidation_diff(
    before: AnalysisIndex,
    after: AnalysisIndex,
) -> InvalidationDiff:
    """Compare two snapshots and return the typed diff."""
    raw = before.diff(after)
    return InvalidationDiff(
        mutated=tuple(raw["mutated"]),
        appeared=tuple(raw["appeared"]),
        removed=tuple(raw["removed"]),
    )


def _transitive_closure(claimed: Iterable[str]) -> frozenset[str]:
    """Forward-closure of ``claimed`` over the dependency graph.

    A pass that claims it invalidates ``graph_dossier_v3`` implicitly
    invalidates every summary that depends on it (``cost_preview``,
    ``llm_action_space``, …). We use the same reverse-dependency walk
    that :meth:`AnalysisIndex.transitively_invalidated_by` uses.
    """
    claimed_list = list(claimed)
    if not claimed_list:
        return frozenset()
    assert_resolvable(claimed_list)
    closure: set[str] = set(claimed_list)
    deps_rev: dict[str, set[str]] = {entry.id: set() for entry in KNOWN_SUMMARIES}
    for entry in KNOWN_SUMMARIES:
        for dep in entry.dependencies:
            if dep in deps_rev:
                deps_rev[dep].add(entry.id)
    worklist = list(claimed_list)
    while worklist:
        current = worklist.pop()
        for downstream in deps_rev.get(current, set()):
            if downstream not in closure:
                closure.add(downstream)
                worklist.append(downstream)
    return frozenset(closure)


def assert_invalidations_match_claim(
    diff: InvalidationDiff,
    claimed: Iterable[str],
    *,
    pass_id: str = "<unknown>",
) -> None:
    """Verify that every observed mutation is covered by the claim.

    A mutation observed in ``diff.mutated`` (or ``diff.removed``) but not
    in the transitive closure of ``claimed`` raises
    :class:`UnannouncedInvalidation`. The transitive closure rule means
    a pass need only declare upstream summaries; its claim implicitly
    extends to everything that depends on them.

    ``diff.appeared`` is treated as additive (new artifact emitted) and
    does not require a claim — appearance cannot stale anyone.
    """
    closure = _transitive_closure(claimed)
    observed = set(diff.mutated) | set(diff.removed)
    unannounced = sorted(observed - closure)
    if unannounced:
        raise UnannouncedInvalidation(
            f"pass {pass_id!r} mutated summaries that were not in its "
            f"declared invalidates closure: {unannounced}. "
            f"Claimed: {sorted(set(claimed))}; "
            f"closure: {sorted(closure)}"
        )


# --------------------------------------------------------------------------- #
# Per-run invalidation log
# --------------------------------------------------------------------------- #


def _utc_now() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass(frozen=True)
class InvalidationLogEntry:
    """One pass's invalidation record."""

    schema_version: str
    pass_id: str
    region_id: str
    candidate_id: str
    claimed: tuple[str, ...]
    diff: InvalidationDiff
    closure: tuple[str, ...]
    matches_claim: bool
    timestamp_utc: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "pass_id": self.pass_id,
            "region_id": self.region_id,
            "candidate_id": self.candidate_id,
            "claimed": list(self.claimed),
            "diff": self.diff.to_dict(),
            "closure": list(self.closure),
            "matches_claim": self.matches_claim,
            "timestamp_utc": self.timestamp_utc,
        }


def make_log_entry(
    *,
    pass_id: str,
    region_id: str = "",
    candidate_id: str = "",
    claimed: Iterable[str],
    diff: InvalidationDiff,
) -> InvalidationLogEntry:
    closure = sorted(_transitive_closure(claimed))
    matches = not (set(diff.mutated) | set(diff.removed)) - set(closure)
    return InvalidationLogEntry(
        schema_version="invalidation_log_entry_v1",
        pass_id=pass_id,
        region_id=region_id,
        candidate_id=candidate_id,
        claimed=tuple(sorted(set(claimed))),
        diff=diff,
        closure=tuple(closure),
        matches_claim=bool(matches),
        timestamp_utc=_utc_now(),
    )


def write_invalidation_log(
    run_dir: Path,
    entries: list[InvalidationLogEntry],
) -> Path:
    """Write the per-run invalidation log to ``03_recipe_planning/``."""
    out_dir = run_dir / "03_recipe_planning"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "invalidation_log.json"
    payload = {
        "schema_version": "invalidation_log_v1",
        "run_dir": str(run_dir),
        "generated_at_utc": _utc_now(),
        "entry_count": len(entries),
        "entries": [e.to_dict() for e in entries],
    }
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return out_path


def append_invalidation_log(
    run_dir: Path,
    entry: InvalidationLogEntry,
) -> Path:
    """Append one entry; load existing log if present."""
    out_dir = run_dir / "03_recipe_planning"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "invalidation_log.json"
    if out_path.exists():
        existing = json.loads(out_path.read_text())
        entries_raw = existing.get("entries", []) or []
    else:
        entries_raw = []
    entries_raw.append(entry.to_dict())
    payload = {
        "schema_version": "invalidation_log_v1",
        "run_dir": str(run_dir),
        "generated_at_utc": _utc_now(),
        "entry_count": len(entries_raw),
        "entries": entries_raw,
    }
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return out_path
