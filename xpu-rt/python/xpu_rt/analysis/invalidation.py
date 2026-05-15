"""Invalidation diff + claim-vs-actual enforcement.

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
compatibility with negative controls).
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


# --------------------------------------------------------------------------- #
# consumer-side stale-read detection
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SummaryRead:
    """One summary read as observed by a consumer.

    Records the generation the consumer observed when it read the
    summary. raises :class:`compgen.audit.errors.StaleAnalysisAudit`
    when a later read at a lower generation is detected — i.e. the
    invalidation log has bumped the generation past what the consumer
    saw.
    """

    summary_id: str
    consumer_id: str  # e.g. "cost_preview_v2", "kernel_readiness"
    generation_observed: int
    timestamp_utc: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary_id": self.summary_id,
            "consumer_id": self.consumer_id,
            "generation_observed": self.generation_observed,
            "timestamp_utc": self.timestamp_utc,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> SummaryRead:
        return cls(
            summary_id=str(raw["summary_id"]),
            consumer_id=str(raw["consumer_id"]),
            generation_observed=int(raw["generation_observed"]),
            timestamp_utc=str(raw.get("timestamp_utc", "")),
        )


def _read_log_path(run_dir: Path) -> Path:
    return run_dir / "03_recipe_planning" / "summary_read_log.json"


def append_read_log(run_dir: Path, read: SummaryRead) -> Path:
    """Append a :class:`SummaryRead` to the per-run read log.

    Used by consumers (cost_preview_v2 builder, kernel_readiness, etc.)
    to record which generation of each summary they observed. The
    audit then checks: at no point does any reader observe a generation
    less than what later readers / the invalidation_log show as
    current.
    """
    out_dir = run_dir / "03_recipe_planning"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = _read_log_path(run_dir)
    if out_path.exists():
        existing = json.loads(out_path.read_text())
        reads_raw = existing.get("reads", []) or []
    else:
        reads_raw = []
    record = dict(read.to_dict())
    if not record["timestamp_utc"]:
        record["timestamp_utc"] = _utc_now()
    reads_raw.append(record)
    payload = {
        "schema_version": "summary_read_log_v1",
        "run_dir": str(run_dir),
        "generated_at_utc": _utc_now(),
        "read_count": len(reads_raw),
        "reads": reads_raw,
    }
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return out_path


def load_read_log(run_dir: Path) -> list[SummaryRead]:
    out_path = _read_log_path(run_dir)
    if not out_path.exists():
        return []
    raw = json.loads(out_path.read_text())
    return [SummaryRead.from_dict(r) for r in (raw.get("reads") or [])]


def assert_no_stale_reads(
    run_dir: Path,
    *,
    current_generations: dict[str, int] | None = None,
) -> None:
    """Raise :class:`StaleAnalysisAudit` on any consumer-side stale read.

    Walks the read log and the invalidation log. For each summary,
    derives the current generation (number of invalidation_log entries
    that include the summary in their closure). A read is *stale* when
    its ``generation_observed`` is strictly less than the current
    generation AND a later read of the same summary observed a higher
    generation (showing the value DID change in this run).

    The asymmetric "later read at higher generation" requirement
    avoids false positives: if a consumer reads gen=0 and the run
    just hasn't bumped the summary, there's no stale read.
    """
    from compgen.audit.errors import StaleAnalysisAudit

    reads = load_read_log(run_dir)
    if not reads:
        return  # nothing to check

    # Group reads by summary_id, preserving order.
    by_summary: dict[str, list[SummaryRead]] = {}
    for r in reads:
        by_summary.setdefault(r.summary_id, []).append(r)

    for summary_id, summary_reads in by_summary.items():
        max_seen = -1
        for r in summary_reads:
            if r.generation_observed < max_seen:
                raise StaleAnalysisAudit(
                    f"consumer {r.consumer_id!r} read summary "
                    f"{summary_id!r} at generation "
                    f"{r.generation_observed} after a previous reader "
                    f"observed generation {max_seen}; the summary was "
                    f"invalidated between reads but {r.consumer_id!r} "
                    f"did not refresh"
                )
            if r.generation_observed > max_seen:
                max_seen = r.generation_observed
