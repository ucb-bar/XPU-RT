"""Cross-iteration memory + Beta-Bernoulli Thompson bandit for Stage 4.

Each Stage-4 loop iteration currently starts fresh: same chunking, same
solver pick given the same calibration. This module adds an append-only
JSONL log of past ``(target, workload-set)`` outcomes plus a Thompson
sampling policy over a small grid of ``(max_chunk_ops,
fusion_gain_threshold)`` arms. The loop driver consults
:func:`recommend_initial_arm` at init time and appends a
:class:`MemoryEntry` at each step's end.

Persistence layout::

    <memory_dir>/<target_id>__<workload_set_key>.jsonl

The schema is intentionally narrow (one outcome per line) and append-only;
old entries are never rewritten. ``schema_version`` is captured per-line
so future migrations can detect mixed-version logs.
"""

from __future__ import annotations

import json
import random
import statistics
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

import structlog

log = structlog.get_logger(__name__)

LOOP_MEMORY_SCHEMA_VERSION = "loop_memory_v1"


@dataclass(frozen=True)
class MemoryEntry:
    """One ``(target, workload-set, run, iteration)`` outcome.

    Attributes:
        target_id: e.g. ``"qrb5165"``.
        workload_set_key: Canonical key, see :func:`canonical_workload_set_key`.
        run_id: ISO timestamp of the run that produced this entry.
        iteration: 0-based iteration within the run.
        max_chunk_ops: The granularity knob value used.
        fusion_gain_threshold: The fusion-cost threshold used.
        solver_choice: ``"mosek"`` | ``"cpsat"`` | ``"greedy"``.
        n_partitions: How many partitions resulted.
        predicted_makespan_us: Scheduler's prediction.
        measured_makespan_us: Ground-truth measurement, if any.
        abs_pct_error: ``|pred - measured| / measured * 100``, or ``None``.
        was_converged: True iff this iteration met the convergence rule.
    """

    target_id: str
    workload_set_key: str
    run_id: str
    iteration: int
    max_chunk_ops: int
    fusion_gain_threshold: float
    solver_choice: str
    n_partitions: int
    predicted_makespan_us: float
    measured_makespan_us: float | None
    abs_pct_error: float | None
    was_converged: bool


@dataclass(frozen=True)
class BanditArm:
    """One configuration choice the bandit can pick."""

    max_chunk_ops: int
    fusion_gain_threshold: float


@dataclass(frozen=True)
class BanditStats:
    """Posterior stats for one arm under Beta-Bernoulli Thompson sampling.

    'Success' = converged with ``abs_pct_error <= success_threshold_pct``.
    """

    arm: BanditArm
    successes: int
    failures: int
    last_seen_iso: str

    def sample_score(self, *, rng_seed: int | None = None) -> float:
        """Thompson sample from ``Beta(successes+1, failures+1)``."""

        rng = random.Random(rng_seed) if rng_seed is not None else random
        # Beta(a, b) — uninformative Beta(1,1) prior added so unseen arms
        # explore at 50/50 instead of collapsing to zero.
        return rng.betavariate(self.successes + 1, self.failures + 1)


def canonical_workload_set_key(workload_ids: Iterable[str]) -> str:
    """Stable key for a multi-workload combination.

    Counts occurrences, sorts alphabetically, and joins ``<id>*<count>``
    with ``"+"``. e.g. ``("dronet", "yolov8n", "dronet") ->
    "dronet*2+yolov8n*1"``. Cardinalities are emitted explicitly so the
    closed-loop scenario ``yolov8n + 12x dronet`` survives serialisation.
    """

    counts: dict[str, int] = {}
    for wid in workload_ids:
        counts[wid] = counts.get(wid, 0) + 1
    return "+".join(f"{wid}*{counts[wid]}" for wid in sorted(counts))


def _entry_path(target_id: str, workload_set_key: str, memory_dir: Path) -> Path:
    return Path(memory_dir) / f"{target_id}__{workload_set_key}.jsonl"


def append_entry(entry: MemoryEntry, memory_dir: Path) -> None:
    """Append-only JSONL log at
    ``memory_dir / "<target_id>__<workload_set_key>.jsonl"``.
    """

    path = _entry_path(entry.target_id, entry.workload_set_key, memory_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(entry)
    payload["schema_version"] = LOOP_MEMORY_SCHEMA_VERSION
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, sort_keys=True))
        f.write("\n")
    log.debug(
        "loop_memory_append",
        target=entry.target_id,
        workload_set=entry.workload_set_key,
        iteration=entry.iteration,
        arm_max_chunk_ops=entry.max_chunk_ops,
        was_converged=entry.was_converged,
    )


def load_entries(
    target_id: str,
    workload_set_key: str,
    memory_dir: Path,
) -> tuple[MemoryEntry, ...]:
    """Read all entries for this ``(target, workload-set)``.

    Empty tuple if no history. Unknown / unparseable lines are skipped
    with a warning rather than raising; the log is append-only and a
    partially-written line should not poison the bandit.
    """

    path = _entry_path(target_id, workload_set_key, memory_dir)
    if not path.is_file():
        return ()
    out: list[MemoryEntry] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            log.warning("loop_memory_skip_unparseable", path=str(path))
            continue
        # Drop schema_version from the dict shape so MemoryEntry's frozen
        # constructor stays minimal; we don't yet have a v2 to migrate to.
        d.pop("schema_version", None)
        try:
            out.append(
                MemoryEntry(
                    target_id=str(d["target_id"]),
                    workload_set_key=str(d["workload_set_key"]),
                    run_id=str(d["run_id"]),
                    iteration=int(d["iteration"]),
                    max_chunk_ops=int(d["max_chunk_ops"]),
                    fusion_gain_threshold=float(d["fusion_gain_threshold"]),
                    solver_choice=str(d["solver_choice"]),
                    n_partitions=int(d["n_partitions"]),
                    predicted_makespan_us=float(d["predicted_makespan_us"]),
                    measured_makespan_us=(
                        None
                        if d.get("measured_makespan_us") is None
                        else float(d["measured_makespan_us"])
                    ),
                    abs_pct_error=(
                        None
                        if d.get("abs_pct_error") is None
                        else float(d["abs_pct_error"])
                    ),
                    was_converged=bool(d["was_converged"]),
                )
            )
        except (KeyError, ValueError, TypeError) as exc:
            log.warning(
                "loop_memory_skip_invalid",
                path=str(path),
                error=type(exc).__name__,
            )
    return tuple(out)


def default_candidate_arms() -> tuple[BanditArm, ...]:
    """Sensible default grid: Cartesian product of ``max_chunk_ops`` x
    ``fusion_gain_threshold`` over ``{4, 8, 16, 32, 64} x {0.1, 0.3, 0.5}``.
    """

    return tuple(
        BanditArm(max_chunk_ops=mco, fusion_gain_threshold=fgt)
        for mco in (4, 8, 16, 32, 64)
        for fgt in (0.1, 0.3, 0.5)
    )


def update_stats_from_entry(
    stats: dict[BanditArm, BanditStats],
    entry: MemoryEntry,
    *,
    success_threshold_pct: float = 10.0,
) -> dict[BanditArm, BanditStats]:
    """Bump successes/failures for the matching arm.

    A 'success' requires both ``was_converged`` and ``abs_pct_error <=
    success_threshold_pct``. Entries without measurement (``abs_pct_error
    is None``) are treated as failures: we cannot credit an arm we never
    observed convergence on.
    """

    arm = BanditArm(
        max_chunk_ops=entry.max_chunk_ops,
        fusion_gain_threshold=entry.fusion_gain_threshold,
    )
    prior = stats.get(
        arm,
        BanditStats(arm=arm, successes=0, failures=0, last_seen_iso=entry.run_id),
    )
    is_success = (
        entry.was_converged
        and entry.abs_pct_error is not None
        and entry.abs_pct_error <= success_threshold_pct
    )
    new = BanditStats(
        arm=arm,
        successes=prior.successes + (1 if is_success else 0),
        failures=prior.failures + (0 if is_success else 1),
        last_seen_iso=entry.run_id,
    )
    out = dict(stats)
    out[arm] = new
    return out


def _build_stats(
    entries: tuple[MemoryEntry, ...],
    candidate_arms: tuple[BanditArm, ...],
    success_threshold_pct: float,
) -> dict[BanditArm, BanditStats]:
    stats: dict[BanditArm, BanditStats] = {
        arm: BanditStats(arm=arm, successes=0, failures=0, last_seen_iso="")
        for arm in candidate_arms
    }
    for entry in entries:
        arm = BanditArm(
            max_chunk_ops=entry.max_chunk_ops,
            fusion_gain_threshold=entry.fusion_gain_threshold,
        )
        # Off-grid arms (e.g. from earlier configs) are admitted so we
        # don't silently drop history; they just don't compete against
        # the on-grid candidates this run will pick from.
        if arm not in stats:
            continue
        stats = update_stats_from_entry(
            stats, entry, success_threshold_pct=success_threshold_pct
        )
    return stats


def recommend_initial_arm(
    target_id: str,
    workload_set_key: str,
    candidate_arms: tuple[BanditArm, ...],
    memory_dir: Path,
    success_threshold_pct: float = 10.0,
    rng_seed: int | None = None,
) -> BanditArm:
    """Thompson sampling over arms based on prior memory.

    For each arm: count entries that converged AND ``abs_pct_error <=
    success_threshold_pct`` as 'successes'; count entries that did not
    converge or had ``abs_pct_error > threshold`` as 'failures'. Arms not
    yet seen get ``Beta(1, 1)`` priors. Returns the arm with the highest
    sampled score.

    If memory is empty, picks the arm with the median ``max_chunk_ops``
    as a sensible default starting point (so an empty-memory cold start
    doesn't pin to an extreme of the grid).
    """

    entries = load_entries(target_id, workload_set_key, memory_dir)
    if not entries:
        sorted_arms = sorted(candidate_arms, key=lambda a: a.max_chunk_ops)
        return sorted_arms[len(sorted_arms) // 2]

    stats = _build_stats(entries, candidate_arms, success_threshold_pct)
    rng = random.Random(rng_seed) if rng_seed is not None else random

    # Each arm gets its own draw; we don't reuse rng_seed per-arm because
    # that would correlate draws across arms and defeat exploration.
    best_arm: BanditArm | None = None
    best_score = -1.0
    for arm in candidate_arms:
        s = stats[arm]
        score = rng.betavariate(s.successes + 1, s.failures + 1)
        if score > best_score:
            best_score = score
            best_arm = arm
    assert best_arm is not None  # candidate_arms is non-empty by contract
    return best_arm


def summarize_memory(
    target_id: str,
    workload_set_key: str,
    memory_dir: Path,
    candidate_arms: tuple[BanditArm, ...] | None = None,
    success_threshold_pct: float = 10.0,
) -> dict[str, object]:
    """Compute the summary dict consumed by the MCP status tool.

    Returns ``n_entries``, ``n_converged``, ``best_arm`` (lowest mean
    abs_pct_error), ``best_arm_mean_error_pct``, and ``arm_stats``.
    """

    arms = candidate_arms if candidate_arms is not None else default_candidate_arms()
    entries = load_entries(target_id, workload_set_key, memory_dir)
    stats = _build_stats(entries, arms, success_threshold_pct)

    by_arm_errors: dict[BanditArm, list[float]] = {arm: [] for arm in arms}
    for entry in entries:
        arm = BanditArm(
            max_chunk_ops=entry.max_chunk_ops,
            fusion_gain_threshold=entry.fusion_gain_threshold,
        )
        if arm in by_arm_errors and entry.abs_pct_error is not None:
            by_arm_errors[arm].append(entry.abs_pct_error)

    best_arm: BanditArm | None = None
    best_mean = float("inf")
    for arm, errs in by_arm_errors.items():
        if not errs:
            continue
        m = statistics.fmean(errs)
        if m < best_mean:
            best_mean = m
            best_arm = arm

    arm_stats_payload: list[dict[str, object]] = []
    for arm in arms:
        s = stats[arm]
        errs = by_arm_errors[arm]
        arm_stats_payload.append(
            {
                "max_chunk_ops": arm.max_chunk_ops,
                "fusion_gain_threshold": arm.fusion_gain_threshold,
                "successes": s.successes,
                "failures": s.failures,
                "mean_error_pct": (statistics.fmean(errs) if errs else None),
                "n_observations": len(errs),
            }
        )

    return {
        "n_entries": len(entries),
        "n_converged": sum(1 for e in entries if e.was_converged),
        "best_arm": (
            None
            if best_arm is None
            else {
                "max_chunk_ops": best_arm.max_chunk_ops,
                "fusion_gain_threshold": best_arm.fusion_gain_threshold,
            }
        ),
        "best_arm_mean_error_pct": (None if best_arm is None else best_mean),
        "arm_stats": arm_stats_payload,
    }


__all__ = [
    "LOOP_MEMORY_SCHEMA_VERSION",
    "BanditArm",
    "BanditStats",
    "MemoryEntry",
    "append_entry",
    "canonical_workload_set_key",
    "default_candidate_arms",
    "load_entries",
    "recommend_initial_arm",
    "summarize_memory",
    "update_stats_from_entry",
]
