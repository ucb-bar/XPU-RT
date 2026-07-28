"""
Feedback-driven compilation: post-schedule dispatch-granularity advisor.

xpu-rt's only granularity lever is `fusion_threshold` in `scheduler.schedule()`
(see `fusion.py`), and it only merges small ops into bigger ones -- nothing
here can split an already-coarse dispatch into finer ones. That capability
lives upstream, in whatever compiler produced the dispatch graph (e.g.
ModelBlaster's Model Partitioner / LLM-agentic kernel-gen). So this module is
advisory only: given a solved schedule, it flags non-periodic (best-effort)
jobs whose dispatch granularity is a bad fit for the periodic jobs sharing
the same schedule, so a human (or an upstream optimizer loop) can act on it.

Motivating case: a non-periodic job gets scheduled as one large, unfused
dispatch that occupies a core for far longer than a periodic job's period --
if the two ever need to share a core, that one coarse dispatch blows through
several periodic deadlines before yielding. The fix is to partition that
non-periodic job's dispatches finer upstream, not something xpu-rt can do to
an already-profiled dispatch graph itself.

Two ways to build the `DispatchRecord`s this module analyzes:
  - `from_workload()` -- precise path, used right after `scheduler.schedule()`
    returns, while `min_start_t`/`max_end_t` are still attached to each
    `Operation` (the same periodicity signal `postprocessing.py` already
    uses to tell periodic and best-effort operations apart).
  - `from_schedule_json()` -- fallback for analyzing an already-saved
    schedule JSON (e.g. `schedules/scheduled_networks_deps_4cores_profiled.json`)
    that predates any periodicity metadata this module can write. Recovers
    instance identity from each dispatch's key (`"<instance>_dispatch_<n>..."`)
    rather than its `job_name` field, since real schedule files disagree on
    whether `job_name` holds the per-instance id (`"dronet0"`) or the shared
    base id (`"dronet"`) -- the dispatch key is consistent either way.

The comparison itself uses two signals sharper than a periodic job's raw
period, both already established elsewhere in this codebase:
  - **Free slot, not raw period.** `profile_metrics.py`'s
    `max_periodic_window_fraction()` observes that periodic jobs only
    occupy a fraction (F_p) of their own period -- the actual room
    available to a non-periodic dispatch is `period * (1 - F_p)`, not the
    full period. Here F_p is approximated per periodic job as its own
    critical-path length (see below) divided by its period, computed
    directly from `DispatchRecord`s rather than requiring the original
    profiled CSVs `profile_metrics.py` reads from.
  - **Critical-path length, not summed duration.** A periodic instance's
    dispatches aren't necessarily serial; summing every dispatch's duration
    would overcount work that actually runs in parallel. `_critical_path_ms`
    instead does a duration-weighted longest-path walk over each instance's
    own dependency DAG -- the same chain-aware idea behind
    `greedy_scheduler.py`'s `_compute_alap_deadlines()`, which backward-
    propagates periodic deadlines through dependency chains rather than
    treating an instance's window as uniformly free.
  - **Linear-chain gating for "coarser".** `fusion.py`'s `fuse_operations()`
    only fuses linear chains (each op has at most one predecessor and one
    successor within the chain) -- fusing a branching dependency structure
    isn't legal there. `_is_linear_chain` applies the same precondition
    before ever recommending "coarser" for a non-periodic job: if its
    dispatches branch, coarsening may not be achievable the way xpu-rt's
    own fusion pass works, so the recommendation is capped at "unchanged".
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
import statistics


# A non-periodic job is only flagged "coarser" when its max dispatch
# duration is under this fraction of the tightest periodic job's free slot
# -- i.e. there's enough slack that going coarser (fewer, bigger dispatches
# -- less scheduling/dispatch overhead) would still pose no deadline risk.
# Tunable; see README "Feedback-driven compilation" section.
_COARSER_SLACK_RATIO = 0.1


@dataclass
class DispatchRecord:
    instance_id: str    # e.g. "dronet0", "mobilenet"
    base_id: str        # e.g. "dronet", "mobilenet"
    is_periodic: bool | None   # None => unknown, infer from grouping
    start_time: float
    duration: float
    dispatch_key: str = ""             # e.g. "dronet0_dispatch_5"; defaults to instance_id
    dependencies: list[str] = field(default_factory=list)   # dispatch_keys this depends on

    def __post_init__(self):
        if not self.dispatch_key:
            self.dispatch_key = self.instance_id


@dataclass
class GranularityAdvice:
    subject: str                       # non-periodic job's base_id, e.g. "mobilenet"
    recommended: str                   # "finer" | "coarser" | "unchanged"
    reason: str
    conflicting_periodic_job: str | None
    period_ms: float | None
    free_slot_ms: float | None         # period_ms adjusted for the periodic job's own utilization
    max_dispatch_duration_ms: float

    def as_dict(self) -> dict:
        return asdict(self)


def _strip_trailing_digits(s: str) -> str:
    i = len(s)
    while i > 0 and s[i - 1].isdigit():
        i -= 1
    return s[:i]


def _instance_id_from_dispatch_key(dispatch_key: str) -> str:
    """'dronet0_dispatch_0' -> 'dronet0'; 'mobilenet_dispatch_22_4' -> 'mobilenet'."""
    return dispatch_key.split("_dispatch_", 1)[0]


def from_workload(combined_workload, t, alpha) -> list[DispatchRecord]:
    """Build DispatchRecords from a live post-schedule() workload.

    Uses the same `min_start_t`/`max_end_t is not None` periodicity signal
    as `postprocessing.trim_periodic_after_nonperiodic_makespan`, so
    periodicity here is known, not inferred.
    """
    machine_combinations = combined_workload.get_machine_combinations()
    records = []
    for op_idx, op in enumerate(combined_workload.operations):
        operation_name = (
            op.operation_name if getattr(op, "operation_name", None) else f"op_{op_idx}"
        )
        instance_id = _instance_id_from_dispatch_key(operation_name)
        base_id = _strip_trailing_digits(instance_id)
        is_periodic = (
            getattr(op, "min_start_t", None) is not None
            or getattr(op, "max_end_t", None) is not None
        )
        combo_idx = int(alpha[op_idx].argmax())
        duration = op.get_duration_for_combination(
            combo_idx, machine_combinations, combined_workload.machines
        )
        dependencies = [
            pred.operation_name for pred in op.get_predecessors()
            if getattr(pred, "operation_name", None)
        ]
        records.append(DispatchRecord(
            instance_id=instance_id,
            base_id=base_id,
            is_periodic=is_periodic,
            start_time=float(t[op_idx]),
            duration=float(duration),
            dispatch_key=operation_name,
            dependencies=dependencies,
        ))
    return records


def from_schedule_json(schedule_dict: dict) -> list[DispatchRecord]:
    """Build DispatchRecords from a saved schedule JSON's `dispatches` dict.

    Periodicity is not in this schema (older files never wrote it), so it's
    inferred from grouping: a base_id backed by more than one distinct
    instance_id is treated as periodic. A periodic network with only one
    instance present in this particular schedule window is indistinguishable
    from a genuinely non-periodic singleton under this fallback -- a real
    limitation, not a bug, when no periodicity metadata was persisted.
    """
    records = []
    for dispatch_key, entry in schedule_dict.get("dispatches", {}).items():
        instance_id = _instance_id_from_dispatch_key(dispatch_key)
        base_id = _strip_trailing_digits(instance_id)
        records.append(DispatchRecord(
            instance_id=instance_id,
            base_id=base_id,
            is_periodic=None,
            start_time=float(entry["start_time"]),
            duration=float(entry["duration"]),
            dispatch_key=dispatch_key,
            dependencies=list(entry.get("dependencies", [])),
        ))
    return records


def _critical_path_ms(records: list[DispatchRecord]) -> float:
    """Duration-weighted longest path through `records`' dependency DAG.

    Same chain-aware idea as `greedy_scheduler.py`'s ALAP backward-
    propagation: an instance's own dispatches aren't necessarily serial, so
    summing every dispatch's duration would overcount parallel work. This
    instead finds how long the longest dependent chain actually takes.
    Dependencies pointing outside `records` (e.g. a cross-instance profiling
    artifact) are ignored -- only edges within this instance's own dispatch
    set matter for its own critical path.
    """
    by_key = {r.dispatch_key: r for r in records}
    memo: dict[str, float] = {}

    def longest_finish(key: str) -> float:
        if key in memo:
            return memo[key]
        r = by_key[key]
        upstream = max(
            (longest_finish(dep) for dep in r.dependencies if dep in by_key and dep != key),
            default=0.0,
        )
        memo[key] = upstream + r.duration
        return memo[key]

    if not by_key:
        return 0.0
    return max(longest_finish(k) for k in by_key)


def _is_linear_chain(records: list[DispatchRecord]) -> bool:
    """True if every dispatch in `records` has at most one predecessor and
    at most one successor *within this same set* -- the exact precondition
    `fusion.py`'s `fuse_operations()` requires before it will fuse a chain.
    A job whose dispatches branch can't be coarsened the way xpu-rt's own
    fusion pass works.
    """
    keys = {r.dispatch_key for r in records}
    out_degree: dict[str, int] = {k: 0 for k in keys}
    for r in records:
        in_degree_local = sum(1 for dep in r.dependencies if dep in keys)
        if in_degree_local > 1:
            return False
        for dep in r.dependencies:
            if dep in keys:
                out_degree[dep] += 1
    return all(d <= 1 for d in out_degree.values())


def _period_from_instances(instance_start_times: dict[str, float]) -> float | None:
    """Median of consecutive deltas between instances' earliest start times.

    Median (not mean) so a solver's occasional early/late jitter on one
    instance doesn't skew the inferred period -- workload_factory.py spaces
    periodic instances uniformly by construction, so deltas should cluster
    tightly around the true period.
    """
    if len(instance_start_times) < 2:
        return None
    ordered = sorted(instance_start_times.values())
    deltas = [b - a for a, b in zip(ordered, ordered[1:])]
    return statistics.median(deltas)


def group_by_periodicity(
    records: list[DispatchRecord],
) -> tuple[dict[str, float], dict[str, list[DispatchRecord]]]:
    """Split `records` into periodic base_ids (with their inferred period_ms)
    and non-periodic base_ids (with their records).

    A base is periodic if any record says so explicitly (`is_periodic=True`,
    from `from_workload`) or, absent that signal (`from_schedule_json`), if
    the base has more than one distinct `instance_id`. Everything else is a
    non-periodic (best-effort) job. Shared by `analyze_granularity` and by
    `postprocessing.output_scheduled_json`'s `metadata["periodic_networks"]`.

    `periodic_periods` holds each periodic base's raw period (the ground
    truth spacing fact, used verbatim for `metadata["periodic_networks"]`);
    `analyze_granularity` separately derives each one's *free slot* (period
    adjusted for the job's own utilization) via `_free_slot_ms`.
    """
    by_base: dict[str, list[DispatchRecord]] = {}
    for r in records:
        by_base.setdefault(r.base_id, []).append(r)

    periodic_periods: dict[str, float] = {}
    non_periodic: dict[str, list[DispatchRecord]] = {}

    for base_id, group in by_base.items():
        distinct_instances = {r.instance_id for r in group}
        explicit_periodic = any(r.is_periodic is True for r in group)
        explicit_non_periodic = any(r.is_periodic is False for r in group) and not explicit_periodic
        inferred_periodic = len(distinct_instances) > 1

        is_periodic = explicit_periodic or (inferred_periodic and not explicit_non_periodic)
        if is_periodic:
            earliest_by_instance: dict[str, float] = {}
            for r in group:
                if r.instance_id not in earliest_by_instance or r.start_time < earliest_by_instance[r.instance_id]:
                    earliest_by_instance[r.instance_id] = r.start_time
            period = _period_from_instances(earliest_by_instance)
            if period is not None:
                periodic_periods[base_id] = period
        else:
            non_periodic[base_id] = group

    return periodic_periods, non_periodic


def _free_slot_ms(base_id: str, group: list[DispatchRecord], period: float) -> float:
    """period * (1 - F_p), where F_p approximates how much of the period the
    periodic job itself occupies -- its worst-case (max across instances)
    critical-path length divided by the period. See module docstring.
    """
    by_instance: dict[str, list[DispatchRecord]] = {}
    for r in group:
        by_instance.setdefault(r.instance_id, []).append(r)
    critical_path = max(
        (_critical_path_ms(instance_records) for instance_records in by_instance.values()),
        default=0.0,
    )
    utilization = min(1.0, critical_path / period) if period > 0 else 1.0
    return period * (1.0 - utilization)


def analyze_granularity(records: list[DispatchRecord]) -> list[GranularityAdvice]:
    """The core comparison: non-periodic dispatch duration vs. periodic free slot.

    For each non-periodic job (see `group_by_periodicity`), compares its
    worst-case (max) dispatch duration against the tightest (smallest) free
    slot among periodic jobs in this schedule -- the period adjusted for how
    much of it the periodic job's own critical path actually occupies (see
    `_free_slot_ms`), not the raw period:
      - duration exceeds the tightest free slot -> "finer" (this dispatch
        would overrun that periodic job's cadence if they ever shared a
        core)
      - duration is under `_COARSER_SLACK_RATIO` of it -> "coarser", *but
        only if* the job's own dispatches form a linear chain
        (`_is_linear_chain`) -- xpu-rt's own fusion pass can't merge a
        branching dependency structure, so a branchy job is capped at
        "unchanged" instead
      - otherwise -> "unchanged"

    Returns one GranularityAdvice per non-periodic base_id found. Bases with
    no periodic job anywhere in `records` are skipped -- there's nothing to
    compare against.
    """
    periodic_periods, non_periodic = group_by_periodicity(records)

    if not periodic_periods:
        return []

    by_base: dict[str, list[DispatchRecord]] = {}
    for r in records:
        by_base.setdefault(r.base_id, []).append(r)

    free_slots = {
        base_id: _free_slot_ms(base_id, by_base[base_id], period)
        for base_id, period in periodic_periods.items()
    }
    tightest_base = min(free_slots, key=lambda b: free_slots[b])
    tightest_slot = free_slots[tightest_base]
    tightest_period = periodic_periods[tightest_base]

    advice = []
    for base_id, group in non_periodic.items():
        max_duration = max(r.duration for r in group)
        slot_note = (
            f" (period {tightest_period:.2f} ms, free slot {tightest_slot:.2f} ms "
            f"after {tightest_base}'s own utilization)"
            if abs(tightest_slot - tightest_period) > 1e-6 else ""
        )

        if max_duration > tightest_slot:
            recommended = "finer"
            reason = (
                f"the desired granularity for {base_id}'s dispatches should be "
                f"finer: its largest dispatch ({max_duration:.2f} ms) exceeds "
                f"{tightest_base}'s available free slot ({tightest_slot:.2f} ms)"
                f"{slot_note} -- it would overrun {tightest_base}'s cadence if "
                f"they shared a core"
            )
        elif max_duration < tightest_slot * _COARSER_SLACK_RATIO:
            if _is_linear_chain(group):
                recommended = "coarser"
                reason = (
                    f"{base_id}'s largest dispatch ({max_duration:.2f} ms) is well "
                    f"under {int(_COARSER_SLACK_RATIO * 100)}% of {tightest_base}'s "
                    f"free slot ({tightest_slot:.2f} ms){slot_note} -- fewer, bigger "
                    f"dispatches would cut scheduling overhead without risking "
                    f"{tightest_base}'s deadline"
                )
            else:
                recommended = "unchanged"
                reason = (
                    f"{base_id}'s largest dispatch ({max_duration:.2f} ms) is well "
                    f"under {tightest_base}'s free slot, but {base_id}'s dispatches "
                    f"branch (not a linear chain) -- xpu-rt's own fusion pass can't "
                    f"merge that shape, so no coarsening recommendation is made"
                )
        else:
            recommended = "unchanged"
            reason = (
                f"{base_id}'s largest dispatch ({max_duration:.2f} ms) fits "
                f"comfortably within {tightest_base}'s free slot ({tightest_slot:.2f} ms)"
                f"{slot_note}"
            )

        advice.append(GranularityAdvice(
            subject=base_id,
            recommended=recommended,
            reason=reason,
            conflicting_periodic_job=tightest_base,
            period_ms=tightest_period,
            free_slot_ms=tightest_slot,
            max_dispatch_duration_ms=max_duration,
        ))

    return advice
