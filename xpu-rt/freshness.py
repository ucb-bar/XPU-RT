"""Producer-consumer freshness evaluation for dependent task chains.

A task can meet its own deadline and still produce an invalid output, because
the input it acted on was already stale by the time it acted. This module makes
that distinction first-class and measurable:

    deadline_valid   the consumer finished by its own deadline
    freshness_valid  the input it consumed was younger than the freshness window
    output_valid     both

The central experimental question is whether these diverge — whether a schedule
can report ~100% deadline success while silently emitting stale control outputs.

Scope and honesty
-----------------
Freshness here is a property *imposed on a schedule and evaluated
analytically*, not a property observed from the running dataflow. Neither
XPU-RT nor ModelBlaster passes a tensor from the perception network to the
control network: XPU-RT's `edges` become MILP precedence constraints
(workload_factory), and ModelBlaster's `deps` / `time_dep_entry_id` become
semaphore ordering edges (pipeline/ingest_xpurt_schedule.py). Buffers are named
per (model, tensor) with no instance or version tag, and all instances of a
network share one output buffer.

So which producer instance a consumer "consumed" is *inferred from timestamps*,
never recorded. Every record carries `producer_instance_provenance` saying so.
The value "explicitly_recorded" is reserved for a future runtime that tags
buffers with a version; nothing emits it today. Do not describe results from
this module as a measurement of runtime dataflow freshness.

Deliberately scheduler-agnostic: the evaluator consumes a normalized list of
`Invocation` intervals, so the same code serves solver fixtures, simulator
traces, and (later) ModelBlaster execution traces. Adapters live outside.

Units
-----
Unit-agnostic arithmetic, but every input must use ONE unit and the caller must
declare it (`time_unit`, default "us"). Mixed ms/us is the single most likely
way to get a plausible-looking wrong answer here: the repo has historically
compared `deadline_us` against millisecond durations without conversion.
`evaluate_freshness` records the declared unit in the evaluation context so
downstream CSV/manifest consumers cannot lose it.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


# --- invalid-reason vocabulary (closed set) ---------------------------------

VALID = "valid"
DEADLINE_MISS = "deadline_miss"
STALE_INPUT = "stale_input"
DEADLINE_AND_STALE = "deadline_and_stale"
NO_COMPLETED_PRODUCER = "no_completed_producer"

INVALID_REASONS = (
    VALID,
    DEADLINE_MISS,
    STALE_INPUT,
    DEADLINE_AND_STALE,
    NO_COMPLETED_PRODUCER,
)

# How the producer instance was attributed. See module docstring.
INFERRED = "inferred_from_schedule_timestamps"
EXPLICIT = "explicitly_recorded"  # reserved; nothing emits this yet

# Sample-time semantics: when the producer's input data was captured.
SAMPLE_AT_RELEASE = "producer_release"
SAMPLE_AT_START = "producer_start"
SAMPLE_SEMANTICS = (SAMPLE_AT_RELEASE, SAMPLE_AT_START)

# Consumption policies.
LATEST_COMPLETED = "latest_completed"
NEWEST_VERSION = "newest_version"
RELEASE_MATCHED = "release_matched"
# `latest_completed` stays first and stays the default: every result reported
# before these alternatives existed used it, and reordering would silently
# restate them.
CONSUMPTION_POLICIES = (LATEST_COMPLETED, NEWEST_VERSION, RELEASE_MATCHED)


# --- inputs ----------------------------------------------------------------


@dataclass(frozen=True)
class Invocation:
    """One execution of one instance of one task.

    `task` is the logical task name shared by all instances (e.g. "dronet"),
    NOT the per-instance identifier ("dronet0"). Instance identity lives in
    `instance` so producer selection can order instances of the same task.

    `sample_time` is when this invocation's input data was captured. Left None,
    it is derived from the edge's `sample_time_semantics`. Set it explicitly
    when a real sensor timestamp is available — that is strictly better than
    deriving it from release.
    """

    task: str
    instance: int
    release_time: float
    start_time: float
    end_time: float
    deadline: Optional[float] = None
    sample_time: Optional[float] = None

    def resolved_sample_time(self, semantics: str) -> float:
        if self.sample_time is not None:
            return float(self.sample_time)
        if semantics == SAMPLE_AT_RELEASE:
            return float(self.release_time)
        if semantics == SAMPLE_AT_START:
            return float(self.start_time)
        raise ValueError(
            f"unknown sample_time_semantics {semantics!r}; "
            f"expected one of {SAMPLE_SEMANTICS}"
        )


@dataclass(frozen=True)
class FreshnessEdge:
    """A producer -> consumer relationship carrying a freshness requirement.

    The window lives on the edge rather than on either task because freshness
    is a property of the relationship: the same perception output can be fresh
    enough for a logger and far too stale for a controller.
    """

    producer_task: str
    consumer_task: str
    freshness_window: float
    sample_time_semantics: str = SAMPLE_AT_RELEASE
    consumption_policy: str = LATEST_COMPLETED
    criticality: str = "hard"

    def __post_init__(self) -> None:
        if self.sample_time_semantics not in SAMPLE_SEMANTICS:
            raise ValueError(
                f"edge {self.producer_task}->{self.consumer_task}: "
                f"sample_time_semantics {self.sample_time_semantics!r} not in "
                f"{SAMPLE_SEMANTICS}"
            )
        if self.consumption_policy not in CONSUMPTION_POLICIES:
            raise ValueError(
                f"edge {self.producer_task}->{self.consumer_task}: "
                f"consumption_policy {self.consumption_policy!r} not in "
                f"{CONSUMPTION_POLICIES}"
            )
        if not self.freshness_window > 0:
            raise ValueError(
                f"edge {self.producer_task}->{self.consumer_task}: "
                f"freshness_window must be positive, got {self.freshness_window}"
            )


# --- outputs ---------------------------------------------------------------


@dataclass
class FreshnessRecord:
    """One consumer invocation evaluated against one freshness edge.

    Field order is the per-invocation CSV column order.
    """

    # experiment context
    experiment_id: str
    seed: Optional[int]
    policy: str
    candidate_id: str
    epoch: Optional[int]
    contention_level: Optional[float]
    freshness_window: float

    # consumer
    consumer_task: str
    consumer_instance: int
    consumer_release_time: float
    consumer_start_time: float
    consumer_end_time: float
    consumer_deadline: Optional[float]

    # producer (all None when no producer had completed)
    producer_task: str
    producer_instance: Optional[int]
    producer_sample_time: Optional[float]
    producer_release_time: Optional[float]
    producer_start_time: Optional[float]
    producer_end_time: Optional[float]

    # ages: at_start is age when the consumer began acting on the input;
    # at_output is age when the consumer's result became available. The paper
    # may need to distinguish "data age at consumption" from "age when the
    # actuation command was produced", so both are kept.
    input_age_at_start: Optional[float]
    input_age_at_output: Optional[float]

    # verdicts
    deadline_valid: bool
    freshness_valid: bool
    output_valid: bool
    invalid_reason: str

    # provenance
    producer_instance_provenance: str = INFERRED

    def as_row(self) -> Dict[str, object]:
        return asdict(self)


CSV_COLUMNS: Tuple[str, ...] = tuple(
    f.name for f in FreshnessRecord.__dataclass_fields__.values()  # type: ignore[attr-defined]
)


@dataclass
class FreshnessEvaluation:
    records: List[FreshnessRecord] = field(default_factory=list)
    aggregate: Dict[str, object] = field(default_factory=dict)
    context: Dict[str, object] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.records)

    def rows(self) -> List[Dict[str, object]]:
        return [r.as_row() for r in self.records]

    def reason_counts(self) -> Dict[str, int]:
        counts = {r: 0 for r in INVALID_REASONS}
        for rec in self.records:
            counts[rec.invalid_reason] += 1
        return counts


# --- producer selection ----------------------------------------------------


# Relative tolerance for the three inclusive-boundary comparisons below
# (producer eligibility, consumer deadline, freshness window).
#
# WHY THIS IS NOT COSMETIC. All three boundaries are documented as INCLUSIVE, but
# they were implemented as exact float comparisons, and this evaluation puts
# invocations exactly ON them by construction: phi is anchored on A0, the
# measured uncontended age ceiling, and the uncontended ages ARE A0. So at
# phi = A0 + delta there are consumer instances whose age equals phi to the last
# bit, and their verdict was decided by accumulated rounding.
#
# Measured instance of the failure: the same consumer instance, same schedule,
# evaluated on two arithmetically equivalent timing bases, came out
# age = 70.54607400000002 against phi = 70.546074 (stale) in one and
# age = phi = 2821.84296 (valid) in the other -- a 1.4e-14 disagreement flipping
# a reported rate by 1/30. Exact comparison made the result depend on the order
# floating-point operations happened to be performed in.
#
# 1e-9 relative is ~7e-8 ms at these magnitudes: far above the ~1e-14 noise and
# far below the smallest real age difference in this workload (10 ms, the control
# period). Ties therefore resolve to VALID / ELIGIBLE, which is what "inclusive"
# was always supposed to mean.
BOUNDARY_RTOL = 1e-9


def _lte(a: float, b: float, rtol: float = BOUNDARY_RTOL) -> bool:
    """a <= b, treating an exact-boundary tie as satisfying the inequality."""
    return a <= b + rtol * max(1.0, abs(b))


def select_producer(
    producers: Sequence[Invocation],
    consumer_start_time: float,
    policy: str = LATEST_COMPLETED,
    *,
    consumer_release_time: Optional[float] = None,
) -> Optional[Invocation]:
    """Return the producer instance a consumer consumes, or None if none is usable.

    Every policy shares one physical constraint: a producer is only readable once
    it has been written, i.e. `end_time <= consumer_start_time`. The boundary is
    inclusive -- a producer finishing exactly at the consumer's start time IS
    eligible. Zero-duration gaps are the normal case in solver output rather than
    a measure-zero edge case, and excluding them would silently reclassify a whole
    class of tight schedules as no_completed_producer. See BOUNDARY_RTOL.

    The policies differ in WHICH readable instance counts as "the input", and the
    difference is not cosmetic: it changes whether a late producer shows up as
    stale input or as no input at all.

    `latest_completed` -- the most recently WRITTEN sample: greatest end_time.
        Ties break toward the higher instance index, the fresher sample. This is
        deliberately not "most recently released": with heterogeneous backends
        producers can complete out of release order, and a consumer cannot read a
        result that has not been written.

    `newest_version` -- the freshest SAMPLE available: greatest instance index
        among the readable ones, ties breaking toward the earlier end_time. Models
        a versioned buffer where the consumer takes the newest version present.
        Differs from latest_completed exactly when producers complete out of
        release order, which is the heterogeneous case this project is about.

    `release_matched` -- strictly the CURRENT frame, no substitution: the producer
        whose release is the latest one at or before the consumer's release, and
        only if it is readable. If that instance has not completed, the result is
        None even though older completed samples exist. Models a controller that
        refuses to actuate on anything but the current frame, which is the
        conservative reading of a freshness requirement.
        Requires `consumer_release_time`.
    """
    if policy not in CONSUMPTION_POLICIES:
        raise ValueError(
            f"unknown consumption_policy {policy!r}; expected one of "
            f"{CONSUMPTION_POLICIES}"
        )

    readable = [p for p in producers if _lte(p.end_time, consumer_start_time)]

    if policy == LATEST_COMPLETED:
        best: Optional[Invocation] = None
        for p in readable:
            if best is None or (p.end_time, p.instance) > (best.end_time, best.instance):
                best = p
        return best

    if policy == NEWEST_VERSION:
        best = None
        for p in readable:
            if best is None or (p.instance, -p.end_time) > (best.instance, -best.end_time):
                best = p
        return best

    # release_matched
    if consumer_release_time is None:
        raise ValueError(
            f"consumption_policy {policy!r} needs consumer_release_time: it "
            f"selects the producer frame matching the consumer's release, which "
            f"cannot be derived from the start time alone"
        )
    frame: Optional[Invocation] = None
    for p in producers:
        if not _lte(p.release_time, consumer_release_time):
            continue
        if frame is None or (p.release_time, p.instance) > (frame.release_time, frame.instance):
            frame = p
    if frame is None:
        return None
    return frame if _lte(frame.end_time, consumer_start_time) else None


# --- evaluation ------------------------------------------------------------


def evaluate_freshness(
    trace: Iterable[Invocation],
    workload: Optional[object] = None,
    dependency_edges: Sequence[FreshnessEdge] = (),
    consumption_policy: str = LATEST_COMPLETED,
    *,
    experiment_id: str = "",
    seed: Optional[int] = None,
    policy: str = "",
    candidate_id: str = "",
    contention_level: Optional[float] = None,
    epoch_length: Optional[float] = None,
    time_unit: str = "us",
    provenance: Optional[Dict[str, object]] = None,
    pipeline_fill_ms: Optional[float] = None,
) -> FreshnessEvaluation:
    """Evaluate producer-consumer freshness over a trace.

    `trace` is any iterable of `Invocation`. `workload` is accepted and recorded
    for provenance but not required — keeping the evaluator independent of the
    XPU-RT Workload type is what lets the same code score solver fixtures,
    simulator traces, and hardware traces.

    `dependency_edges` carries the freshness windows. An edge's own
    `consumption_policy` wins over the `consumption_policy` argument, which is
    only the default for edges that do not state one.

    Returns per-invocation records plus aggregate rates. One record per
    (consumer invocation, edge), so a consumer with two producers yields two
    records and both must be valid for that output to be usable — that
    conjunction is left to the caller, since how to combine multiple input
    chains is a modelling choice, not arithmetic.
    """
    invocations = list(trace)
    by_task: Dict[str, List[Invocation]] = {}
    for inv in invocations:
        by_task.setdefault(inv.task, []).append(inv)

    records: List[FreshnessRecord] = []

    for edge in dependency_edges:
        producers = sorted(
            by_task.get(edge.producer_task, []), key=lambda i: (i.end_time, i.instance)
        )
        consumers = sorted(
            by_task.get(edge.consumer_task, []), key=lambda i: (i.release_time, i.instance)
        )
        if not consumers:
            # A declared edge whose consumer never appears in the trace is a
            # workload/trace mismatch, not an absence of consumers. Say so
            # rather than silently contributing zero records.
            raise ValueError(
                f"edge {edge.producer_task}->{edge.consumer_task}: consumer task "
                f"{edge.consumer_task!r} has no invocations in the trace "
                f"(tasks present: {sorted(by_task)})"
            )

        eff_policy = edge.consumption_policy or consumption_policy

        for c in consumers:
            p = select_producer(producers, c.start_time, eff_policy,
                                consumer_release_time=c.release_time)

            deadline_valid = c.deadline is None or _lte(c.end_time, c.deadline)

            if p is None:
                records.append(
                    FreshnessRecord(
                        experiment_id=experiment_id,
                        seed=seed,
                        policy=policy,
                        candidate_id=candidate_id,
                        epoch=_epoch_of(c.release_time, epoch_length),
                        contention_level=contention_level,
                        freshness_window=edge.freshness_window,
                        consumer_task=c.task,
                        consumer_instance=c.instance,
                        consumer_release_time=c.release_time,
                        consumer_start_time=c.start_time,
                        consumer_end_time=c.end_time,
                        consumer_deadline=c.deadline,
                        producer_task=edge.producer_task,
                        producer_instance=None,
                        producer_sample_time=None,
                        producer_release_time=None,
                        producer_start_time=None,
                        producer_end_time=None,
                        input_age_at_start=None,
                        input_age_at_output=None,
                        deadline_valid=deadline_valid,
                        freshness_valid=False,
                        output_valid=False,
                        invalid_reason=NO_COMPLETED_PRODUCER,
                    )
                )
                continue

            sample_t = p.resolved_sample_time(edge.sample_time_semantics)
            age_at_start = c.start_time - sample_t
            age_at_output = c.end_time - sample_t

            freshness_valid = _lte(age_at_output, edge.freshness_window)
            output_valid = deadline_valid and freshness_valid

            records.append(
                FreshnessRecord(
                    experiment_id=experiment_id,
                    seed=seed,
                    policy=policy,
                    candidate_id=candidate_id,
                    epoch=_epoch_of(c.release_time, epoch_length),
                    contention_level=contention_level,
                    freshness_window=edge.freshness_window,
                    consumer_task=c.task,
                    consumer_instance=c.instance,
                    consumer_release_time=c.release_time,
                    consumer_start_time=c.start_time,
                    consumer_end_time=c.end_time,
                    consumer_deadline=c.deadline,
                    producer_task=p.task,
                    producer_instance=p.instance,
                    producer_sample_time=sample_t,
                    producer_release_time=p.release_time,
                    producer_start_time=p.start_time,
                    producer_end_time=p.end_time,
                    input_age_at_start=age_at_start,
                    input_age_at_output=age_at_output,
                    deadline_valid=deadline_valid,
                    freshness_valid=freshness_valid,
                    output_valid=output_valid,
                    invalid_reason=_classify(deadline_valid, freshness_valid),
                )
            )

    context: Dict[str, object] = {
        "experiment_id": experiment_id,
        "seed": seed,
        "policy": policy,
        "candidate_id": candidate_id,
        "contention_level": contention_level,
        "epoch_length": epoch_length,
        "time_unit": time_unit,
        "consumption_policy": consumption_policy,
        "producer_instance_provenance": INFERRED,
        "edges": [asdict(e) for e in dependency_edges],
        "n_invocations_in_trace": len(invocations),
        "tasks_in_trace": sorted(by_task),
    }
    if workload is not None:
        context["workload_type"] = type(workload).__name__
    if provenance:
        context["timing_provenance"] = dict(provenance)

    return FreshnessEvaluation(
        records=records,
        aggregate=aggregate_metrics(records, pipeline_fill_ms=pipeline_fill_ms),
        context=context,
    )


def _classify(deadline_valid: bool, freshness_valid: bool) -> str:
    if deadline_valid and freshness_valid:
        return VALID
    if deadline_valid and not freshness_valid:
        return STALE_INPUT
    if not deadline_valid and freshness_valid:
        return DEADLINE_MISS
    return DEADLINE_AND_STALE


def _epoch_of(t: float, epoch_length: Optional[float]) -> Optional[int]:
    if epoch_length is None or epoch_length <= 0:
        return None
    return int(t // epoch_length)


def aggregate_metrics(
    records: Sequence[FreshnessRecord],
    *,
    pipeline_fill_ms: Optional[float] = None,
) -> Dict[str, object]:
    """Aggregate rates and age percentiles over per-invocation records.

    Age percentiles are computed over records that HAVE an age, i.e. excluding
    no_completed_producer. Those are counted separately rather than folded in
    as an infinite age, because a missing producer and a very old producer are
    different failures with different fixes, and imputing a value for the
    former would quietly move the percentiles.

    `pipeline_fill_ms` separates two failures that `output_valid_rate` otherwise
    merges, and the merge was actively misleading. Measured at phi = A0+20 on the
    canonical workload, static_nominal at B=1 lost 11 of 30 consumer invocations
    -- but only 2 were STALE_INPUT. The other 9 were NO_COMPLETED_PRODUCER: the
    consumer had no input at all, because the producer's first instance lost the
    t=0 race to the soft burst and did not finish until 87 ms. "Meets its deadline
    while acting on stale input" and "has nothing to act on" are different claims,
    and only the first is the phenomenon under study.

    Two of those 9 are unavoidable at ANY contention level, including B=0: the
    pipeline starts empty, so a consumer released before the producer could
    possibly have finished even uncontended is structurally unservable. That is a
    property of the workload, not of the policy.

    So `pipeline_fill_ms` must be the UNCONTENDED first-producer completion time
    -- a fixed workload constant -- and never each policy's own first completion.
    Deriving it per policy would excuse a policy that starves the producer for
    87 ms from exactly the 87 ms of damage being measured, the same way a
    self-relative deadline window excuses a candidate for accepting a harder
    target (see trace.deadline_compliance).

    Rates are emitted both over the full trace and over the steady-state subset
    (`steady_*`); neither replaces the other, and the excluded count is reported.
    """
    total = len(records)
    if total == 0:
        return {
            "total_consumer_invocations": 0,
            "deadline_success_rate": None,
            "freshness_success_rate": None,
            "output_valid_rate": None,
            "p50_input_age": None,
            "p95_input_age": None,
            "p99_input_age": None,
            "max_input_age": None,
            "deadline_miss_count": 0,
            "stale_input_count": 0,
            "no_producer_count": 0,
            "n_with_age": 0,
            "pipeline_fill_ms": pipeline_fill_ms,
            "structurally_unservable_count": 0,
            "steady_total_consumer_invocations": 0,
            "steady_output_valid_rate": None,
            "steady_stale_input_rate": None,
            "steady_no_producer_rate": None,
        }

    ages = [
        r.input_age_at_output
        for r in records
        if r.input_age_at_output is not None
    ]
    ages_at_start = [
        r.input_age_at_start for r in records if r.input_age_at_start is not None
    ]

    def pct(vals: Sequence[float], q: float) -> Optional[float]:
        if not vals:
            return None
        return float(np.percentile(np.asarray(vals, dtype=float), q))

    n_deadline_ok = sum(1 for r in records if r.deadline_valid)
    n_fresh_ok = sum(1 for r in records if r.freshness_valid)
    n_output_ok = sum(1 for r in records if r.output_valid)

    # Steady state = the subset a scheduling policy could in principle have
    # served. A consumer that starts before the producer could have finished even
    # with the machine to itself is unservable by construction.
    if pipeline_fill_ms is None:
        steady = list(records)
        n_unservable = 0
    else:
        steady = [r for r in records
                  if r.consumer_start_time >= float(pipeline_fill_ms)]
        n_unservable = total - len(steady)
    n_steady = len(steady)

    def _rate(pred) -> Optional[float]:
        if n_steady == 0:
            return None
        return sum(1 for r in steady if pred(r)) / n_steady

    return {
        "total_consumer_invocations": total,
        "deadline_success_rate": n_deadline_ok / total,
        "freshness_success_rate": n_fresh_ok / total,
        "output_valid_rate": n_output_ok / total,
        "p50_input_age": pct(ages, 50),
        "p95_input_age": pct(ages, 95),
        "p99_input_age": pct(ages, 99),
        "max_input_age": max(ages) if ages else None,
        "p50_input_age_at_start": pct(ages_at_start, 50),
        "p95_input_age_at_start": pct(ages_at_start, 95),
        "max_input_age_at_start": max(ages_at_start) if ages_at_start else None,
        # Counts are by primary reason, so they partition `total` together with
        # the valid count — deadline_and_stale is its own bucket rather than
        # being added into both deadline_miss and stale_input.
        "valid_count": sum(1 for r in records if r.invalid_reason == VALID),
        "deadline_miss_count": sum(
            1 for r in records if r.invalid_reason == DEADLINE_MISS
        ),
        "stale_input_count": sum(
            1 for r in records if r.invalid_reason == STALE_INPUT
        ),
        "deadline_and_stale_count": sum(
            1 for r in records if r.invalid_reason == DEADLINE_AND_STALE
        ),
        "no_producer_count": sum(
            1 for r in records if r.invalid_reason == NO_COMPLETED_PRODUCER
        ),
        "n_with_age": len(ages),
        # Rates over the full trace, decomposed. output_valid_rate alone cannot
        # distinguish "acted on stale input" from "had no input", and the two
        # were 2 vs 9 out of 30 at the operating point this project reports.
        "stale_input_rate": sum(
            1 for r in records
            if r.invalid_reason in (STALE_INPUT, DEADLINE_AND_STALE)
        ) / total,
        "no_producer_rate": sum(
            1 for r in records if r.invalid_reason == NO_COMPLETED_PRODUCER
        ) / total,
        # Steady state: the same rates over invocations a policy could have served.
        "pipeline_fill_ms": pipeline_fill_ms,
        "structurally_unservable_count": n_unservable,
        "steady_total_consumer_invocations": n_steady,
        "steady_output_valid_rate": _rate(lambda r: r.output_valid),
        "steady_stale_input_rate": _rate(
            lambda r: r.invalid_reason in (STALE_INPUT, DEADLINE_AND_STALE)),
        "steady_no_producer_rate": _rate(
            lambda r: r.invalid_reason == NO_COMPLETED_PRODUCER),
    }


# --- workload-spec plumbing ------------------------------------------------
#
# Freshness edges are read from a top-level `freshness_edges` key, deliberately
# SEPARATE from the existing `edges` key.
#
# `edges` already means precedence: workload_factory turns each one into
# `Operation.predecessors`, i.e. a MILP constraint that the consumer may not
# start until the producer has finished. A freshness edge must not do that. A
# real control loop does not block waiting for perception — it reads the most
# recent estimate available and acts on it. That is precisely why its output can
# be stale. Make the same pair a precedence edge and the consumer can never
# consume a stale input; it can only miss its deadline, and the phenomenon under
# study disappears.
#
# So freshness is evaluated post-hoc against the schedule, never enforced
# during scheduling, and the two relations stay visibly distinct in serialized
# specs.

FRESHNESS_EDGES_KEY = "freshness_edges"
PRECEDENCE_EDGES_KEY = "edges"

CRITICALITY_VALUES = ("hard", "soft")


def freshness_edges_from_config(
    networks_data: Dict[str, object],
    *,
    freshness_window_override: Optional[float] = None,
) -> List[FreshnessEdge]:
    """Build FreshnessEdge objects from a top-level workload JSON.

    Expected shape:

        "freshness_edges": [
          {
            "producer_task": "dronet",
            "consumer_task": "mlp_control",
            "freshness_window": 70.5,
            "sample_time_semantics": "producer_release",
            "consumption_policy": "latest_completed",
            "criticality": "hard"
          }
        ]

    `freshness_window_override` replaces every edge's window, which is how the
    phi sweep drives one config across many windows without rewriting it.

    Raises if a freshness edge duplicates a precedence edge on the same pair:
    that combination silently makes staleness impossible (see the note above),
    so it is far more likely to be a mistake than an intent.
    """
    raw = networks_data.get(FRESHNESS_EDGES_KEY, []) or []
    if not isinstance(raw, list):
        raise ValueError(f"{FRESHNESS_EDGES_KEY} must be a list, got {type(raw).__name__}")

    precedence_pairs = {
        (e.get("from"), e.get("to"))
        for e in (networks_data.get(PRECEDENCE_EDGES_KEY, []) or [])
        if isinstance(e, dict)
    }

    edges: List[FreshnessEdge] = []
    for i, spec in enumerate(raw):
        if not isinstance(spec, dict):
            raise ValueError(f"{FRESHNESS_EDGES_KEY}[{i}] must be an object")
        missing = [k for k in ("producer_task", "consumer_task") if not spec.get(k)]
        if missing:
            raise ValueError(
                f"{FRESHNESS_EDGES_KEY}[{i}] missing required field(s): {missing}"
            )
        producer = str(spec["producer_task"])
        consumer = str(spec["consumer_task"])

        if (producer, consumer) in precedence_pairs:
            raise ValueError(
                f"{producer}->{consumer} is declared both as a precedence edge "
                f"(under {PRECEDENCE_EDGES_KEY!r}) and as a freshness edge. A "
                f"precedence edge forces the consumer to wait for the producer, "
                f"which makes a stale input impossible and defeats the "
                f"measurement. Declare it as one or the other."
            )

        window = (
            freshness_window_override
            if freshness_window_override is not None
            else spec.get("freshness_window")
        )
        if window is None:
            raise ValueError(
                f"{FRESHNESS_EDGES_KEY}[{i}] ({producer}->{consumer}) has no "
                f"freshness_window and no override was supplied. There is no "
                f"defensible default: the window must come from the experiment "
                f"config so every result records which one produced it."
            )

        criticality = str(spec.get("criticality", "hard"))
        if criticality not in CRITICALITY_VALUES:
            raise ValueError(
                f"{FRESHNESS_EDGES_KEY}[{i}]: criticality {criticality!r} not in "
                f"{CRITICALITY_VALUES}"
            )

        edges.append(
            FreshnessEdge(
                producer_task=producer,
                consumer_task=consumer,
                freshness_window=float(window),
                sample_time_semantics=str(
                    spec.get("sample_time_semantics", SAMPLE_AT_RELEASE)
                ),
                consumption_policy=str(
                    spec.get("consumption_policy", LATEST_COMPLETED)
                ),
                criticality=criticality,
            )
        )
    return edges


def criticality_from_config(networks_data: Dict[str, object]) -> Dict[str, str]:
    """Map network identifier -> criticality, from a per-network `criticality`.

    Networks that do not declare one default to "soft". Defaulting to soft
    rather than hard is deliberate: a task silently promoted to hard-critical
    would inflate the reported hard-validity denominator, so the safe default
    is the one that cannot flatter the result.
    """
    out: Dict[str, str] = {}
    networks = networks_data.get("networks", {}) or {}
    if isinstance(networks, dict):
        items = networks.items()
    else:
        items = [(n.get("identifier", str(i)), n) for i, n in enumerate(networks)]
    for name, info in items:
        if not isinstance(info, dict):
            continue
        c = str(info.get("criticality", "soft"))
        if c not in CRITICALITY_VALUES:
            raise ValueError(
                f"network {name!r}: criticality {c!r} not in {CRITICALITY_VALUES}"
            )
        out[str(info.get("identifier", name))] = c
    return out


def split_instance_name(name: str, known_tasks: Sequence[str]) -> Tuple[str, int]:
    """Split a per-instance job name into (task, instance).

    workload_factory names periodic instances `<identifier><i>` ("dronet0"),
    while aperiodic networks keep their bare identifier ("yolov8_nano_64" ->
    instance 0).

    Matching is longest-prefix against `known_tasks`, not a trailing-digit
    regex. A digit-suffix rule mis-splits any model whose own name ends in
    digits: "yolov8_nano_64" would become ("yolov8_nano_", 64). Longest-prefix
    also resolves the genuinely ambiguous case in favour of the more specific
    task, so with both "yolov8_nano" and "yolov8_nano_64" registered,
    "yolov8_nano_640" reads as instance 0 of "yolov8_nano_64" rather than
    instance 640 of "yolov8_nano".
    """
    candidates = [t for t in known_tasks if name.startswith(t)]
    if not candidates:
        raise ValueError(
            f"job name {name!r} matches none of the known tasks {list(known_tasks)}"
        )
    for task in sorted(candidates, key=len, reverse=True):
        suffix = name[len(task):]
        if suffix == "":
            return task, 0
        if suffix.isdigit():
            return task, int(suffix)
    raise ValueError(
        f"job name {name!r} starts with a known task but the remainder is not an "
        f"instance index (tried {sorted(candidates, key=len, reverse=True)})"
    )


# --- analytic cross-checks -------------------------------------------------
#
# These exist so a sweep can be checked against closed-form arithmetic before
# trusting it. If the evaluator and the closed form disagree, the evaluator (or
# the trace adapter) is wrong — not the arithmetic.


def analytic_age_supremum(
    producer_period: float,
    producer_latency: float,
    consumer_latency: float,
) -> float:
    """Least upper bound on input_age_at_output with no contention.

    Worst case is a consumer that starts an instant before a producer completes,
    so it must use the previous producer instance:

        age -> producer_period + producer_latency + consumer_latency

    This is a supremum over consumer phase offsets. It is only attained when the
    period grid lets a consumer land arbitrarily close to a producer completion;
    when the consumer period divides the producer period it is NOT reached (see
    analytic_age_ceiling_realized).
    """
    return float(producer_period + producer_latency + consumer_latency)


def analytic_age_ceiling_realized(
    producer_period: float,
    producer_latency: float,
    consumer_period: float,
    consumer_latency: float,
    horizon: float,
    *,
    producer_phase: float = 0.0,
    consumer_phase: float = 0.0,
) -> Dict[str, object]:
    """Exact uncontended age distribution on the actual period grid.

    Builds the zero-contention schedule (every instance starts at its release
    and runs for its latency) and evaluates it, so the result is the same
    quantity the sweep reports rather than a parallel reimplementation.

    Returns the realized max ("A0"), the sorted distinct ages, and the count of
    consumer invocations with no completed producer. A0 is the anchor for the
    freshness-window sweep: choosing phi below A0 measures the producer's
    sampling period rather than contention.
    """
    producers = [
        Invocation(
            task="_producer",
            instance=k,
            release_time=producer_phase + k * producer_period,
            start_time=producer_phase + k * producer_period,
            end_time=producer_phase + k * producer_period + producer_latency,
        )
        for k in range(int((horizon - producer_phase) // producer_period) + 1)
    ]
    consumers = [
        Invocation(
            task="_consumer",
            instance=j,
            release_time=consumer_phase + j * consumer_period,
            start_time=consumer_phase + j * consumer_period,
            end_time=consumer_phase + j * consumer_period + consumer_latency,
            deadline=consumer_phase + (j + 1) * consumer_period,
        )
        for j in range(int((horizon - consumer_phase) // consumer_period) + 1)
    ]
    # A window of +inf would make freshness trivially valid; we only want the
    # ages here, so use the supremum as a nominal window.
    nominal = analytic_age_supremum(
        producer_period, producer_latency, consumer_latency
    ) + 1.0
    ev = evaluate_freshness(
        producers + consumers,
        dependency_edges=[
            FreshnessEdge(
                producer_task="_producer",
                consumer_task="_consumer",
                freshness_window=nominal,
            )
        ],
        experiment_id="analytic_uncontended",
    )
    ages = sorted(
        {
            round(float(r.input_age_at_output), 9)
            for r in ev.records
            if r.input_age_at_output is not None
        }
    )
    return {
        "A0_realized": max(ages) if ages else None,
        "A0_supremum": analytic_age_supremum(
            producer_period, producer_latency, consumer_latency
        ),
        "distinct_ages": ages,
        "no_producer_count": ev.aggregate["no_producer_count"],
        "total_consumer_invocations": ev.aggregate["total_consumer_invocations"],
    }
