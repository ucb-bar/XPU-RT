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
CONSUMPTION_POLICIES = (LATEST_COMPLETED,)


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


def select_producer(
    producers: Sequence[Invocation],
    consumer_start_time: float,
    policy: str = LATEST_COMPLETED,
) -> Optional[Invocation]:
    """Return the producer instance a consumer starting at `consumer_start_time`
    consumes, or None if none had completed.

    `latest_completed`: the eligible producer with the greatest end_time, where
    eligible means `end_time <= consumer_start_time`.

    Two things this is deliberately NOT:

    * not the most recently *released* producer. With heterogeneous backends
      producers can complete out of release order (a perception instance placed
      on the slow backend can finish after a later instance placed on the fast
      one), and the consumer physically cannot read a result that has not been
      written yet.
    * not the producer with the greatest instance index.

    The boundary is inclusive: a producer finishing exactly at the consumer's
    start time IS eligible. Zero-duration gaps are common in solver output
    (equality is the normal case, not a measure-zero edge case), and excluding
    it would silently reclassify a whole class of tight schedules as
    no_completed_producer. Ties in end_time break toward the higher instance
    index, which is the fresher sample.
    """
    if policy != LATEST_COMPLETED:
        raise ValueError(
            f"unknown consumption_policy {policy!r}; expected one of "
            f"{CONSUMPTION_POLICIES}"
        )
    best: Optional[Invocation] = None
    for p in producers:
        if p.end_time > consumer_start_time:  # inclusive boundary
            continue
        if best is None:
            best = p
        elif (p.end_time, p.instance) > (best.end_time, best.instance):
            best = p
    return best


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
            p = select_producer(producers, c.start_time, eff_policy)

            deadline_valid = c.deadline is None or c.end_time <= c.deadline

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

            freshness_valid = age_at_output <= edge.freshness_window
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
        aggregate=aggregate_metrics(records),
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


def aggregate_metrics(records: Sequence[FreshnessRecord]) -> Dict[str, object]:
    """Aggregate rates and age percentiles over per-invocation records.

    Age percentiles are computed over records that HAVE an age, i.e. excluding
    no_completed_producer. Those are counted separately rather than folded in
    as an infinite age, because a missing producer and a very old producer are
    different failures with different fixes, and imputing a value for the
    former would quietly move the percentiles.
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
    }


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
