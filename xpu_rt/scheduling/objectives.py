"""Multi-objective scoring for the Stage-4 feedback loop.

The feedback loop's solver minimises makespan, but the closed-loop's
real success criterion is multi-dimensional: makespan + deadline
compliance + peak memory + energy + jitter. This module exposes a
user-configurable :class:`MultiObjectiveSpec` plus the post-solve
:func:`compute_metrics` / :func:`evaluate` pair the loop uses to score
each iteration.

Today the multi-objective is evaluated *post-hoc* on the solver's
outputs. CP-SAT still optimises makespan only; the multi-objective
shapes the loop's convergence decision and lets the agent explore the
trade-off surface via :func:`pareto_frontier`.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import StrEnum

import structlog

log = structlog.get_logger(__name__)


class ObjectiveKind(StrEnum):
    """Names of all metrics the multi-objective spec can weight."""

    MAKESPAN = "makespan"
    DEADLINE_VIOLATION_COUNT = "deadline_violation_count"
    DEADLINE_VIOLATION_MS = "deadline_violation_ms"
    PEAK_MEMORY_BYTES = "peak_memory_bytes"
    ENERGY_PROXY = "energy_proxy"
    MAKESPAN_VARIANCE = "makespan_variance"


@dataclass(frozen=True)
class ObjectiveWeight:
    """One objective's weight + optional normaliser.

    Attributes:
        kind: Which metric this weight applies to.
        weight: Non-negative scalar. ``0`` means ignore.
        target_value: Optional baseline used to normalise the raw metric
            before weighting. When ``None``, :func:`evaluate` uses the
            metric's own raw value as the denominator (so an objective
            with no target degenerates to a unit-weighted contribution).
    """

    kind: ObjectiveKind
    weight: float
    target_value: float | None = None


@dataclass(frozen=True)
class MultiObjectiveSpec:
    """User-configurable multi-objective definition.

    Default: makespan-only with weight ``1.0`` — backwards compatible
    with the single-objective loop. Pass a tuple of :class:`ObjectiveWeight`
    to broaden the objective.
    """

    weights: tuple[ObjectiveWeight, ...] = field(
        default_factory=lambda: (ObjectiveWeight(ObjectiveKind.MAKESPAN, 1.0),)
    )

    def normalized(self) -> MultiObjectiveSpec:
        """Return an equivalent spec whose non-zero weights sum to 1.0."""

        total = sum(w.weight for w in self.weights if w.weight > 0)
        if total <= 0:
            return self
        rescaled = tuple(
            ObjectiveWeight(w.kind, w.weight / total, w.target_value)
            for w in self.weights
        )
        return MultiObjectiveSpec(weights=rescaled)

    def active_kinds(self) -> tuple[ObjectiveKind, ...]:
        """Names of objectives with strictly positive weight."""

        return tuple(w.kind for w in self.weights if w.weight > 0)


@dataclass(frozen=True)
class ScheduleMetrics:
    """All metrics observed post-solve.

    Attributes:
        makespan_us: Total wall time of the schedule, in microseconds.
        deadline_violations: Count of partitions whose end time exceeded
            their deadline.
        deadline_violation_total_us: Sum of per-partition overages, in
            microseconds.
        peak_memory_bytes: Peak live bytes across the schedule. ``0`` if
            no buffer information was supplied.
        energy_proxy_joules: ``Σ_d busy_us[d] * power_proxy[d] / 1e6``
            or ``None`` if no power proxies were supplied.
        makespan_variance_us: Sample variance across
            ``measured_makespans_us``; ``0.0`` if fewer than two samples.
    """

    makespan_us: float
    deadline_violations: int
    deadline_violation_total_us: float
    peak_memory_bytes: int
    energy_proxy_joules: float | None
    makespan_variance_us: float


@dataclass(frozen=True)
class ObjectiveScore:
    """One weighted multi-objective scalar plus a per-objective breakdown."""

    score: float
    component_scores: dict[str, float]
    raw_metrics: ScheduleMetrics


def _safe_div(num: float, den: float) -> float:
    if den <= 0:
        return num
    return num / den


def _metric_value(kind: ObjectiveKind, metrics: ScheduleMetrics) -> float:
    if kind is ObjectiveKind.MAKESPAN:
        return float(metrics.makespan_us)
    if kind is ObjectiveKind.DEADLINE_VIOLATION_COUNT:
        return float(metrics.deadline_violations)
    if kind is ObjectiveKind.DEADLINE_VIOLATION_MS:
        return float(metrics.deadline_violation_total_us) / 1000.0
    if kind is ObjectiveKind.PEAK_MEMORY_BYTES:
        return float(metrics.peak_memory_bytes)
    if kind is ObjectiveKind.ENERGY_PROXY:
        return float(metrics.energy_proxy_joules or 0.0)
    if kind is ObjectiveKind.MAKESPAN_VARIANCE:
        return float(metrics.makespan_variance_us)
    raise ValueError(f"unknown ObjectiveKind: {kind}")  # pragma: no cover


def compute_metrics(
    *,
    start_times: dict[str, float],
    end_times: dict[str, float],
    device_assignments: dict[str, int],
    deadlines_us: dict[str, float] | None = None,
    buffer_specs: list | None = None,
    backend_power_proxy_w: dict[str, float] | None = None,
    measured_makespans_us: tuple[float, ...] = (),
) -> ScheduleMetrics:
    """Distil a solved schedule plus side info into :class:`ScheduleMetrics`.

    Args:
        start_times: Partition ``id -> start_us``.
        end_times: Partition ``id -> end_us``.
        device_assignments: Partition ``id -> device_index``.
        deadlines_us: Optional ``id -> deadline_us``. Missing entries are
            ignored (no deadline = never violated).
        buffer_specs: Optional iterable of ``(start_us, end_us, bytes)``
            tuples for peak-memory estimation. ``None`` → ``0`` bytes.
        backend_power_proxy_w: Optional ``device_index -> watts``. The
            energy proxy is ``Σ busy_us[d] * watts[d] / 1e6``. ``None`` →
            ``None`` (consumer treats as unset).
        measured_makespans_us: Repeated end-to-end measurements for
            variance/jitter scoring. Fewer than two samples → 0.0.

    Returns:
        A frozen :class:`ScheduleMetrics`.
    """

    makespan = max(end_times.values()) if end_times else 0.0

    deadline_violations = 0
    deadline_overage_us = 0.0
    if deadlines_us:
        for pid, deadline in deadlines_us.items():
            end = end_times.get(pid)
            if end is None:
                continue
            if end > deadline:
                deadline_violations += 1
                deadline_overage_us += end - deadline

    peak_memory = 0
    if buffer_specs:
        events: list[tuple[float, int]] = []
        for spec in buffer_specs:
            s, e, b = float(spec[0]), float(spec[1]), int(spec[2])
            events.append((s, +b))
            events.append((e, -b))
        events.sort()
        live = 0
        for _, delta in events:
            live += delta
            if live > peak_memory:
                peak_memory = live

    energy: float | None = None
    if backend_power_proxy_w:
        busy_by_dev: dict[int, float] = {}
        for pid, dev in device_assignments.items():
            dur = float(end_times.get(pid, 0.0) - start_times.get(pid, 0.0))
            if dur < 0:
                dur = 0.0
            busy_by_dev[dev] = busy_by_dev.get(dev, 0.0) + dur
        total_j = 0.0
        for dev_idx, busy_us in busy_by_dev.items():
            watts = float(backend_power_proxy_w.get(str(dev_idx), 0.0))
            total_j += busy_us * watts / 1e6
        energy = total_j

    variance = 0.0
    if len(measured_makespans_us) >= 2:
        mean = sum(measured_makespans_us) / len(measured_makespans_us)
        variance = sum((m - mean) ** 2 for m in measured_makespans_us) / (
            len(measured_makespans_us) - 1
        )

    return ScheduleMetrics(
        makespan_us=float(makespan),
        deadline_violations=int(deadline_violations),
        deadline_violation_total_us=float(deadline_overage_us),
        peak_memory_bytes=int(peak_memory),
        energy_proxy_joules=energy,
        makespan_variance_us=float(variance),
    )


def evaluate(spec: MultiObjectiveSpec, metrics: ScheduleMetrics) -> ObjectiveScore:
    """Compute the weighted score (smaller = better).

    Normalisation rule per objective:

    * If the weight has an explicit ``target_value > 0``, the contribution
      is ``weight * raw / target_value``.
    * Else the contribution is ``weight * raw`` (no rescaling — degenerate
      "value is its own scale" case).

    The single-weight makespan-only default returns
    ``score == makespan_us``, preserving backwards compatibility with
    the single-objective loop.
    """

    components: dict[str, float] = {}
    total = 0.0
    for w in spec.weights:
        if w.weight <= 0:
            continue
        raw = _metric_value(w.kind, metrics)
        target = w.target_value
        if target is not None and target > 0:
            contribution = w.weight * raw / target
        else:
            contribution = w.weight * raw
        components[str(w.kind)] = contribution
        total += contribution
    return ObjectiveScore(
        score=total, component_scores=components, raw_metrics=metrics
    )


def pareto_frontier(scores: Iterable[ObjectiveScore]) -> tuple[ObjectiveScore, ...]:
    """Return the non-dominated subset of ``scores``.

    Dominance is computed over the union of metric kinds appearing in
    each score's ``component_scores``. Score A dominates B iff A is
    ``<=`` B on every considered metric and ``<`` B on at least one. Ties
    on all considered metrics keep both points (incomparable).
    """

    pool = tuple(scores)
    if not pool:
        return ()

    kinds: set[str] = set()
    for s in pool:
        kinds.update(s.component_scores.keys())
    kind_list = sorted(kinds)
    if not kind_list:
        return pool

    def _vector(s: ObjectiveScore) -> tuple[float, ...]:
        return tuple(s.component_scores.get(k, 0.0) for k in kind_list)

    frontier: list[ObjectiveScore] = []
    vectors = [_vector(s) for s in pool]
    for i, si in enumerate(pool):
        vi = vectors[i]
        dominated = False
        for j, _sj in enumerate(pool):
            if i == j:
                continue
            vj = vectors[j]
            if all(vj[k] <= vi[k] for k in range(len(kind_list))) and any(
                vj[k] < vi[k] for k in range(len(kind_list))
            ):
                dominated = True
                break
        if not dominated:
            frontier.append(si)
    return tuple(frontier)


__all__ = [
    "MultiObjectiveSpec",
    "ObjectiveKind",
    "ObjectiveScore",
    "ObjectiveWeight",
    "ScheduleMetrics",
    "compute_metrics",
    "evaluate",
    "pareto_frontier",
]
