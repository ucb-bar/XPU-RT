"""Multi-rate dominant-workload analysis for joint scheduling.

Given a cost matrix and a set of workload IDs, this module decides which
workload is dominant (longest min-over-backends critical-path latency)
and computes, for every other workload, the maximum number of cycles
that fit inside one period of the dominant. The output feeds Stage 3's
chunking and Stage 4's joint solve so the joint partition set is sized
from the data instead of hand-tuned.

The model is intentionally a static upper bound:

* The dominant workload is assumed to occupy only its preferred lane.
* Other lanes are treated as fully idle during the dominant's period.
* Real schedules will reduce the multiplicity if specialty chunking
  spills the dominant onto more than one lane.

The honest framing is propagated through :class:`MultiRateAnalysis.notes`
so downstream consumers do not over-interpret the recommendation.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

import structlog

from xpu_rt.scheduling.granularity import compute_specialty_matrix

log = structlog.get_logger(__name__)

_DEFAULT_BACKENDS: tuple[str, ...] = ("CPU", "GPU", "DSP")


@dataclass(frozen=True)
class WorkloadRate:
    """Per-workload rate analysis result.

    Attributes:
        workload_id: e.g. ``"yolov8n"``, ``"dronet"``.
        period_us: Period of one iteration of this workload (best
            single-backend critical path + per-lane overhead).
        multiplicity: How many cycles of this workload fit in one period
            of the dominant. For the dominant itself, ``multiplicity == 1``.
        preferred_lane: Backend chosen for this workload's primary path.
        primary_lane_busy_us: Time the preferred lane is busy with this
            workload per workload-cycle.
        achievable_frequency_hz: ``1e6 / period_us``.
    """

    workload_id: str
    period_us: float
    multiplicity: int
    preferred_lane: str
    primary_lane_busy_us: float
    achievable_frequency_hz: float


@dataclass(frozen=True)
class LaneAvailability:
    """Per-lane idle-time model during the dominant workload's run.

    Attributes:
        lane: Backend identifier (``CPU`` / ``GPU`` / ``DSP``).
        busy_us_per_dominant_period: Time this lane is busy with the
            dominant workload over one dominant period.
        idle_us_per_dominant_period: Remaining time per dominant period
            (``dominant_period_us - busy``).
        busy_fraction: ``busy / dominant_period_us`` in ``[0, 1]``.
    """

    lane: str
    busy_us_per_dominant_period: float
    idle_us_per_dominant_period: float
    busy_fraction: float


@dataclass(frozen=True)
class MultiRateAnalysis:
    """Full multi-rate analysis output.

    Attributes:
        dominant_workload_id: Longest critical path.
        dominant_period_us: The full period.
        rates: Per-workload rate result (includes the dominant with
            multiplicity ``1``).
        lane_availability: Per-backend idle time during the dominant
            period.
        notes: Human-readable rationale lines (including the static
            upper-bound caveats).
    """

    dominant_workload_id: str
    dominant_period_us: float
    rates: tuple[WorkloadRate, ...]
    lane_availability: tuple[LaneAvailability, ...]
    notes: tuple[str, ...]


def _per_backend_sum(
    cost_matrix: dict,
    workload_id: str,
    backend: str,
) -> float | None:
    """Sum per-op costs for ``workload_id`` on ``backend``.

    Returns ``None`` if ``backend`` is not measured for at least one op
    in the workload (treated as infeasible at workload granularity).
    """

    workload = cost_matrix.get(workload_id)
    if not isinstance(workload, dict) or not workload:
        return None
    total = 0.0
    covered = 0
    for _op_id, costs in workload.items():
        if not isinstance(costs, dict):
            continue
        if backend not in costs or costs[backend] is None:
            continue
        try:
            value = float(costs[backend])
        except (TypeError, ValueError):
            continue
        if math.isinf(value) or math.isnan(value):
            continue
        total += value
        covered += 1
    if covered == 0:
        return None
    return total


def _best_backend_period(
    cost_matrix: dict,
    workload_id: str,
    calibration_overhead_us: dict[str, float] | None,
    backends: tuple[str, ...] = _DEFAULT_BACKENDS,
) -> tuple[str, float, float]:
    """Pick the cheapest backend for the whole workload.

    Returns ``(backend, period_us, per_op_sum_us)``. ``period_us``
    includes a single per-lane dispatch overhead (when calibration is
    supplied). Returns ``(_, inf, _)`` if no backend is feasible.
    """

    best_backend = ""
    best_period = math.inf
    best_sum = math.inf
    for backend in backends:
        per_op_sum = _per_backend_sum(cost_matrix, workload_id, backend)
        if per_op_sum is None:
            continue
        overhead = 0.0
        if calibration_overhead_us is not None:
            overhead = float(calibration_overhead_us.get(backend, 0.0))
        period = per_op_sum + overhead
        if period < best_period:
            best_period = period
            best_backend = backend
            best_sum = per_op_sum
    return (best_backend, best_period, best_sum)


def identify_dominant_workload(
    workload_ids: Iterable[str],
    cost_matrix: dict,
    calibration_overhead_us: dict[str, float] | None = None,
) -> tuple[str, dict[str, float]]:
    """Pick the workload with the largest min-over-backends critical-path latency.

    Args:
        workload_ids: Workloads to compare.
        cost_matrix: ``{workload_id: {op_id: {backend: us}}}``.
        calibration_overhead_us: Optional per-lane dispatch overhead
            added once per workload.

    Returns:
        ``(dominant_id, {workload_id: best_period_us})``. Periods that
        could not be computed for any backend appear as ``math.inf``.

    Raises:
        ValueError: If ``workload_ids`` is empty or none of the workloads
            have at least one feasible backend.
    """

    ids = [w for w in workload_ids]
    if not ids:
        raise ValueError("workload_ids must contain at least one entry")

    periods: dict[str, float] = {}
    for w in ids:
        _, period, _ = _best_backend_period(
            cost_matrix, w, calibration_overhead_us
        )
        periods[w] = period

    finite = {w: p for w, p in periods.items() if math.isfinite(p)}
    if not finite:
        raise ValueError(
            f"no feasible backend found for any workload in {ids!r}"
        )

    # Tie-break deterministically: largest period wins, then lex-greatest
    # workload id, so the dominant decision is reproducible across runs.
    dominant = max(finite.items(), key=lambda kv: (kv[1], kv[0]))[0]
    return (dominant, periods)


def estimate_lane_availability(
    dominant_workload_id: str,
    dominant_preferred_lane: str,
    dominant_period_us: float,
    cost_matrix: dict,
    calibration_overhead_us: dict[str, float] | None = None,
    backends: tuple[str, ...] = _DEFAULT_BACKENDS,
) -> tuple[LaneAvailability, ...]:
    """Build the per-lane busy/idle breakdown during the dominant's period.

    Optimistic single-lane model: the dominant workload runs only on
    ``dominant_preferred_lane``; every other lane is reported fully idle.
    Real chunked schedules may spill onto other lanes — when they do,
    the secondary multiplicity computed downstream becomes a true upper
    bound, not an exact ceiling.
    """

    if dominant_period_us <= 0 or not math.isfinite(dominant_period_us):
        return tuple(
            LaneAvailability(
                lane=lane,
                busy_us_per_dominant_period=0.0,
                idle_us_per_dominant_period=0.0,
                busy_fraction=0.0,
            )
            for lane in backends
        )

    per_op_sum = _per_backend_sum(
        cost_matrix, dominant_workload_id, dominant_preferred_lane
    )
    overhead = 0.0
    if calibration_overhead_us is not None:
        overhead = float(
            calibration_overhead_us.get(dominant_preferred_lane, 0.0)
        )
    busy_primary = (per_op_sum or 0.0) + overhead
    # Cap at the full period so floating-point drift can never produce a
    # negative idle window.
    busy_primary = min(busy_primary, dominant_period_us)

    out: list[LaneAvailability] = []
    for lane in backends:
        if lane == dominant_preferred_lane:
            busy = busy_primary
        else:
            busy = 0.0
        idle = max(dominant_period_us - busy, 0.0)
        fraction = busy / dominant_period_us if dominant_period_us > 0 else 0.0
        out.append(
            LaneAvailability(
                lane=lane,
                busy_us_per_dominant_period=busy,
                idle_us_per_dominant_period=idle,
                busy_fraction=fraction,
            )
        )
    return tuple(out)


def compute_multiplicity(
    secondary_workload_id: str,
    dominant_period_us: float,
    lane_availability: tuple[LaneAvailability, ...],
    cost_matrix: dict,
    calibration_overhead_us: dict[str, float] | None = None,
) -> tuple[int, str, float]:
    """Compute the maximum multiplicity + preferred lane for a secondary workload.

    Algorithm:
        1. For each lane with positive idle time, compute
           ``cost_per_cycle[lane] = per_op_sum[secondary][lane] + overhead[lane]``.
        2. ``max_fits[lane] = floor(idle_us[lane] / cost_per_cycle[lane])``.
        3. Pick the lane with the largest ``max_fits``. Ties break by the
           secondary workload's specialty argmin (the lane its op-family
           specialty matrix favours), then by lane name.

    Returns:
        ``(multiplicity, preferred_lane, cost_per_cycle_us)``.
        ``multiplicity == 0`` is returned when no lane fits even one
        cycle of the secondary; the returned lane is then the cheapest
        lane regardless of fit (so callers can still report a sensible
        recommendation).
    """

    specialty_argmin: str | None = None
    try:
        specialty = compute_specialty_matrix(cost_matrix, secondary_workload_id)
        # Vote across families for a single preferred lane.
        if specialty:
            counts: dict[str, int] = {}
            for lane in specialty.values():
                counts[lane] = counts.get(lane, 0) + 1
            specialty_argmin = max(
                counts.items(), key=lambda kv: (kv[1], kv[0])
            )[0]
    except KeyError:
        specialty_argmin = None

    candidates: list[tuple[int, str, float]] = []
    # Fallback "cheapest lane" is computed only over lanes with positive
    # idle time, so a lane fully occupied by the dominant never shadows
    # the recommendation when nothing fits.
    cheapest_lane = ""
    cheapest_cost = math.inf
    for la in lane_availability:
        per_op_sum = _per_backend_sum(
            cost_matrix, secondary_workload_id, la.lane
        )
        if per_op_sum is None:
            continue
        overhead = 0.0
        if calibration_overhead_us is not None:
            overhead = float(calibration_overhead_us.get(la.lane, 0.0))
        cost_per_cycle = per_op_sum + overhead
        if (
            la.idle_us_per_dominant_period > 0
            and cost_per_cycle > 0
            and cost_per_cycle < cheapest_cost
        ):
            cheapest_cost = cost_per_cycle
            cheapest_lane = la.lane
        if cost_per_cycle <= 0 or la.idle_us_per_dominant_period <= 0:
            continue
        fits = int(math.floor(la.idle_us_per_dominant_period / cost_per_cycle))
        if fits <= 0:
            continue
        candidates.append((fits, la.lane, cost_per_cycle))

    if not candidates:
        return (0, cheapest_lane, cheapest_cost if math.isfinite(cheapest_cost) else 0.0)

    def _rank(c: tuple[int, str, float]) -> tuple[int, int, str]:
        fits, lane, _ = c
        specialty_bonus = 1 if specialty_argmin and lane == specialty_argmin else 0
        # Higher fits first, then specialty preference, then deterministic
        # lane-name tie-break (CPU < DSP < GPU lex order).
        return (fits, specialty_bonus, lane)

    candidates.sort(key=_rank, reverse=True)
    best = candidates[0]
    return (int(best[0]), str(best[1]), float(best[2]))


def analyze(
    workload_ids: Iterable[str],
    cost_matrix: dict,
    calibration_overhead_us: dict[str, float] | None = None,
) -> MultiRateAnalysis:
    """End-to-end multi-rate analysis.

    Identifies the dominant workload, estimates per-lane availability
    during its period, and computes a multiplicity + preferred lane for
    every non-dominant workload.

    Args:
        workload_ids: Iterable of workload keys (must all appear in
            ``cost_matrix``).
        cost_matrix: Per-workload, per-op, per-backend cost dict.
        calibration_overhead_us: Optional per-lane dispatch overhead
            added once per workload-cycle.

    Returns:
        A frozen :class:`MultiRateAnalysis`.
    """

    ids = list(workload_ids)
    dominant_id, periods = identify_dominant_workload(
        ids, cost_matrix, calibration_overhead_us
    )
    dominant_period = periods[dominant_id]
    dominant_lane, _, _ = _best_backend_period(
        cost_matrix, dominant_id, calibration_overhead_us
    )
    lane_avail = estimate_lane_availability(
        dominant_workload_id=dominant_id,
        dominant_preferred_lane=dominant_lane,
        dominant_period_us=dominant_period,
        cost_matrix=cost_matrix,
        calibration_overhead_us=calibration_overhead_us,
    )

    rates: list[WorkloadRate] = []
    notes: list[str] = [
        "Static upper-bound model: assumes dominant occupies only its preferred lane.",
        "Real schedules may reduce multiplicity if specialty chunking spills onto other lanes.",
        "Without a board run, the recommendation is a planning estimate, not a measured ceiling.",
    ]

    for wid in ids:
        if wid == dominant_id:
            primary_busy = 0.0
            for la in lane_avail:
                if la.lane == dominant_lane:
                    primary_busy = la.busy_us_per_dominant_period
                    break
            freq = (1e6 / dominant_period) if dominant_period > 0 else 0.0
            rates.append(
                WorkloadRate(
                    workload_id=wid,
                    period_us=dominant_period,
                    multiplicity=1,
                    preferred_lane=dominant_lane,
                    primary_lane_busy_us=primary_busy,
                    achievable_frequency_hz=freq,
                )
            )
            continue

        secondary_period = periods.get(wid, math.inf)
        mult, lane, cost_per_cycle = compute_multiplicity(
            secondary_workload_id=wid,
            dominant_period_us=dominant_period,
            lane_availability=lane_avail,
            cost_matrix=cost_matrix,
            calibration_overhead_us=calibration_overhead_us,
        )
        # Per-cycle frequency: how often a single instance fires when N fit
        # in one dominant period; if N == 0 fall back to the secondary's
        # solo period for a meaningful Hz figure.
        if mult > 0 and dominant_period > 0:
            per_instance_period = dominant_period / mult
            freq = 1e6 / per_instance_period
        elif math.isfinite(secondary_period) and secondary_period > 0:
            freq = 1e6 / secondary_period
        else:
            freq = 0.0
        if mult == 0:
            notes.append(
                f"workload {wid!r} does not fit even once in the dominant's "
                f"period on any lane (cost/cycle={cost_per_cycle:.0f} us)."
            )
        rates.append(
            WorkloadRate(
                workload_id=wid,
                period_us=secondary_period,
                multiplicity=mult,
                preferred_lane=lane,
                primary_lane_busy_us=cost_per_cycle,
                achievable_frequency_hz=freq,
            )
        )

    log.info(
        "multi_rate_analyze",
        dominant=dominant_id,
        dominant_period_us=dominant_period,
        n_workloads=len(rates),
    )
    return MultiRateAnalysis(
        dominant_workload_id=dominant_id,
        dominant_period_us=dominant_period,
        rates=tuple(rates),
        lane_availability=lane_avail,
        notes=tuple(notes),
    )


__all__ = [
    "LaneAvailability",
    "MultiRateAnalysis",
    "WorkloadRate",
    "analyze",
    "compute_multiplicity",
    "estimate_lane_availability",
    "identify_dominant_workload",
]
