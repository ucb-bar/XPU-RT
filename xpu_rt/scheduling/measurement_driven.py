"""Measurement-first decision layer for the feedback loop.

The model-based predictor (v4 calibration + CP-SAT/MOSEK/greedy) remains
the prior. When a real measurement exists for a candidate schedule
configuration, this module returns the measurement instead. The loop's
convergence rules consume the resulting value unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import structlog

from xpu_rt.runtime.measurement_cache import (
    CacheKey,
    MeasurementCache,
)

log = structlog.get_logger(__name__)


@dataclass(frozen=True)
class CandidateSchedule:
    """One configuration the loop is evaluating.

    Multi-workload schedules are expressed as a tuple of (workload, lane,
    period_us, deployment_techniques) — one per dispatchable instance.
    Each placement's techniques tuple is expected to already be
    canonical-sorted (matching ``CacheKey.make`` semantics).
    """

    target_id: str
    placements: tuple[tuple[str, str, int, tuple[str, ...]], ...]


@dataclass(frozen=True)
class MeasurementResolution:
    """Result of resolving a candidate against the measurement cache."""

    candidate: CandidateSchedule
    fully_measured: bool
    per_placement_p50_us: tuple[float | None, ...]
    aggregate_makespan_us: float | None
    sources: tuple[str | None, ...]


def resolve_candidate(
    candidate: CandidateSchedule,
    cache: MeasurementCache,
) -> MeasurementResolution:
    """Look up every placement; if all hit, return aggregated stats.

    Aggregation rule for multi-workload schedules: makespan is the max
    over per-placement p50 latencies. This treats each placement as if
    it runs on its own lane in parallel — true for the bundle's
    yolov8n@DSP || dronet@GPU setup where the lanes are disjoint.
    Future multi-placement work on shared lanes would need contention
    folding here; flagged as a follow-up.
    """

    per_placement_p50: list[float | None] = []
    sources: list[str | None] = []
    any_miss = False
    for workload_id, lane, period_us, techniques in candidate.placements:
        key = CacheKey.make(
            target_id=candidate.target_id,
            workload_id=workload_id,
            lane=lane,
            techniques=techniques,
            period_us=period_us,
        )
        entry = cache.get(key)
        if entry is None:
            # Fall back 1: same lane, technique-superset match.
            entry = cache.get_best(
                target_id=candidate.target_id,
                workload_id=workload_id,
                lane=lane,
                prefer_techniques=tuple(techniques),
            )
        if entry is None:
            # Fall back 2: any lane for the workload (technique-superset
            # preferred). Justified: when the bundle measured the
            # workload on a different lane than the planner chose, the
            # bundle's number is still the only real wall-time we have;
            # treating it as ground truth beats the model-only prior.
            entry = cache.get_best(
                target_id=candidate.target_id,
                workload_id=workload_id,
                lane=None,
                prefer_techniques=tuple(techniques),
            )
        if entry is None:
            per_placement_p50.append(None)
            sources.append(None)
            any_miss = True
            continue
        per_placement_p50.append(entry.stats.p50_us)
        sources.append(entry.stats.source)
    aggregate: float | None
    if any_miss:
        aggregate = None
    else:
        finite = [p for p in per_placement_p50 if p is not None]
        aggregate = max(finite) if finite else 0.0
    return MeasurementResolution(
        candidate=candidate,
        fully_measured=not any_miss,
        per_placement_p50_us=tuple(per_placement_p50),
        aggregate_makespan_us=aggregate,
        sources=tuple(sources),
    )


def evaluate_with_cache(
    candidate: CandidateSchedule,
    cache: MeasurementCache,
    *,
    predicted_makespan_fallback_us: float | None = None,
    deadline_us_per_workload: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Return a typed evaluation dict consumed by the loop.

    Shape:
        {
          "source": "measurement_cache" | "predicted" | "mixed",
          "makespan_us": float,
          "deadlines_met": bool,
          "per_placement": [{workload, lane, p50_us, source}, ...],
          "missing": [(workload, lane, techniques), ...],
        }

    When ``source == "measurement_cache"`` the loop should treat this as
    ground truth and bypass the predictor's regression-guard math entirely.
    """

    resolution = resolve_candidate(candidate, cache)
    per_placement = []
    missing: list[tuple[str, str, tuple[str, ...]]] = []
    for (workload_id, lane, _period_us, techniques), p50, src in zip(
        candidate.placements,
        resolution.per_placement_p50_us,
        resolution.sources,
        strict=True,
    ):
        per_placement.append({
            "workload": workload_id,
            "lane": lane,
            "p50_us": p50,
            "source": src,
        })
        if p50 is None:
            missing.append((workload_id, lane, tuple(techniques)))

    deadlines = deadline_us_per_workload or {}
    deadlines_met = True
    if resolution.fully_measured and resolution.aggregate_makespan_us is not None:
        # Per-placement check: each placement's p50 must satisfy its
        # workload's deadline. If a deadline is absent for a workload,
        # treat it as unconstrained.
        for entry in per_placement:
            wl = entry["workload"]
            p50 = entry["p50_us"]
            if wl in deadlines and p50 is not None and p50 > deadlines[wl]:
                deadlines_met = False
                break

    if resolution.fully_measured:
        source = "measurement_cache"
        makespan = float(resolution.aggregate_makespan_us or 0.0)
    elif all(p is None for p in resolution.per_placement_p50_us):
        source = "predicted"
        makespan = (
            float(predicted_makespan_fallback_us)
            if predicted_makespan_fallback_us is not None
            else float("inf")
        )
        deadlines_met = True
    else:
        source = "mixed"
        # Conservative aggregate: take the max over (measured p50s) ∪
        # {fallback}. Without per-placement predictor breakdown the
        # fallback is the best single number we have for the missing
        # placements, so it dominates.
        measured = [p for p in resolution.per_placement_p50_us if p is not None]
        if predicted_makespan_fallback_us is not None:
            measured.append(float(predicted_makespan_fallback_us))
        makespan = max(measured) if measured else float("inf")
        deadlines_met = True

    return {
        "source": source,
        "makespan_us": makespan,
        "deadlines_met": deadlines_met,
        "per_placement": per_placement,
        "missing": missing,
    }
