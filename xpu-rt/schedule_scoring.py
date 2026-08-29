"""Turn a solved schedule into a `CandidateOutcome`, once.

Scoring a schedule is four steps that must not vary between callers: render the
schedule as trace rows, summarise them against the periods and windows the
workload declares, hand the summary to `candidate_objective`, and patch in the
heavy model when it has no period of its own.

WHY THIS IS A MODULE. Those four steps had three implementations --
`profile_schedulers.score` (the sweep), `compare_candidates.score` (the
verdict) and `plot_loop_iterations.score` (the figure) -- and they had already
drifted. Only the first applied the heavy-model fallback, so a schedule whose
heavy net is non-periodic scored one way in the sweep and a different way in
the verdict that decides whether to keep it. Two of the three also predated
`periods_ms(schedule, known)`, so they could not repair a schedule written
before network names stopped being digit-stripped.

A comparison is only meaningful if both sides were scored identically. That is
easiest to guarantee by there being one scorer.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import candidate_objective as objective
import schedule_trace
import trace_metrics


def heavy_stats(rows: Sequence[dict], model: str) -> Tuple[float, float]:
    """`(max instance latency ms, completion rate Hz)` for an aperiodic model.

    `trace_metrics` reports only models it was given a period for, which is
    right -- but the objective's heavy-model terms also apply to a
    non-periodic background net (a one-shot YOLO pass). Latency is then the
    instance's own span, not a response measured from a release that does not
    exist.
    """
    spans: Dict[str, List[float]] = defaultdict(lambda: [1e18, -1e18])
    for r in rows:
        if trace_metrics.model_of(r["job_name"]) != model:
            continue
        j = r["job_name"]
        spans[j][0] = min(spans[j][0], float(r["start_us"]) / 1000.0)
        spans[j][1] = max(spans[j][1], float(r["end_us"]) / 1000.0)
    if not spans:
        return 0.0, 0.0
    latencies = [en - st for st, en in spans.values()]
    last_end = max(en for _, en in spans.values())
    hz = (len(spans) / (last_end / 1000.0)) if last_end > 0 else 0.0
    return max(latencies), hz


def score(name: str, schedule: dict,
          windows_ms: Optional[Dict[str, float]] = None,
          critical: Sequence[str] = (),
          heavy: Optional[str] = None,
          known: Optional[Sequence[str]] = None,
          ) -> Tuple[dict, objective.CandidateOutcome, List[dict]]:
    """`(summary, outcome, rows)` for one solved schedule.

    `windows_ms` is the workload's declared `window_duration` per network, and
    it is the DEADLINE -- `trace_metrics` falls back to the period without it,
    which is a different and more forgiving test.

    `known` is the set of real network names. It repairs a schedule written
    before `metadata.periodic_networks` stopped being digit-stripped, where
    `yolov8_nano_64x96` was recorded as `yolov8_nano_64x` and instance 0 then
    read as instance 960 -- making that model's deadline 48 seconds and its
    miss count a structural zero.
    """
    windows_ms = windows_ms or {}
    rows = schedule_trace.trace_rows_from_schedule(schedule)
    periods = schedule_trace.periods_ms(schedule, known)
    summary = trace_metrics.summarise_trace(
        rows, periods, {k: v for k, v in windows_ms.items() if k in periods})
    out = objective.from_trace_summary(
        name, summary, critical_models=tuple(critical), heavy_model=heavy,
        standalone_cycles=int(round(
            schedule_trace.standalone_service_us(schedule))))
    if heavy and heavy not in out.per_model:
        out.heavy_max_latency_ms, out.heavy_throughput_hz = heavy_stats(rows, heavy)
    return summary, out, rows
