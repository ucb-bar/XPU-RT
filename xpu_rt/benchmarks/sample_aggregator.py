"""Aggregate N :class:`CanonicalCellRow` samples per cell.

Phase-D writes one row per (backend, target, workload, shape, repeat)
to a per-cell JSONL. The aggregator collapses the N samples for each
(backend, target, workload, shape) into a single :class:`CellSummary`
with median + min/max + IQR over the correct subset. The final
report quotes these summaries with explicit ±range so the reader
sees variance.

Missing-cycle samples (correctness=False) are skipped in the cycle
aggregation but counted toward the correctness rate.
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable

from xpu_rt.benchmarks.canonical_metrics import CanonicalCellRow


@dataclass(frozen=True)
class CellSummary:
    """Aggregate over the N samples of one cell."""

    backend: str
    target: str
    workload: str
    shape_id: str
    n_samples: int
    n_correct: int
    correctness_rate: float
    median_cycles: float | None
    min_cycles: int | None
    max_cycles: int | None
    iqr_cycles: float | None
    median_rounds: float
    mean_cost_usd: float
    total_cost_usd: float
    total_wall_s: float
    cycle_source: str  # most common across samples; "mixed" if disagreement


def aggregate(rows: Iterable[CanonicalCellRow]) -> list[CellSummary]:
    """Group rows by (backend, target, workload, shape_id) and emit
    one :class:`CellSummary` per group. Rows from different repeats
    of the same cell collapse into one summary."""
    grouped: dict[tuple[str, str, str, str], list[CanonicalCellRow]] = defaultdict(list)
    for row in rows:
        grouped[(row.backend, row.target, row.workload, row.shape_id)].append(row)

    summaries: list[CellSummary] = []
    for key, samples in sorted(grouped.items()):
        summaries.append(_summarise(samples))
    return summaries


def _summarise(samples: list[CanonicalCellRow]) -> CellSummary:
    n = len(samples)
    correct = [s for s in samples if s.correctness and s.cycles is not None]
    n_correct = len(correct)
    cycles = sorted(s.cycles for s in correct if s.cycles is not None)
    median_c = float(statistics.median(cycles)) if cycles else None
    min_c = int(cycles[0]) if cycles else None
    max_c = int(cycles[-1]) if cycles else None
    iqr_c = (
        float(statistics.quantiles(cycles, n=4)[2] - statistics.quantiles(cycles, n=4)[0])
        if len(cycles) >= 4 else (max_c - min_c if cycles and len(cycles) > 1 else 0.0)
    )
    rounds_seq = [float(s.rounds_used) for s in samples]
    median_rounds = float(statistics.median(rounds_seq)) if rounds_seq else 0.0
    total_cost = sum(s.cost_usd for s in samples)
    mean_cost = total_cost / n if n else 0.0
    total_wall = sum(s.wall_s for s in samples)
    # Pick the most-common cycle_source. If multiple differ, label "mixed".
    sources = [s.cycle_source for s in samples]
    if not sources:
        cycle_source = "none"
    else:
        counts: dict[str, int] = defaultdict(int)
        for src in sources:
            counts[src] += 1
        top = max(counts.items(), key=lambda kv: kv[1])
        # When the top hit is shared (all-same), use it; otherwise "mixed".
        cycle_source = top[0] if top[1] == len(sources) else "mixed"

    first = samples[0]
    return CellSummary(
        backend=first.backend,
        target=first.target,
        workload=first.workload,
        shape_id=first.shape_id,
        n_samples=n,
        n_correct=n_correct,
        correctness_rate=n_correct / n if n else 0.0,
        median_cycles=median_c,
        min_cycles=min_c,
        max_cycles=max_c,
        iqr_cycles=iqr_c,
        median_rounds=median_rounds,
        mean_cost_usd=mean_cost,
        total_cost_usd=total_cost,
        total_wall_s=total_wall,
        cycle_source=cycle_source,
    )


__all__ = ["CellSummary", "aggregate"]
