"""Tests for the N-repeat aggregator."""

from __future__ import annotations

import pytest

from xpu_rt.benchmarks.canonical_metrics import CanonicalCellRow
from xpu_rt.benchmarks.sample_aggregator import CellSummary, aggregate


def _row(**kw) -> CanonicalCellRow:
    base = dict(
        backend="kb-v2",
        target="gemmini_mx",
        workload="smolvla_matmuls",
        shape_id="[64, 720]×[720, 320]",
        repeat=0,
        correctness=True,
        cycles=12000,
        rounds_used=1,
        tokens_in=1000,
        tokens_out=300,
        cost_usd=0.005,
        wall_s=10.0,
        cycle_source="MAIN_LD_ST_EX_CYCLES",
        notes="",
    )
    base.update(kw)
    return CanonicalCellRow(**base)


def test_aggregate_returns_one_summary_per_cell() -> None:
    rows = [
        _row(repeat=0, cycles=12000),
        _row(repeat=1, cycles=12100),
        _row(repeat=2, cycles=11900),
    ]
    summaries = aggregate(rows)
    assert len(summaries) == 1
    s = summaries[0]
    assert s.n_samples == 3
    assert s.n_correct == 3
    assert s.correctness_rate == 1.0
    assert s.median_cycles == 12000
    assert s.min_cycles == 11900
    assert s.max_cycles == 12100


def test_aggregate_skips_failed_samples_for_cycles() -> None:
    """Failed samples (correctness=False) count toward correctness
    rate but must not pollute the cycle stats."""
    rows = [
        _row(repeat=0, correctness=True, cycles=12000),
        _row(repeat=1, correctness=False, cycles=None, cycle_source="none"),
        _row(repeat=2, correctness=True, cycles=12200),
    ]
    summaries = aggregate(rows)
    assert len(summaries) == 1
    s = summaries[0]
    assert s.n_samples == 3
    assert s.n_correct == 2
    assert s.correctness_rate == pytest.approx(2 / 3)
    assert s.median_cycles == 12100  # median of {12000, 12200}
    assert s.min_cycles == 12000


def test_aggregate_groups_by_cell_dimensions() -> None:
    """Rows for different (backend, target, workload, shape) cells
    must end up in different summaries."""
    rows = [
        _row(backend="kb-v2", shape_id="[64, 720]×[720, 320]", repeat=0),
        _row(backend="kb-vanilla", shape_id="[64, 720]×[720, 320]", repeat=0,
             cycle_source="cached"),
        _row(backend="autocomp", shape_id="[64, 720]×[720, 320]", repeat=0,
             cycle_source="Generated implementation latency"),
        _row(backend="kb-v2", shape_id="[64, 32]×[32, 720]", repeat=0,
             cycles=22000),
    ]
    summaries = aggregate(rows)
    assert len(summaries) == 4  # 3 backends on one shape + kb-v2 on another


def test_aggregate_handles_zero_correct_samples() -> None:
    """Cell where every repeat failed → correctness_rate=0,
    cycle stats all None."""
    rows = [
        _row(repeat=0, correctness=False, cycles=None, cycle_source="none"),
        _row(repeat=1, correctness=False, cycles=None, cycle_source="none"),
        _row(repeat=2, correctness=False, cycles=None, cycle_source="none"),
    ]
    summaries = aggregate(rows)
    s = summaries[0]
    assert s.n_correct == 0
    assert s.correctness_rate == 0.0
    assert s.median_cycles is None
    assert s.min_cycles is None
    assert s.max_cycles is None
    assert s.iqr_cycles is None or s.iqr_cycles == 0.0


def test_aggregate_marks_cycle_source_mixed_when_disagree() -> None:
    """If the N samples disagree on cycle_source (e.g. some hit a
    cached fallback, others ran live), surface ``"mixed"`` so the
    report flags it for the operator."""
    rows = [
        _row(repeat=0, cycle_source="MAIN_LD_ST_EX_CYCLES"),
        _row(repeat=1, cycle_source="cached"),
    ]
    summaries = aggregate(rows)
    assert summaries[0].cycle_source == "mixed"


def test_aggregate_totals_cost_and_wall_across_samples() -> None:
    rows = [
        _row(repeat=0, cost_usd=0.01, wall_s=5.0),
        _row(repeat=1, cost_usd=0.012, wall_s=6.0),
        _row(repeat=2, cost_usd=0.011, wall_s=5.5),
    ]
    summaries = aggregate(rows)
    s = summaries[0]
    assert s.total_cost_usd == pytest.approx(0.033)
    assert s.mean_cost_usd == pytest.approx(0.011)
    assert s.total_wall_s == pytest.approx(16.5)
