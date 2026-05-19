"""Tests for the multi-rate dominant-workload analysis."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
from xpu_rt.scheduler.qnn_real_workload import load_cost_matrix
from xpu_rt.scheduling.multi_rate import (
    LaneAvailability,
    analyze,
    compute_multiplicity,
    estimate_lane_availability,
    identify_dominant_workload,
)

COST_MATRIX_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "profiled"
    / "qnn_cost_matrix.json"
)


@pytest.fixture(scope="module")
def real_cost_matrix() -> dict:
    return load_cost_matrix(COST_MATRIX_PATH)


def test_dominant_is_yolov8n_in_real_data(real_cost_matrix: dict) -> None:
    result = analyze(["yolov8n", "dronet"], real_cost_matrix)
    assert result.dominant_workload_id == "yolov8n"
    assert result.dominant_period_us > 0


def test_dronet_multiplicity_at_least_12(real_cost_matrix: dict) -> None:
    # The closed-loop's hardcoded value (12) should be the lower bound
    # for the static analysis: with no contention beyond the dominant's
    # preferred lane, more dronet cycles fit than the human picked.
    result = analyze(["yolov8n", "dronet"], real_cost_matrix)
    dronet_rate = next(r for r in result.rates if r.workload_id == "dronet")
    assert dronet_rate.multiplicity >= 12, (
        f"dronet multiplicity {dronet_rate.multiplicity} below "
        f"closed-loop's hand-tuned 12"
    )


def test_dronet_multiplicity_uses_cpu_when_dominant_on_dsp(
    real_cost_matrix: dict,
) -> None:
    # Force yolov8n onto DSP (zero DSP overhead) and inflate GPU dispatch
    # so dronet prefers CPU over GPU. DSP is busy with yolov8n; the
    # static analysis must route dronet onto a free lane and pick CPU
    # over GPU because GPU's dispatch overhead dwarfs its per-op savings.
    overhead = {"CPU": 5_000.0, "GPU": 200_000.0, "DSP": 0.0}
    result = analyze(
        ["yolov8n", "dronet"], real_cost_matrix, calibration_overhead_us=overhead
    )
    yolo_rate = next(r for r in result.rates if r.workload_id == "yolov8n")
    assert yolo_rate.preferred_lane == "DSP"
    dronet_rate = next(r for r in result.rates if r.workload_id == "dronet")
    # DSP is fully occupied by yolov8n (idle == 0); the recommendation
    # must land on a non-DSP lane.
    assert dronet_rate.preferred_lane != "DSP"
    assert dronet_rate.preferred_lane == "CPU"


def test_lane_availability_sums_correctly(real_cost_matrix: dict) -> None:
    result = analyze(["yolov8n", "dronet"], real_cost_matrix)
    for la in result.lane_availability:
        total = la.busy_us_per_dominant_period + la.idle_us_per_dominant_period
        assert math.isclose(total, result.dominant_period_us, rel_tol=1e-9)


def test_synthetic_three_workload_picks_longest_as_dominant() -> None:
    # Three synthetic workloads with target periods 100, 50, 200.
    # We use a single backend "X" so per-op sums equal the period.
    cost_matrix = {
        "small": {f"op_{i}": {"CPU": 5.0} for i in range(10)},  # 50 us
        "med": {f"op_{i}": {"CPU": 10.0} for i in range(10)},  # 100 us
        "big": {f"op_{i}": {"CPU": 20.0} for i in range(10)},  # 200 us
    }
    dominant_id, periods = identify_dominant_workload(
        ["small", "med", "big"], cost_matrix
    )
    assert dominant_id == "big"
    assert math.isclose(periods["big"], 200.0)
    assert math.isclose(periods["med"], 100.0)
    assert math.isclose(periods["small"], 50.0)


def test_multiplicity_zero_when_secondary_too_big() -> None:
    # Single-lane synthetic: dominant fully occupies CPU (the only
    # feasible backend), and the secondary is also CPU-only. The
    # dominant's preferred-lane busy fraction is 100% so there is
    # literally no room — multiplicity should be 0 without crashing.
    cost_matrix = {
        "dom": {f"op_{i}": {"CPU": 100.0} for i in range(10)},  # 1000 us on CPU
        "small": {f"op_{i}": {"CPU": 10.0} for i in range(10)},  # 100 us on CPU
    }
    result = analyze(["dom", "small"], cost_matrix)
    assert result.dominant_workload_id == "dom"
    small_rate = next(r for r in result.rates if r.workload_id == "small")
    assert small_rate.multiplicity == 0
    # Notes should mention the no-fit case.
    assert any("does not fit" in n for n in result.notes)


def test_compute_multiplicity_returns_zero_with_idle_lanes_when_too_costly() -> None:
    # Direct unit test of compute_multiplicity with a cooked
    # LaneAvailability tuple. cost_per_cycle=200 us across the board,
    # but every lane has idle <= 100 us — nothing fits.
    cost_matrix = {
        "tiny": {"op_0": {"CPU": 200.0, "GPU": 200.0, "DSP": 200.0}},
    }
    lanes = (
        LaneAvailability("CPU", 50.0, 50.0, 0.5),
        LaneAvailability("GPU", 0.0, 100.0, 0.0),
        LaneAvailability("DSP", 0.0, 100.0, 0.0),
    )
    mult, lane, _ = compute_multiplicity(
        secondary_workload_id="tiny",
        dominant_period_us=100.0,
        lane_availability=lanes,
        cost_matrix=cost_matrix,
    )
    assert mult == 0
    assert lane in ("CPU", "GPU", "DSP")


def test_estimate_lane_availability_handles_zero_period() -> None:
    cost_matrix = {"x": {"op_0": {"CPU": 1.0}}}
    out = estimate_lane_availability(
        dominant_workload_id="x",
        dominant_preferred_lane="CPU",
        dominant_period_us=0.0,
        cost_matrix=cost_matrix,
    )
    assert all(la.busy_fraction == 0.0 for la in out)
