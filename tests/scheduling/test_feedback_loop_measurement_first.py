"""Measurement-first short-circuit in the feedback loop."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from xpu_rt.runtime.calibration import (
    CALIBRATION_SCHEMA_VERSION,
    WARM_TECHNIQUES,
    CalibrationModel,
)
from xpu_rt.runtime.measurement_cache import (
    CacheEntry,
    CacheKey,
    MeasuredStats,
    append_entry,
)
from xpu_rt.scheduler.qnn_real_workload import load_cost_matrix
from xpu_rt.scheduling.feedback_loop import (
    LoopConfig,
    init_loop_state,
    step,
)

COST_MATRIX_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "profiled"
    / "qnn_cost_matrix.json"
)


@pytest.fixture(scope="module")
def cost_matrix() -> dict:
    return load_cost_matrix(COST_MATRIX_PATH)


@pytest.fixture()
def calibration() -> CalibrationModel:
    return CalibrationModel(
        schema_version=CALIBRATION_SCHEMA_VERSION,
        target_id="qrb5165",
        overhead_us={
            "dronet": {"CPU": 1000.0, "GPU": 500.0, "DSP": 800.0},
            "yolov8n": {"CPU": 1000.0, "GPU": 500.0, "DSP": 800.0},
        },
        contention_factor={
            "dronet": {"CPU": 1.0, "GPU": 1.0, "DSP": 1.0},
            "yolov8n": {"CPU": 1.0, "GPU": 1.0, "DSP": 1.0},
        },
        history=(),
        created_at="2026-05-15T00:00:00+00:00",
    )


def _seed_cache_for_all_lanes(
    cache_dir: Path, workload_id: str, *, p50_us: float, techniques: tuple[str, ...]
) -> None:
    """Seed the cache for all three lanes so whichever the solver picks hits.

    The candidate lane is derived from the solver's actual device
    assignment; the test would be brittle if we guessed wrong. Seeding
    every lane keeps the test deterministic.
    """

    stats = MeasuredStats(
        mean_us=p50_us,
        p50_us=p50_us,
        p99_us=p50_us * 1.05,
        stdev_us=p50_us * 0.01,
        n_iters=200,
        deadline_met_rate=1.0,
        captured_at=datetime.now(UTC).isoformat(),
        source="synthetic",
    )
    for lane in ("CPU", "GPU", "DSP"):
        append_entry(
            cache_dir, "qrb5165",
            CacheEntry(
                key=CacheKey.make("qrb5165", workload_id, lane, techniques),
                stats=stats,
            ),
        )


def test_measurement_first_flag_short_circuits_predictor(
    cost_matrix, calibration, tmp_path: Path,
) -> None:
    cache_dir = tmp_path / "mcache"
    mem_dir = tmp_path / "loops"
    sentinel = 42_000.0
    _seed_cache_for_all_lanes(
        cache_dir, "dronet", p50_us=sentinel,
        techniques=tuple(sorted(WARM_TECHNIQUES)),
    )
    cfg = LoopConfig(
        measurement_first=True,
        measurement_cache_dir=cache_dir,
        deployment_mode="warm_loop",
    )
    s = init_loop_state(
        "dronet", "qrb5165", cost_matrix, calibration,
        config=cfg, memory_dir=mem_dir,
    )
    s = step(
        s, None, cost_matrix=cost_matrix, config=cfg, memory_dir=mem_dir,
    )
    assert s.current_predicted_makespan_us == pytest.approx(sentinel)
    assert s.history[-1].prediction_source == "measurement_cache"


def test_measurement_first_disabled_uses_predictor(
    cost_matrix, calibration, tmp_path: Path,
) -> None:
    cache_dir = tmp_path / "mcache"
    mem_dir = tmp_path / "loops"
    _seed_cache_for_all_lanes(
        cache_dir, "dronet", p50_us=42_000.0,
        techniques=tuple(sorted(WARM_TECHNIQUES)),
    )
    cfg = LoopConfig(
        measurement_first=False,
        measurement_cache_dir=cache_dir,
        deployment_mode="warm_loop",
    )
    s = init_loop_state(
        "dronet", "qrb5165", cost_matrix, calibration,
        config=cfg, memory_dir=mem_dir,
    )
    s = step(
        s, None, cost_matrix=cost_matrix, config=cfg, memory_dir=mem_dir,
    )
    assert s.history[-1].prediction_source == "predicted"
    # The calibrated number should differ from the sentinel.
    assert s.current_predicted_makespan_us != pytest.approx(42_000.0)


def test_cache_miss_falls_back_to_predictor(
    cost_matrix, calibration, tmp_path: Path,
) -> None:
    cache_dir = tmp_path / "mcache_empty"
    mem_dir = tmp_path / "loops"
    cfg = LoopConfig(
        measurement_first=True,
        measurement_cache_dir=cache_dir,
        deployment_mode="warm_loop",
    )
    s = init_loop_state(
        "dronet", "qrb5165", cost_matrix, calibration,
        config=cfg, memory_dir=mem_dir,
    )
    s = step(
        s, None, cost_matrix=cost_matrix, config=cfg, memory_dir=mem_dir,
    )
    assert s.status in ("running", "max_iter")
    assert s.current_predicted_makespan_us is not None
    assert s.current_predicted_makespan_us > 0
    assert s.history[-1].prediction_source == "predicted"


def test_loop_round_records_prediction_source(
    cost_matrix, calibration, tmp_path: Path,
) -> None:
    cache_dir = tmp_path / "mcache"
    mem_dir = tmp_path / "loops"
    _seed_cache_for_all_lanes(
        cache_dir, "yolov8n", p50_us=55_000.0,
        techniques=tuple(sorted(WARM_TECHNIQUES)),
    )
    cfg_hit = LoopConfig(
        measurement_first=True,
        measurement_cache_dir=cache_dir,
        deployment_mode="warm_loop",
    )
    s_hit = init_loop_state(
        "yolov8n", "qrb5165", cost_matrix, calibration,
        config=cfg_hit, memory_dir=mem_dir,
    )
    s_hit = step(
        s_hit, None, cost_matrix=cost_matrix, config=cfg_hit, memory_dir=mem_dir,
    )
    assert s_hit.history[-1].prediction_source == "measurement_cache"

    cfg_miss = LoopConfig(
        measurement_first=True,
        measurement_cache_dir=tmp_path / "empty",
        deployment_mode="warm_loop",
    )
    s_miss = init_loop_state(
        "dronet", "qrb5165", cost_matrix, calibration,
        config=cfg_miss, memory_dir=mem_dir,
    )
    s_miss = step(
        s_miss, None, cost_matrix=cost_matrix, config=cfg_miss, memory_dir=mem_dir,
    )
    assert s_miss.history[-1].prediction_source == "predicted"
