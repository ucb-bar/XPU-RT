"""Tests for the v3 two-term per-(workload, backend) calibration model."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from xpu_rt.runtime.calibration import (
    CALIBRATION_SCHEMA_VERSION,
    DEPLOYMENT_MODE_COLD,
    DEPLOYMENT_MODE_WARM,
    PROVENANCE_DEFAULT_NO_DATA,
    PROVENANCE_MEASURED,
    TECHNIQUE_CACHED_CONTEXT,
    TECHNIQUE_PREALLOC_BUFFERS,
    CalibrationModel,
    CalibrationSchemaMismatchError,
    MeasurementRecord,
    apply,
    bootstrap_contention_from_closed_loop,
    bootstrap_from_solo_measurements,
    bootstrap_warm_from_csv_traces,
    bootstrap_warm_from_measurements,
    load,
    save,
    techniques_to_mode,
    update_from_measurement,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
COST_MATRIX_PATH = REPO_ROOT / "xpu-rt" / "data" / "profiled" / "qnn_cost_matrix.json"
E2E_PATH = REPO_ROOT / "xpu-rt" / "data" / "profiled" / "qnn_e2e" / "measurements.json"
WARM_E2E_PATH = REPO_ROOT / "xpu-rt" / "data" / "profiled" / "qnn_warm" / "measurements.json"
RT_YOLO_CSV = REPO_ROOT / "realtime_qnn" / "rt_yolo_f.csv"
RT_DRONE_CSV = REPO_ROOT / "realtime_qnn" / "rt_drone_f.csv"


@pytest.fixture(scope="module")
def raw_cost_matrix() -> dict:
    return json.loads(COST_MATRIX_PATH.read_text())


@pytest.fixture(scope="module")
def e2e_measurements() -> dict:
    return json.loads(E2E_PATH.read_text())


@pytest.fixture(scope="module")
def closed_loop_rounds() -> list[dict[str, object]]:
    """Hand-encoded yolov8n DSP closed-loop rounds (per-iter measured)."""
    return [
        {"workload_id": "yolov8n", "backend": "DSP", "measured_us": 254800.0},
        {"workload_id": "yolov8n", "backend": "DSP", "measured_us": 350900.0},
        {"workload_id": "yolov8n", "backend": "DSP", "measured_us": 255600.0},
        {"workload_id": "yolov8n", "backend": "DSP", "measured_us": 257300.0},
    ]


def test_bootstrap_yields_positive_dsp_overhead(raw_cost_matrix, e2e_measurements) -> None:
    model = bootstrap_from_solo_measurements(raw_cost_matrix, e2e_measurements)
    assert model.overhead_us["yolov8n"]["DSP"] > 50_000.0


def test_bootstrap_cpu_overhead_small_or_zero(raw_cost_matrix, e2e_measurements) -> None:
    model = bootstrap_from_solo_measurements(raw_cost_matrix, e2e_measurements)
    for w in ("yolov8n", "dronet"):
        e2e_cpu = float(e2e_measurements["matrix"][w]["CPU"]["mean_us"])
        assert model.overhead_us[w]["CPU"] <= 0.20 * e2e_cpu


def test_bootstrap_seeds_unit_contention_default_no_data(
    raw_cost_matrix, e2e_measurements,
) -> None:
    model = bootstrap_from_solo_measurements(raw_cost_matrix, e2e_measurements)
    for w in ("yolov8n", "dronet"):
        for b in ("CPU", "GPU", "DSP"):
            assert model.contention_factor[w][b] == 1.0
            assert model.contention_provenance[w][b] == PROVENANCE_DEFAULT_NO_DATA


def test_ema_update_moves_overhead_toward_residual(raw_cost_matrix, e2e_measurements) -> None:
    model = bootstrap_from_solo_measurements(raw_cost_matrix, e2e_measurements)
    old = model.overhead_us["yolov8n"]["DSP"]
    target = 500_000.0
    measurement = MeasurementRecord(
        workload_id="yolov8n",
        backend="DSP",
        measured_us=target + 60_000.0,
        per_op_sum_us=60_000.0,
        predicted_us=old + 60_000.0,
    )
    updated = update_from_measurement(model, measurement, ema_alpha=0.5)
    new = updated.overhead_us["yolov8n"]["DSP"]
    implied_target = measurement.measured_us - measurement.per_op_sum_us
    assert old < new < implied_target
    assert new == pytest.approx(old + 0.5 * (implied_target - old))
    assert len(updated.history) == 1
    assert updated.overhead_us["dronet"]["DSP"] == model.overhead_us["dronet"]["DSP"]


def test_save_load_round_trips(raw_cost_matrix, e2e_measurements, tmp_path) -> None:
    model = bootstrap_from_solo_measurements(raw_cost_matrix, e2e_measurements)
    measurement = MeasurementRecord("yolov8n", "DSP", 300_000.0, 60_000.0, 250_000.0)
    model = update_from_measurement(model, measurement)
    path = tmp_path / "calib.json"
    save(model, path)
    reloaded = load(path)
    assert reloaded.schema_version == CALIBRATION_SCHEMA_VERSION
    assert reloaded.target_id == model.target_id
    assert reloaded.overhead_us == model.overhead_us
    assert reloaded.contention_factor == model.contention_factor
    assert reloaded.contention_provenance == model.contention_provenance
    assert reloaded.history == model.history


def test_apply_preserves_op_count(raw_cost_matrix, e2e_measurements) -> None:
    model = bootstrap_from_solo_measurements(raw_cost_matrix, e2e_measurements)
    projected = apply(model, raw_cost_matrix, workload_id="yolov8n")
    for workload in ("yolov8n", "dronet"):
        assert len(projected[workload]) == len(raw_cost_matrix[workload])
    assert "_calibration_overhead_us" in projected
    assert "_calibration_overhead_us_per_workload" in projected
    assert "_calibration_contention_factor" in projected
    assert "_calibration_contention_factor_per_workload" in projected
    assert projected.get("_meta") == raw_cost_matrix.get("_meta")
    assert projected["_calibration_overhead_us"] == model.overhead_us["yolov8n"]
    assert projected["_calibration_contention_factor"] == model.contention_factor["yolov8n"]


def test_per_workload_overhead_independent(raw_cost_matrix, e2e_measurements) -> None:
    model = bootstrap_from_solo_measurements(raw_cost_matrix, e2e_measurements)
    yolo_dsp = model.overhead_us["yolov8n"]["DSP"]
    dro_dsp = model.overhead_us["dronet"]["DSP"]
    assert yolo_dsp != dro_dsp
    assert dro_dsp < yolo_dsp


def test_apply_without_workload_id_warns_and_zeros(
    raw_cost_matrix, e2e_measurements,
) -> None:
    model = bootstrap_from_solo_measurements(raw_cost_matrix, e2e_measurements)
    with pytest.warns(DeprecationWarning, match="workload_id"):
        projected = apply(model, raw_cost_matrix)
    overhead = projected["_calibration_overhead_us"]
    contention = projected["_calibration_contention_factor"]
    assert all(v == 0.0 for v in overhead.values())
    assert all(v == 1.0 for v in contention.values())
    assert "_calibration_overhead_us_per_workload" in projected
    assert projected["_calibration_overhead_us_per_workload"]["yolov8n"]["DSP"] > 0.0


def test_load_legacy_v1_raises_typed_error(tmp_path: Path) -> None:
    v1_path = tmp_path / "v1.json"
    v1_path.write_text(json.dumps({
        "schema_version": "calibration_model_v1",
        "target_id": "qrb5165",
        "overhead_us": {"CPU": 0.0, "GPU": 0.0, "DSP": 1000.0},
        "contention_factor": {"CPU": 1.0, "GPU": 1.0, "DSP": 1.0},
        "history": [],
        "created_at": "2026-05-15T00:00:00+00:00",
    }))
    with pytest.raises(CalibrationSchemaMismatchError, match="bootstrap"):
        load(v1_path)


def test_load_legacy_v2_raises_typed_error(tmp_path: Path) -> None:
    v2_path = tmp_path / "v2.json"
    v2_path.write_text(json.dumps({
        "schema_version": "calibration_model_v2",
        "target_id": "qrb5165",
        "overhead_us": {"yolov8n": {"DSP": 1000.0}},
        "contention_factor": {"CPU": 1.0, "GPU": 1.0, "DSP": 1.0},
        "history": [],
        "created_at": "2026-05-15T00:00:00+00:00",
    }))
    with pytest.raises(CalibrationSchemaMismatchError, match="bootstrap"):
        load(v2_path)


def test_bootstrap_contention_from_closed_loop_yolov8n_dsp_under_one(
    raw_cost_matrix, e2e_measurements, closed_loop_rounds,
) -> None:
    """The yolov8n DSP contention factor must be in (0.5, 1.0).

    Mean measured per-iter is 279.65 ms vs solo DSP E2E of 354.88 ms,
    giving a ratio of ~0.788. Under contention yolov8n on DSP runs
    *faster* than its solo baseline (12× dronet are on CPU, leaving
    DSP fully available with possible cache-warmth effects). We
    document the counter-intuitive direction honestly.
    """
    model = bootstrap_from_solo_measurements(raw_cost_matrix, e2e_measurements)
    fitted = bootstrap_contention_from_closed_loop(
        model, closed_loop_rounds, raw_cost_matrix,
    )
    factor = fitted.contention_factor["yolov8n"]["DSP"]
    assert 0.5 < factor < 1.0
    # The mean of the four measured / solo_e2e ratios is ~0.788.
    assert factor == pytest.approx(0.788, abs=0.05)
    assert fitted.contention_provenance["yolov8n"]["DSP"] == PROVENANCE_MEASURED


def test_contention_default_one_for_unmeasured_workloads(
    raw_cost_matrix, e2e_measurements, closed_loop_rounds,
) -> None:
    """Cells without contended ground truth must stay 1.0 + flagged."""
    model = bootstrap_from_solo_measurements(raw_cost_matrix, e2e_measurements)
    fitted = bootstrap_contention_from_closed_loop(
        model, closed_loop_rounds, raw_cost_matrix,
    )
    # dronet has no contended observations in the rounds list.
    for b in ("CPU", "GPU", "DSP"):
        assert fitted.contention_factor["dronet"][b] == 1.0
        assert fitted.contention_provenance["dronet"][b] == PROVENANCE_DEFAULT_NO_DATA
    # yolov8n on CPU/GPU also have no observations.
    for b in ("CPU", "GPU"):
        assert fitted.contention_factor["yolov8n"][b] == 1.0
        assert fitted.contention_provenance["yolov8n"][b] == PROVENANCE_DEFAULT_NO_DATA


def test_apply_v3_two_term_predictor() -> None:
    """Synthetic test: apply() exposes overhead AND contention per workload."""
    raw = {
        "wl_a": {"op0": {"CPU": 100.0, "DSP": 200.0}},
        "wl_b": {"op0": {"CPU": 50.0,  "DSP": 80.0}},
    }
    model = CalibrationModel(
        schema_version=CALIBRATION_SCHEMA_VERSION,
        target_id="synthetic",
        overhead_us={
            "wl_a": {"CPU": 10.0, "DSP": 30.0},
            "wl_b": {"CPU": 5.0,  "DSP": 8.0},
        },
        contention_factor={
            "wl_a": {"CPU": 1.5, "DSP": 0.8},
            "wl_b": {"CPU": 1.0, "DSP": 1.0},
        },
        history=(),
        created_at="2026-05-15T00:00:00+00:00",
        contention_provenance={
            "wl_a": {"CPU": PROVENANCE_MEASURED, "DSP": PROVENANCE_MEASURED},
            "wl_b": {"CPU": PROVENANCE_DEFAULT_NO_DATA, "DSP": PROVENANCE_DEFAULT_NO_DATA},
        },
    )
    projected = apply(model, raw, workload_id="wl_a")
    # Per-op costs are passthrough (calibration applied once per partition).
    assert projected["wl_a"]["op0"]["DSP"] == 200.0
    # Flat keys for wl_a expose overhead and contention.
    assert projected["_calibration_overhead_us"] == {"CPU": 10.0, "DSP": 30.0}
    assert projected["_calibration_contention_factor"] == {"CPU": 1.5, "DSP": 0.8}
    # Two-term predicted cost for wl_a on DSP (one-op partition):
    chain = projected["wl_a"]["op0"]["DSP"]
    overhead = projected["_calibration_overhead_us"]["DSP"]
    contention = projected["_calibration_contention_factor"]["DSP"]
    predicted = (chain + overhead) * contention
    assert predicted == pytest.approx((200.0 + 30.0) * 0.8)


def test_bootstrap_contention_skips_zero_base(raw_cost_matrix, e2e_measurements) -> None:
    """Cells with chain_sum + base_overhead == 0 must be skipped, not div-by-zero."""
    model = bootstrap_from_solo_measurements(raw_cost_matrix, e2e_measurements)
    rounds = [{"workload_id": "nonexistent", "backend": "DSP", "measured_us": 1234.0}]
    fitted = bootstrap_contention_from_closed_loop(model, rounds, raw_cost_matrix)
    # No KeyError, no ZeroDivisionError; contention map is unchanged.
    assert fitted.contention_factor == model.contention_factor


def test_bootstrap_contention_median_aggregator(
    raw_cost_matrix, e2e_measurements, closed_loop_rounds,
) -> None:
    """Median is more robust to round 2's outlier than mean."""
    model = bootstrap_from_solo_measurements(raw_cost_matrix, e2e_measurements)
    mean_fit = bootstrap_contention_from_closed_loop(
        model, closed_loop_rounds, raw_cost_matrix, aggregator="mean",
    )
    median_fit = bootstrap_contention_from_closed_loop(
        model, closed_loop_rounds, raw_cost_matrix, aggregator="median",
    )
    # Median ratio (0.720, 0.725, 0.989) middle ≈ 0.725 < mean ≈ 0.788.
    assert median_fit.contention_factor["yolov8n"]["DSP"] < mean_fit.contention_factor["yolov8n"]["DSP"]


# === Bug 1 + Bug 2 fix coverage ==========================================


def test_solo_measurement_updates_overhead_leaves_contention(raw_cost_matrix, e2e_measurements) -> None:
    """Solo MeasurementRecord (empty concurrent_workloads) updates only overhead."""
    model = bootstrap_from_solo_measurements(raw_cost_matrix, e2e_measurements)
    closed_loop_rounds_list = [
        {"workload_id": "yolov8n", "backend": "DSP", "measured_us": 254800.0},
        {"workload_id": "yolov8n", "backend": "DSP", "measured_us": 350900.0},
    ]
    model = bootstrap_contention_from_closed_loop(model, closed_loop_rounds_list, raw_cost_matrix)
    old_overhead = model.overhead_us["yolov8n"]["DSP"]
    old_contention = model.contention_factor["yolov8n"]["DSP"]

    m = MeasurementRecord(
        workload_id="yolov8n",
        backend="DSP",
        measured_us=364700.0,
        per_op_sum_us=60400.0,
        predicted_us=0.0,
        concurrent_workloads=(),  # solo
    )
    updated = update_from_measurement(model, m, ema_alpha=0.5)
    assert updated.overhead_us["yolov8n"]["DSP"] != pytest.approx(old_overhead)
    assert updated.contention_factor["yolov8n"]["DSP"] == pytest.approx(old_contention)


def test_contended_measurement_updates_contention_leaves_overhead(raw_cost_matrix, e2e_measurements) -> None:
    """Contended MeasurementRecord updates only contention."""
    model = bootstrap_from_solo_measurements(raw_cost_matrix, e2e_measurements)
    closed_loop_rounds_list = [
        {"workload_id": "yolov8n", "backend": "DSP", "measured_us": 254800.0},
        {"workload_id": "yolov8n", "backend": "DSP", "measured_us": 350900.0},
    ]
    model = bootstrap_contention_from_closed_loop(model, closed_loop_rounds_list, raw_cost_matrix)
    old_overhead = model.overhead_us["yolov8n"]["DSP"]
    old_contention = model.contention_factor["yolov8n"]["DSP"]

    m = MeasurementRecord(
        workload_id="yolov8n",
        backend="DSP",
        measured_us=254800.0,
        per_op_sum_us=60400.0,
        predicted_us=0.0,
        concurrent_workloads=("dronet",) * 12,
    )
    updated = update_from_measurement(model, m, ema_alpha=0.5)
    assert updated.overhead_us["yolov8n"]["DSP"] == pytest.approx(old_overhead)
    assert updated.contention_factor["yolov8n"]["DSP"] != pytest.approx(old_contention)


def test_compose_predicted_makespan_solo_ignores_contention(raw_cost_matrix, e2e_measurements) -> None:
    """compose_predicted_makespan_us with empty concurrent_workloads forces contention=1.0."""
    from xpu_rt.runtime.calibration import compose_predicted_makespan_us

    model = bootstrap_from_solo_measurements(raw_cost_matrix, e2e_measurements)
    closed_loop_rounds_list = [{"workload_id": "yolov8n", "backend": "DSP", "measured_us": 254800.0}]
    model = bootstrap_contention_from_closed_loop(model, closed_loop_rounds_list, raw_cost_matrix)
    # Force a non-unit contention to be sure it would change the output if applied.
    assert model.contention_factor["yolov8n"]["DSP"] != pytest.approx(1.0)

    per_lane = {"DSP": 60000.0}
    solo = compose_predicted_makespan_us(
        model=model,
        workload_id="yolov8n",
        per_lane_busy_us=per_lane,
        concurrent_workloads=(),
    )
    overhead = model.overhead_us["yolov8n"]["DSP"]
    assert solo == pytest.approx(60000.0 + overhead)


def test_compose_predicted_makespan_contended_uses_contention(raw_cost_matrix, e2e_measurements) -> None:
    """compose_predicted_makespan_us with concurrent_workloads applies stored contention."""
    from xpu_rt.runtime.calibration import compose_predicted_makespan_us

    model = bootstrap_from_solo_measurements(raw_cost_matrix, e2e_measurements)
    closed_loop_rounds_list = [{"workload_id": "yolov8n", "backend": "DSP", "measured_us": 254800.0}]
    model = bootstrap_contention_from_closed_loop(model, closed_loop_rounds_list, raw_cost_matrix)
    contention = model.contention_factor["yolov8n"]["DSP"]
    overhead = model.overhead_us["yolov8n"]["DSP"]

    per_lane = {"DSP": 60000.0}
    contended = compose_predicted_makespan_us(
        model=model,
        workload_id="yolov8n",
        per_lane_busy_us=per_lane,
        concurrent_workloads=("dronet",),
    )
    assert contended == pytest.approx((60000.0 + overhead) * contention)


def test_compose_predicted_makespan_takes_max_over_lanes(raw_cost_matrix, e2e_measurements) -> None:
    """When multiple lanes are used, predicted = max over lanes (lane-parallel finish)."""
    from xpu_rt.runtime.calibration import compose_predicted_makespan_us

    model = bootstrap_from_solo_measurements(raw_cost_matrix, e2e_measurements)
    # Two lanes with different busy + overhead — the slower lane defines makespan.
    per_lane = {"CPU": 10000.0, "DSP": 50000.0}
    result = compose_predicted_makespan_us(
        model=model,
        workload_id="yolov8n",
        per_lane_busy_us=per_lane,
        concurrent_workloads=(),
    )
    cpu_finish = 10000.0 + model.overhead_us["yolov8n"]["CPU"]
    dsp_finish = 50000.0 + model.overhead_us["yolov8n"]["DSP"]
    assert result == pytest.approx(max(cpu_finish, dsp_finish))


# === v4 deployment-mode coverage =========================================


def test_warm_bootstrap_from_csv_yields_zero_overhead_for_yolov8n_dsp(
    raw_cost_matrix, e2e_measurements,
) -> None:
    """yolov8n's warm p50 (~55.5ms) is below DSP chain-sum (~60.4ms), so warm
    overhead clamps to ~0 (well under 5ms)."""
    model = bootstrap_from_solo_measurements(raw_cost_matrix, e2e_measurements)
    warm_model = bootstrap_warm_from_csv_traces(
        model, [RT_YOLO_CSV], raw_cost_matrix,
    )
    assert "yolov8n" in warm_model.overhead_us_warm
    assert "DSP" in warm_model.overhead_us_warm["yolov8n"]
    assert warm_model.overhead_us_warm["yolov8n"]["DSP"] < 5000.0
    # Cold overhead must be untouched.
    assert warm_model.overhead_us["yolov8n"]["DSP"] == model.overhead_us["yolov8n"]["DSP"]


def test_warm_bootstrap_from_csv_yields_small_overhead_for_dronet_gpu(
    raw_cost_matrix, e2e_measurements,
) -> None:
    """dronet GPU warm p50 ~1.67ms minus chain ~0.85ms ≈ 0.8ms positive overhead."""
    model = bootstrap_from_solo_measurements(raw_cost_matrix, e2e_measurements)
    warm_model = bootstrap_warm_from_csv_traces(
        model, [RT_DRONE_CSV], raw_cost_matrix,
    )
    warm_ovh = warm_model.overhead_us_warm["dronet"]["GPU"]
    assert 0.0 < warm_ovh < 2000.0


def test_apply_cold_vs_warm_returns_different_overhead(
    raw_cost_matrix, e2e_measurements,
) -> None:
    """Same model + workload, two deployment modes → cold overhead ≫ warm overhead."""
    model = bootstrap_from_solo_measurements(raw_cost_matrix, e2e_measurements)
    model = bootstrap_warm_from_csv_traces(model, [RT_YOLO_CSV, RT_DRONE_CSV], raw_cost_matrix)
    cold_view = apply(
        model, raw_cost_matrix, workload_id="yolov8n", deployment_mode=DEPLOYMENT_MODE_COLD
    )
    warm_view = apply(
        model, raw_cost_matrix, workload_id="yolov8n", deployment_mode=DEPLOYMENT_MODE_WARM
    )
    cold_dsp = cold_view["_calibration_overhead_us"]["DSP"]
    warm_dsp = warm_view["_calibration_overhead_us"].get("DSP", 0.0)
    assert cold_dsp > 200_000.0  # ~295ms graph init included
    assert warm_dsp < 5_000.0     # init amortised away
    assert cold_dsp > warm_dsp * 10
    assert cold_view["_calibration_deployment_mode"] == DEPLOYMENT_MODE_COLD
    assert warm_view["_calibration_deployment_mode"] == DEPLOYMENT_MODE_WARM


def test_update_from_measurement_with_cached_context_routes_to_warm_dict(
    raw_cost_matrix, e2e_measurements,
) -> None:
    """A measurement tagged with TECHNIQUE_CACHED_CONTEXT updates the warm
    overhead dict and leaves the cold dict untouched."""
    model = bootstrap_from_solo_measurements(raw_cost_matrix, e2e_measurements)
    # Warm map is currently empty for yolov8n DSP — start at 0.0.
    assert model.overhead_us_warm.get("yolov8n", {}).get("DSP", 0.0) == 0.0
    old_cold = model.overhead_us["yolov8n"]["DSP"]
    measurement = MeasurementRecord(
        workload_id="yolov8n",
        backend="DSP",
        measured_us=55_000.0,
        per_op_sum_us=60_400.0,
        predicted_us=0.0,
        deployment_techniques=(TECHNIQUE_CACHED_CONTEXT, TECHNIQUE_PREALLOC_BUFFERS),
    )
    updated = update_from_measurement(model, measurement, ema_alpha=0.5)
    # Cold dict unchanged.
    assert updated.overhead_us["yolov8n"]["DSP"] == pytest.approx(old_cold)
    # Warm dict moved off zero (target_overhead = measured - chain = -5400, clamped to 0;
    # EMA from 0 with alpha=0.5 toward 0 stays at 0 — bump the measured to verify movement).
    measurement2 = MeasurementRecord(
        workload_id="yolov8n",
        backend="DSP",
        measured_us=70_000.0,
        per_op_sum_us=60_400.0,
        predicted_us=0.0,
        deployment_techniques=(TECHNIQUE_CACHED_CONTEXT,),
    )
    updated2 = update_from_measurement(updated, measurement2, ema_alpha=0.5)
    warm_after = updated2.overhead_us_warm["yolov8n"]["DSP"]
    assert warm_after > 0.0
    assert updated2.overhead_us["yolov8n"]["DSP"] == pytest.approx(old_cold)
    assert techniques_to_mode((TECHNIQUE_CACHED_CONTEXT,)) == DEPLOYMENT_MODE_WARM
    assert techniques_to_mode(()) == DEPLOYMENT_MODE_COLD


def test_load_v3_raises_typed_error(tmp_path: Path) -> None:
    """v3-shaped files (no _warm fields) refuse to load under v4."""
    v3_path = tmp_path / "v3.json"
    v3_path.write_text(json.dumps({
        "schema_version": "calibration_model_v3",
        "target_id": "qrb5165",
        "overhead_us": {"yolov8n": {"DSP": 295000.0, "GPU": 286000.0, "CPU": 56000.0}},
        "contention_factor": {"yolov8n": {"DSP": 0.788, "GPU": 1.0, "CPU": 1.0}},
        "contention_provenance": {"yolov8n": {"DSP": "measured", "GPU": "default_no_data", "CPU": "default_no_data"}},
        "history": [],
        "created_at": "2026-05-15T00:00:00+00:00",
    }))
    with pytest.raises(CalibrationSchemaMismatchError, match="bootstrap"):
        load(v3_path)


def test_warm_bootstrap_from_measurements_matches_csv(
    raw_cost_matrix, e2e_measurements,
) -> None:
    """The JSON-driven warm bootstrap and the CSV-driven one agree within
    a few ms (JSON aggregates the same trace post-warmup)."""
    base = bootstrap_from_solo_measurements(raw_cost_matrix, e2e_measurements)
    warm_json = json.loads(WARM_E2E_PATH.read_text())
    from_json = bootstrap_warm_from_measurements(base, warm_json, raw_cost_matrix)
    from_csv = bootstrap_warm_from_csv_traces(
        base, [RT_YOLO_CSV, RT_DRONE_CSV], raw_cost_matrix,
    )
    # Both populate the same (workload, backend) cells.
    assert set(from_json.overhead_us_warm.keys()) == set(from_csv.overhead_us_warm.keys())
    # dronet GPU: agree within 1ms (median-of-csv vs p50-of-json).
    j = from_json.overhead_us_warm["dronet"]["GPU"]
    c = from_csv.overhead_us_warm["dronet"]["GPU"]
    assert abs(j - c) < 1000.0
