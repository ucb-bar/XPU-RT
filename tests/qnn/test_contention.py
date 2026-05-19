"""Unit tests for the multiplicative contention model."""

from __future__ import annotations

import json

from xpu_rt.targets.backends.qnn.contention import (
    ContentionState,
    per_backend_measured_from_execution,
    per_backend_predicted_from_schedule,
    write_contention_log,
)


def test_initial_factors_default_to_one():
    s = ContentionState()
    s.ensure(["CPU", "GPU", "DSP"])
    assert s.factors == {"CPU": 1.0, "GPU": 1.0, "DSP": 1.0}


def test_update_produces_measured_over_predicted_ratio():
    s = ContentionState(max_factor=10.0, ema_weight=1.0)
    s.ensure(["CPU", "GPU"])
    new = s.update(
        per_backend_predicted_us={"CPU": 100.0, "GPU": 200.0},
        per_backend_measured_us={"CPU": 120.0, "GPU": 180.0},
    )
    assert abs(new["CPU"] - 1.2) < 1e-6
    assert abs(new["GPU"] - 0.9) < 1e-6


def test_ema_smooths_against_prior():
    s = ContentionState(max_factor=10.0, ema_weight=0.5)
    s.ensure(["CPU"])
    s.factors["CPU"] = 1.0
    new = s.update(
        per_backend_predicted_us={"CPU": 100.0},
        per_backend_measured_us={"CPU": 200.0},
    )
    # raw ratio = 2.0; EMA 0.5×2.0 + 0.5×1.0 = 1.5.
    assert abs(new["CPU"] - 1.5) < 1e-6


def test_apply_scales_solo_latencies():
    s = ContentionState()
    s.factors = {"CPU": 1.2, "GPU": 0.9}
    matrix = {
        "yolov8n": {"CPU": 100.0, "GPU": 200.0, "DSP": None},
        "dronet":  {"CPU":  10.0, "GPU":  20.0, "DSP": None},
    }
    out = s.apply(matrix)
    assert out["yolov8n"]["CPU"] == 120.0
    assert out["yolov8n"]["GPU"] == 180.0
    assert out["yolov8n"]["DSP"] is None


def test_convergence_after_two_stable_rounds():
    s = ContentionState(max_factor=5.0, ema_weight=1.0)
    s.ensure(["CPU"])
    # Two rounds with the same prediction = same measured.
    for _ in range(2):
        s.update(
            per_backend_predicted_us={"CPU": 100.0},
            per_backend_measured_us={"CPU": 100.0},
        )
    assert s.is_converged()


def test_not_converged_with_oscillating_factor():
    s = ContentionState(max_factor=5.0, ema_weight=1.0)
    s.ensure(["CPU"])
    s.update(
        per_backend_predicted_us={"CPU": 100.0},
        per_backend_measured_us={"CPU": 100.0},
    )
    s.update(
        per_backend_predicted_us={"CPU": 100.0},
        per_backend_measured_us={"CPU": 200.0},
    )
    assert not s.is_converged()


def test_per_backend_predicted_from_schedule():
    sched = {"ops": [
        {"machine": "CPU", "predicted_us": 30.0},
        {"machine": "CPU", "predicted_us": 70.0},
        {"machine": "GPU", "predicted_us": 50.0},
    ]}
    pred = per_backend_predicted_from_schedule(sched)
    assert pred == {"CPU": 100.0, "GPU": 50.0}


def test_per_backend_measured_from_execution_lane_finish_us():
    execution = {"lane_finish_us": {"CPU": 123.0, "GPU": 45.0}}
    assert per_backend_measured_from_execution(execution) == {
        "CPU": 123.0, "GPU": 45.0,
    }


def test_write_contention_log_appends(tmp_path):
    s = ContentionState()
    s.factors = {"CPU": 1.2}
    p = write_contention_log(tmp_path, round_index=0, state=s)
    p = write_contention_log(tmp_path, round_index=1, state=s)
    lines = (tmp_path / "contention.jsonl").read_text().splitlines()
    assert len(lines) == 2
    for ln in lines:
        rec = json.loads(ln)
        assert rec["factors"] == {"CPU": 1.2}
