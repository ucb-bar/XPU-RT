"""Tests for the board-runner MCP tools (host-side, no board required).

The fixtures build a minimal :class:`LoopState` by exercising the real
``init_loop_state`` + first ``step`` on a small synthetic cost matrix.
That avoids hand-rolling chunk dicts and keeps the tests honest about
the actual ``state_to_dict`` shape.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from xpu_rt.mcp.tools.board_runner import (
    MEASUREMENT_SCHEMA_VERSION,
    PLAN_SCHEMA_VERSION,
    xpu_rt_emit_board_plan,
    xpu_rt_ingest_board_measurement,
    xpu_rt_run_board_loop_step,
)
from xpu_rt.runtime.calibration import (
    CALIBRATION_SCHEMA_VERSION,
    CalibrationModel,
)
from xpu_rt.scheduler.qnn_real_workload import load_cost_matrix
from xpu_rt.scheduling.feedback_loop import (
    init_loop_state,
    state_to_dict,
)

WORKLOAD = "yolov8n"
TARGET = "qrb5165"

COST_MATRIX_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "profiled"
    / "qnn_cost_matrix.json"
)


@pytest.fixture(scope="module")
def real_cost_matrix() -> dict[str, Any]:
    return load_cost_matrix(COST_MATRIX_PATH)


def _bootstrap_state_dict(cost_matrix: dict[str, Any]) -> dict[str, Any]:
    cal = CalibrationModel(
        schema_version=CALIBRATION_SCHEMA_VERSION,
        target_id=TARGET,
        overhead_us={WORKLOAD: {"CPU": 100.0, "GPU": 50.0, "DSP": 80.0}},
        contention_factor={WORKLOAD: {"CPU": 1.0, "GPU": 1.0, "DSP": 1.0}},
        history=(),
        created_at="2026-05-15T00:00:00+00:00",
    )
    state = init_loop_state(
        workload_id=WORKLOAD,
        target_id=TARGET,
        cost_matrix=cost_matrix,
        calibration=cal,
    )
    return state_to_dict(state)


def _fake_measurement(
    *,
    target_id: str = TARGET,
    workload_id: str = WORKLOAD,
    per_backend: dict[str, float] | None = None,
    schema: str = MEASUREMENT_SCHEMA_VERSION,
) -> dict[str, Any]:
    if per_backend is None:
        per_backend = {"DSP": 220.0, "CPU": 280.0}
    return {
        "schema_version": schema,
        "target_id": target_id,
        "workload_id": workload_id,
        "captured_at": "2026-05-15T00:00:00+00:00",
        "iters": 10,
        "per_backend_mean_us": per_backend,
        "raw_per_partition_us": [
            {"partition_id": "c0", "backend": b, "mean_us": v,
             "iters": 10, "ok": True, "error": ""}
            for b, v in per_backend.items()
        ],
    }


def test_emit_board_plan_shape(real_cost_matrix) -> None:
    plan = xpu_rt_emit_board_plan(loop_state_dict=_bootstrap_state_dict(real_cost_matrix))
    assert plan["schema_version"] == PLAN_SCHEMA_VERSION
    assert plan["workload_id"] == WORKLOAD
    assert plan["target_id"] == TARGET
    assert plan["iters"] == 10
    assert "generated_at" in plan
    assert len(plan["partitions"]) >= 1
    p0 = plan["partitions"][0]
    for key in ("partition_id", "backend", "dlc_path", "iters", "n_ops"):
        assert key in p0


def test_emit_board_plan_resolves_dsp_dlc_to_quantized_variant(
    real_cost_matrix,
) -> None:
    state = _bootstrap_state_dict(real_cost_matrix)
    # Force one chunk's preferred backend to DSP so the variant suffix
    # is deterministic regardless of solver pick.
    state["current_chunks"][0]["preferred_backend"] = "DSP"
    plan = xpu_rt_emit_board_plan(loop_state_dict=state)
    dsp = [p for p in plan["partitions"] if p["backend"] == "DSP"]
    assert dsp, "expected at least one DSP partition after override"
    assert dsp[0]["dlc_path"].endswith("_quantized.dlc")
    assert WORKLOAD in dsp[0]["dlc_path"]


def test_ingest_board_measurement_validates_target_id_mismatch(
    tmp_path: Path, real_cost_matrix,
) -> None:
    state = _bootstrap_state_dict(real_cost_matrix)
    measurement = _fake_measurement(target_id="some_other_board")
    path = tmp_path / "measurement.json"
    path.write_text(json.dumps(measurement), encoding="utf-8")
    with pytest.raises(ValueError, match="target_id mismatch"):
        xpu_rt_ingest_board_measurement(
            measurement_json_path=str(path),
            loop_state_dict=state,
            persist=False,
        )


def test_ingest_board_measurement_validates_schema_version(
    tmp_path: Path, real_cost_matrix,
) -> None:
    state = _bootstrap_state_dict(real_cost_matrix)
    measurement = _fake_measurement(schema="bogus_v0")
    path = tmp_path / "measurement.json"
    path.write_text(json.dumps(measurement), encoding="utf-8")
    with pytest.raises(ValueError, match="schema_version"):
        xpu_rt_ingest_board_measurement(
            measurement_json_path=str(path),
            loop_state_dict=state,
            persist=False,
        )


def test_ingest_board_measurement_round_trips_minimal(
    tmp_path: Path, real_cost_matrix,
) -> None:
    state = _bootstrap_state_dict(real_cost_matrix)
    measurement = _fake_measurement()
    path = tmp_path / "measurement.json"
    path.write_text(json.dumps(measurement), encoding="utf-8")
    out = xpu_rt_ingest_board_measurement(
        measurement_json_path=str(path),
        loop_state_dict=state,
        persist=False,
    )
    assert out["ok"] is True
    assert out["n_applied"] == 2
    assert out["workload_id"] == WORKLOAD
    new_state = out["state"]
    new_overheads = new_state["current_calibration"]["overhead_us"]
    assert WORKLOAD in new_overheads
    # EMA should have absorbed at least one of the two backends.
    assert set(new_overheads[WORKLOAD].keys()) >= {"DSP", "CPU"}


def test_run_board_loop_step_no_wait_returns_awaiting(
    tmp_path: Path, real_cost_matrix,
) -> None:
    state = _bootstrap_state_dict(real_cost_matrix)
    out = xpu_rt_run_board_loop_step(
        loop_state_dict=state,
        cost_matrix_path=str(tmp_path / "unused.json"),
        wait_for_measurement=False,
    )
    assert out["ok"] is True
    assert out["status"] == "awaiting_measurement"
    assert out["plan"]["schema_version"] == PLAN_SCHEMA_VERSION
    assert "measurement_path" in out
    assert out["waited_s"] == 0.0
