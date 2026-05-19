"""Unit tests for the MOSEK MILP bridge."""

from __future__ import annotations

import pytest

from xpu_rt.targets.backends.qnn.mosek_bridge import solve_qnn_mosek
from xpu_rt.targets.backends.qnn.realtime import build_realtime_workload


def _yaml(copies: int = 12, dronet_deadline_us: int = 40_000):
    return {
        "machines": ["HTA", "GPU", "CPU"],
        "workloads": [
            {"id": "yolov8n", "copies": 1,
             "period_us": None, "deadline_us": None},
            {"id": "dronet", "copies": copies,
             "period_us": 40_000, "deadline_us": dronet_deadline_us},
        ],
    }


def _lat(dronet_cpu_us: int = 7_400):
    return {
        "yolov8n": {"HTA": 355_000, "GPU": 369_000, "CPU": 325_000},
        "dronet":  {"HTA": 158_000, "GPU":  87_000, "CPU": dronet_cpu_us},
    }


@pytest.mark.slow
def test_feasible_with_realistic_bounds():
    wl, _ = build_realtime_workload(
        _yaml(), _lat(), makespan_bound_us=355_000.0,
    )
    result = solve_qnn_mosek(wl, time_limit=60.0)
    assert result["feasible"], result
    assert result["status"] in ("optimal", "optimal_inaccurate")
    assert result["makespan_us"] <= 355_000.0 + 1e-3
    # All 12 dronets should land on CPU (only feasible backend with
    # latency < 40 ms in this test fixture).
    dronet_machines = {
        o["machine"] for o in result["ops"] if o["workload"] == "dronet"
    }
    assert dronet_machines == {"CPU"}
    # YOLOv8n should not also pile on top of CPU (otherwise the 12
    # dronets + 325ms yolov8n would exceed the 355ms bound).
    yolo_machines = {
        o["machine"] for o in result["ops"] if o["workload"] == "yolov8n"
    }
    assert yolo_machines <= {"HTA", "GPU"}


@pytest.mark.slow
def test_infeasible_when_no_backend_meets_deadline():
    # DroNet on CPU = 50 ms > 40 ms deadline → no feasible cell.
    # Real-only enforcement now raises BEFORE the MILP runs, since
    # every backend is infeasible — the loud failure is correct.
    with pytest.raises(ValueError, match="all .* backend cells rejected"):
        build_realtime_workload(
            _yaml(copies=1),
            _lat(dronet_cpu_us=50_000),  # all dronet cells > deadline_us
            makespan_bound_us=355_000.0,
        )


@pytest.mark.slow
def test_milp_status_carried_through():
    wl, _ = build_realtime_workload(
        _yaml(copies=2), _lat(),
        makespan_bound_us=355_000.0,
    )
    result = solve_qnn_mosek(wl, time_limit=30.0)
    assert "milp_status" in result
    assert "problem_status" in result["milp_status"]
    if result["feasible"]:
        assert result["milp_status"]["makespan_us"] is not None
