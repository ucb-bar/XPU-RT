"""Realtime workload builder rejects analytical-bound cells by default."""

from __future__ import annotations

import pytest

from xpu_rt.targets.backends.qnn.realtime import build_realtime_workload


def _yaml():
    return {
        "machines": ["HTA", "GPU", "CPU"],
        "workloads": [
            {"id": "yolov8n", "copies": 1,
             "period_us": None, "deadline_us": None},
            {"id": "dronet", "copies": 1,
             "period_us": 40_000, "deadline_us": 40_000},
        ],
    }


def test_bare_float_cells_are_treated_as_measured():
    wl, _ = build_realtime_workload(
        _yaml(),
        latency_matrix={
            "yolov8n": {"HTA": 355_000, "GPU": 369_000, "CPU": 325_000},
            "dronet":  {"HTA":  20_000, "GPU":  87_000, "CPU":   7_400},
        },
        makespan_bound_us=355_000.0,
    )
    # Both ops should have ZERO infeasible cells from provenance gating.
    machines = wl.machines
    yolov = next(o for o in wl.operations if o.job_id == "yolov8n")
    assert yolov.infeasible_combinations == set()


def test_analytical_bound_without_bound_only_is_rejected():
    wl, _ = build_realtime_workload(
        _yaml(),
        latency_matrix={
            "yolov8n": {"HTA": 355_000, "GPU": 369_000, "CPU": 325_000},
            "dronet":  {"HTA": {"mean_us": 30_000,
                                 "provenance": "analytical_bound"},
                        "GPU": 87_000,
                        "CPU": 7_400},
        },
        makespan_bound_us=355_000.0,
    )
    machines = wl.machines
    dronet = next(o for o in wl.operations if o.job_id == "dronet")
    assert machines.index("HTA") in dronet.infeasible_combinations


def test_analytical_bound_accepted_when_opted_in():
    wl, _ = build_realtime_workload(
        _yaml(),
        latency_matrix={
            "yolov8n": {"HTA": 355_000, "GPU": 369_000, "CPU": 325_000},
            "dronet":  {"HTA": {"mean_us": 30_000,
                                 "provenance": "analytical_bound",
                                 "bound_only": True},
                        "GPU": 87_000,
                        "CPU": 7_400},
        },
        makespan_bound_us=355_000.0,
        allow_analytical_bounds=True,
    )
    machines = wl.machines
    dronet = next(o for o in wl.operations if o.job_id == "dronet")
    assert machines.index("HTA") not in dronet.infeasible_combinations


def test_all_cells_rejected_raises_loudly():
    with pytest.raises(ValueError, match="all .* backend cells rejected"):
        build_realtime_workload(
            _yaml(),
            latency_matrix={
                "yolov8n": {"HTA": 355_000, "GPU": 369_000, "CPU": 325_000},
                "dronet":  {"HTA": 50_000, "GPU": 87_000, "CPU": 158_000},
            },
            makespan_bound_us=355_000.0,
        )


def test_none_cell_marked_infeasible():
    wl, _ = build_realtime_workload(
        _yaml(),
        latency_matrix={
            "yolov8n": {"HTA": 355_000, "GPU": 369_000, "CPU": 325_000},
            "dronet":  {"HTA": None, "GPU": 87_000, "CPU": 7_400},
        },
        makespan_bound_us=355_000.0,
    )
    machines = wl.machines
    dronet = next(o for o in wl.operations if o.job_id == "dronet")
    assert machines.index("HTA") in dronet.infeasible_combinations
