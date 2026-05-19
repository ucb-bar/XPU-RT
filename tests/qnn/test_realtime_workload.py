"""Unit tests for the realtime workload builder."""

from __future__ import annotations

import yaml

from xpu_rt.targets.backends.qnn.realtime import (
    build_realtime_workload, load_workload_yaml,
)


def _yaml_doc():
    return yaml.safe_load("""
machines: ["HTA", "GPU", "CPU"]
workloads:
  - id: yolov8n
    copies: 1
    sla_us: 200000
    period_us: null
    deadline_us: null
  - id: dronet
    copies: 12
    sla_us: 40000
    period_us: 40000
    deadline_us: 40000
""")


def test_thirteen_ops_one_yolov8n_twelve_dronets():
    wl, summ = build_realtime_workload(
        _yaml_doc(),
        latency_matrix={
            "yolov8n": {"HTA": 355_000, "GPU": 369_000, "CPU": 325_000},
            "dronet":  {"HTA": 158_000, "GPU":  87_000, "CPU":   7_400},
        },
        makespan_bound_us=355_000.0,
    )
    assert len(wl.operations) == 13
    assert sum(1 for s in summ if s.workload_id == "yolov8n") == 1
    assert sum(1 for s in summ if s.workload_id == "dronet") == 12


def test_dronet_infeasible_combinations_block_slow_backends():
    """Backends slower than 40 ms per dronet instance must be excluded."""
    wl, _ = build_realtime_workload(
        _yaml_doc(),
        latency_matrix={
            "yolov8n": {"HTA": 355_000, "GPU": 369_000, "CPU": 325_000},
            "dronet":  {"HTA": 158_000, "GPU":  87_000, "CPU":   7_400},
        },
        makespan_bound_us=355_000.0,
    )
    machines = wl.machines
    cpu = machines.index("CPU")
    for op in wl.operations:
        if op.job_id == "dronet":
            assert cpu not in op.infeasible_combinations
            # 87000 > 40000 and 158000 > 40000 → both blocked.
            assert machines.index("GPU") in op.infeasible_combinations
            assert machines.index("HTA") in op.infeasible_combinations


def test_yolov8n_no_infeasible_cells_when_no_deadline():
    """YOLOv8n has no per-instance deadline → every backend is feasible."""
    wl, _ = build_realtime_workload(
        _yaml_doc(),
        latency_matrix={
            "yolov8n": {"HTA": 355_000, "GPU": 369_000, "CPU": 325_000},
            "dronet":  {"HTA": 158_000, "GPU":  87_000, "CPU":   7_400},
        },
        makespan_bound_us=355_000.0,
    )
    for op in wl.operations:
        if op.job_id == "yolov8n":
            assert op.infeasible_combinations == set()


def test_max_end_t_propagates():
    wl, _ = build_realtime_workload(
        _yaml_doc(),
        latency_matrix={
            "yolov8n": {"HTA": 355_000, "GPU": 369_000, "CPU": 325_000},
            "dronet":  {"HTA": 158_000, "GPU":  87_000, "CPU":   7_400},
        },
        makespan_bound_us=355_000.0,
    )
    for op in wl.operations:
        assert op.max_end_t == 355_000.0
        assert op.deadline_us == 355_000.0
        assert op.min_start_t == 0.0


def test_missing_cell_is_infeasible():
    wl, _ = build_realtime_workload(
        _yaml_doc(),
        latency_matrix={
            "yolov8n": {"HTA": 355_000, "GPU": 369_000, "CPU": 325_000},
            "dronet":  {"HTA": None, "GPU": 87_000, "CPU": 7_400},  # missing HTA
        },
        makespan_bound_us=355_000.0,
    )
    machines = wl.machines
    hta = machines.index("HTA")
    for op in wl.operations:
        if op.job_id == "dronet":
            assert hta in op.infeasible_combinations
