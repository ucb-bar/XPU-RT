"""Tests for SchedulerReport (xpu-rt/profiling.py)."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

import numpy as np

# Resolve top-level modules without needing the package to be installed.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from workload import Operation, Workload  # noqa: E402
from profiling import SchedulerReport  # noqa: E402


def _tiny_two_op_workload():
    """A->B chain on a 2-machine system. Same shape as test_scheduler_registry.py."""
    op_a = Operation(processing_times=[10.0, 20.0], operation_name="a")
    op_b = Operation(
        processing_times=[15.0, 25.0],
        predecessors=[op_a],
        operation_name="b",
    )
    machines = ["CPU_P", "CPU_E"]
    transfer = np.zeros((2, 2), dtype=float)
    return Workload([op_a, op_b], machines, transfer)


class SchedulerReportTests(unittest.TestCase):
    """Build a report from a hand-crafted feasible schedule (no MOSEK needed)."""

    def setUp(self):
        self.wl = _tiny_two_op_workload()
        # Both ops on machine 0 (CPU_P), back to back.
        self.t = np.array([0.0, 10.0])
        self.alpha = np.array([[1.0, 0.0], [1.0, 0.0]])

    def test_report_has_required_fields(self):
        r = SchedulerReport.from_solver_state(
            self.wl, self.t, self.alpha,
            solver_name="MOSEK", solve_wall_s=0.42, solver_status="optimal",
        )
        d = r.to_dict()
        for key in (
            "schema_version", "solver_name", "solver_status", "solve_wall_s",
            "n_operations", "n_combinations", "n_resources_by_kind",
            "makespan_cycles", "utilization", "granularity",
            "dispatch_durations", "critical_path",
            "fusion_applied", "git_sha", "captured_at",
        ):
            self.assertIn(key, d, f"missing key: {key}")
        self.assertEqual(d["schema_version"], 1)
        self.assertEqual(d["solver_name"], "MOSEK")
        self.assertEqual(d["solver_status"], "optimal")
        self.assertAlmostEqual(d["solve_wall_s"], 0.42)
        self.assertEqual(d["n_operations"], 2)

    def test_utilization_shape(self):
        r = SchedulerReport.from_solver_state(
            self.wl, self.t, self.alpha,
            solver_name="MOSEK", solve_wall_s=0.0,
        )
        # Both machines must be present, each with busy/idle/frac_busy.
        self.assertEqual(set(r.utilization.keys()), {"CPU_P", "CPU_E"})
        for k, v in r.utilization.items():
            self.assertIn("busy_cycles", v)
            self.assertIn("idle_cycles", v)
            self.assertIn("frac_busy", v)
        # CPU_P ran both ops (dur 10+15=25), makespan=25 → frac_busy ≈ 1.0.
        self.assertAlmostEqual(r.utilization["CPU_P"]["frac_busy"], 1.0, places=5)
        # CPU_E ran nothing → frac_busy ≈ 0.0.
        self.assertAlmostEqual(r.utilization["CPU_E"]["frac_busy"], 0.0, places=5)

    def test_granularity_buckets(self):
        r = SchedulerReport.from_solver_state(
            self.wl, self.t, self.alpha,
            solver_name="MOSEK", solve_wall_s=0.0,
        )
        # Durations 10 and 15 both < 1k, so all-2 in lt_1k.
        buckets = r.granularity["buckets"]
        self.assertEqual(buckets["lt_1k"], 2)
        self.assertEqual(buckets["lt_10k"], 0)
        self.assertEqual(buckets["ge_1M"], 0)
        # Numeric percentiles.
        self.assertEqual(len(r.dispatch_durations), 2)
        self.assertAlmostEqual(r.granularity["max"], 15.0)

    def test_write_json_roundtrip(self):
        r = SchedulerReport.from_solver_state(
            self.wl, self.t, self.alpha,
            solver_name="HIGHS", solve_wall_s=1.0, solver_status="optimal",
        )
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "report.json")
            r.write_json(path)
            with open(path) as f:
                d = json.load(f)
            self.assertEqual(d["solver_name"], "HIGHS")
            self.assertEqual(d["n_operations"], 2)
            self.assertEqual(len(d["dispatch_durations"]), 2)


if __name__ == "__main__":
    unittest.main()
