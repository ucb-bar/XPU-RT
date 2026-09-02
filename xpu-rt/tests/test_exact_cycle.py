"""Regression tests for exact-work cyclic separation certificates."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

import exact_cycle  # noqa: E402
from workload import Operation, Workload  # noqa: E402


class ExactCycleTests(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.graph = os.path.join(self.tmp.name, "graph.json")
        with open(self.graph, "w") as f:
            json.dump({"dispatches": {
                "a": {"id": 0, "dependencies": []},
                "b": {"id": 1, "dependencies": ["a"]},
            }}, f)
        self.config = {
            "horizon_ms": 10.0,
            "networks": {
                "model": {
                    "period": 10.0,
                    "window_duration": 10.0,
                    "num_instances": 1,
                    "dispatch_deps_path": self.graph,
                }
            },
        }

    def tearDown(self):
        self.tmp.cleanup()

    @staticmethod
    def _schedule(a_duration=1.0, b_duration=3.0):
        return {"dispatches": {
            "a": {
                "id": 0, "job_name": "model0", "start_time": 0.0,
                "duration": a_duration, "dependencies": [],
                "hardware_target": "CPU_P#0",
            },
            "b": {
                "id": 1, "job_name": "model0", "start_time": a_duration,
                "duration": b_duration, "dependencies": ["a"],
                "hardware_target": "CPU_P#0",
            },
        }}

    def test_contract_requires_integral_exact_count(self):
        contract = exact_cycle.declared_contract(self.config)
        self.assertEqual(contract["total_instances"], 1)
        bad = json.loads(json.dumps(self.config))
        bad["networks"]["model"]["num_instances"] = 2
        with self.assertRaisesRegex(ValueError, "declares 2 instances"):
            exact_cycle.declared_contract(bad)

    def test_schedule_checks_exact_dispatches_and_wrap(self):
        report = exact_cycle.assess_schedule(
            self._schedule(), self.config, ["model"])
        self.assertEqual(report["status"], "pass")
        self.assertEqual(report["dispatches_observed"], 2)
        self.assertEqual(
            report["objective"]["worst_critical_response_ms"], 4.0)

        missing = self._schedule()
        del missing["dispatches"]["b"]
        report = exact_cycle.assess_schedule(missing, self.config, ["model"])
        self.assertEqual(report["status"], "fail")
        self.assertTrue(any("expected 2" in e for e in report["errors"]))

    def test_schedule_must_contain_each_graph_dispatch_and_dependency(self):
        duplicate = self._schedule()
        duplicate["dispatches"]["b"]["id"] = 0
        report = exact_cycle.assess_schedule(
            duplicate, self.config, ["model"])
        self.assertEqual(report["status"], "fail")
        self.assertTrue(any("duplicate dispatch id" in e
                            for e in report["errors"]))

        missing_edge = self._schedule()
        missing_edge["dispatches"]["b"]["dependencies"] = []
        report = exact_cycle.assess_schedule(
            missing_edge, self.config, ["model"])
        self.assertEqual(report["status"], "fail")
        self.assertTrue(any("missing graph dependencies" in e
                            for e in report["errors"]))

    def test_fastest_dag_path_is_solver_independent_lower_bound(self):
        first = Operation(
            [2.0, 1.0], operation_name="a", job_id=0,
            min_start_t=0.0, max_end_t=10.0)
        second = Operation(
            [3.0, 5.0], predecessors=[first], operation_name="b", job_id=0,
            min_start_t=0.0, max_end_t=10.0)
        workload = Workload(
            [first, second], ["CPU_P#0", "CPU_P#1"], np.zeros((2, 2)),
            job_names=["model0"])
        bounds = exact_cycle.workload_lower_bounds(
            workload, self.config, ["model"])
        self.assertEqual(bounds["per_model_ms"]["model"], 4.0)
        self.assertEqual(
            bounds["worst_critical_response_lower_bound_ms"], 4.0)

    def test_feasible_feedback_below_original_floor_is_proven(self):
        original = self._schedule()
        original["metadata"] = {"analytic_response_lower_bounds": {
            "worst_critical_response_lower_bound_ms": 4.0,
            "per_model_ms": {"model": 4.0},
        }}
        feedback = self._schedule(a_duration=1.0, b_duration=1.5)
        result = exact_cycle.separation_certificate(
            original, self.config, feedback, self.config, ["model"])
        self.assertEqual(result["verdict"], "PROVEN")
        self.assertEqual(result["separation_ms"], 1.5)
        self.assertEqual(result["improvement_pct"], 37.5)


if __name__ == "__main__":
    unittest.main()
