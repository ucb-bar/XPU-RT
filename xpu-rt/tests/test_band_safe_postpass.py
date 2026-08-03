"""Phase A3 tests: compaction never violates max_end_t; automerge
never crosses periodic instance boundaries."""

from __future__ import annotations

import os
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

import numpy as np

from automerge import automerge_adjacent, _instance_idx_from_name


class TestInstanceIdxParse(unittest.TestCase):
    def test_periodic(self):
        self.assertEqual(_instance_idx_from_name("mlp_control0_dispatch_5"), 0)
        self.assertEqual(_instance_idx_from_name("dronet1_dispatch_22"), 1)
        self.assertEqual(_instance_idx_from_name("mlp_control3_dispatch_0"), 3)

    def test_nonperiodic(self):
        self.assertIsNone(_instance_idx_from_name("yolov8_nano_dispatch_3"))

    def test_garbage(self):
        self.assertIsNone(_instance_idx_from_name("just_some_name"))


class TestAutomergeRefusesCrossInstance(unittest.TestCase):
    def _fixture(self):
        # Two ops that LOOK adjacent on the same core but belong to
        # different periodic instances. The merge MUST be refused.
        return {
            "dispatches": {
                "mlp_control0_dispatch_0": {
                    "id": 0, "ordinal": 1, "total": 1,
                    "dependencies": [],
                    "hardware_target": "CPU_P#0",
                    "start_time": 9.99,
                    "duration": 0.005,
                    "job_name": "mlp_control0",
                },
                "mlp_control1_dispatch_0": {
                    "id": 1, "ordinal": 1, "total": 1,
                    "dependencies": [],
                    "hardware_target": "CPU_P#0",
                    "start_time": 10.0,
                    "duration": 0.005,
                    "job_name": "mlp_control1",
                },
            },
            "metadata": {"makespan": 10.005, "num_operations": 2,
                         "machines": ["CPU_P#0"], "machine_combinations": [["CPU_P#0"]]},
        }

    def test_refuse_different_job_names(self):
        before = self._fixture()
        after = automerge_adjacent(before, max_gap_us=50.0)
        self.assertEqual(len(after["dispatches"]), 2,
                         "cross-instance ops with different job_names must NOT merge")

    def test_refuse_when_only_instance_suffix_differs(self):
        # Force same job_name but different parsed instance — the defensive
        # 1a guard should still refuse.
        before = self._fixture()
        before["dispatches"]["mlp_control1_dispatch_0"]["job_name"] = "mlp_control0"
        after = automerge_adjacent(before, max_gap_us=50.0)
        self.assertEqual(len(after["dispatches"]), 2,
                         "instance-suffix mismatch must refuse merge")


class TestAutomergeRefusesDeadlineMiss(unittest.TestCase):
    def test_refuse_when_either_op_overruns(self):
        fixture = {
            "dispatches": {
                "mlp_control0_dispatch_0": {
                    "id": 0, "ordinal": 1, "total": 1, "dependencies": [],
                    "hardware_target": "CPU_P#0",
                    "start_time": 0.0, "duration": 1.0,
                    "job_name": "mlp_control0",
                    "deadline_miss": True,
                },
                "mlp_control0_dispatch_1": {
                    "id": 1, "ordinal": 1, "total": 1,
                    "dependencies": ["mlp_control0_dispatch_0"],
                    "hardware_target": "CPU_P#0",
                    "start_time": 1.05, "duration": 0.5,
                    "job_name": "mlp_control0",
                },
            },
            "metadata": {"makespan": 1.55, "num_operations": 2,
                         "machines": ["CPU_P#0"], "machine_combinations": [["CPU_P#0"]]},
        }
        after = automerge_adjacent(fixture, max_gap_us=50.0)
        self.assertEqual(len(after["dispatches"]), 2,
                         "must not merge when either op has deadline_miss")


class TestCompactionBandSafe(unittest.TestCase):
    """The post-shift finish of any op with max_end_t must not exceed it."""

    def test_compaction_monotone_in_slack(self):
        from workload import Operation, Workload
        from compaction import left_shift_compact

        # Two-op chain: a -> b, both on machine M1 with max_end_t.
        # The solver leaves a 5us gap before b. After compaction, b
        # should slide left but stay within max_end_t.
        # Use processing_times[0] = duration on combo 0.
        a = Operation(
            processing_times=[3.0],
            operation_id=0, operation_name="a",
            job_id=0, min_start_t=0.0, max_end_t=10.0,
        )
        b = Operation(
            processing_times=[4.0],
            predecessors=[a],
            operation_id=1, operation_name="b",
            job_id=0, min_start_t=0.0, max_end_t=10.0,
        )
        wl = Workload(
            operations=[a, b],
            machines=["M1"],
            transfer_times=np.zeros((1, 1)),
        )

        t = np.array([0.0, 8.0])           # b placed at 8 (gap of 5 after a ends at 3)
        alpha = np.array([[1.0], [1.0]])    # both on combo 0

        t_new, alpha_new = left_shift_compact(t, alpha, wl)
        # b should slide left to start at 3.0 (immediately after a).
        self.assertAlmostEqual(float(t_new[0]), 0.0)
        self.assertAlmostEqual(float(t_new[1]), 3.0)
        # And b's finish 3+4=7 is well within max_end_t=10.
        self.assertLessEqual(float(t_new[1]) + 4.0, 10.0 + 1e-6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
