"""Unit tests for xpu-rt/feedback.py derive_dispatch_hints.

No MOSEK or board dependency — synthesises Workload + (t, alpha) arrays
directly so the hint logic can be exercised in isolation.

Run with:
    python -m pytest xpu-rt/tests/test_feedback_derivation.py -v
or:
    python xpu-rt/tests/test_feedback_derivation.py
"""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))  # xpu-rt/ on the path

from feedback import derive_dispatch_hints  # noqa: E402
from xpu_rt.scheduler.workload import Operation, Workload  # noqa: E402


def _two_machine_workload(ops):
    machines = ["CPU_P", "CPU_E"]
    transfer = np.zeros((2, 2), dtype=float)
    return Workload(ops, machines, transfer)


class FeedbackDerivationTests(unittest.TestCase):

    def test_clean_run_emits_no_hints(self):
        # Two ops back-to-back on the same machine, no slack, no penalty.
        op_a = Operation(processing_times=[100.0, 200.0], operation_name="a")
        op_b = Operation(processing_times=[100.0, 200.0],
                         predecessors=[op_a], operation_name="b")
        wl = _two_machine_workload([op_a, op_b])

        t = np.array([0.0, 100.0])
        alpha = np.array([[1.0, 0.0], [1.0, 0.0]])

        payload = derive_dispatch_hints(wl, t, alpha)
        self.assertEqual(payload["schema_version"], 1)
        self.assertEqual(payload["dispatches"], {})
        self.assertTrue(payload["model_signals"]["deadline_met"])

    def test_idle_gap_triggers_prefer_finer(self):
        # Op b sits idle for 50us before starting; that's 0.5x its 100us
        # duration → idle_fraction 0.5 ≥ 0.30 threshold → prefer_finer.
        op_a = Operation(processing_times=[100.0, 200.0], operation_name="a")
        op_b = Operation(processing_times=[100.0, 200.0],
                         predecessors=[op_a], operation_name="b")
        wl = _two_machine_workload([op_a, op_b])

        t = np.array([0.0, 150.0])  # 50us gap on CPU_P
        alpha = np.array([[1.0, 0.0], [1.0, 0.0]])

        payload = derive_dispatch_hints(wl, t, alpha)
        self.assertIn("b", payload["dispatches"])
        self.assertIn("prefer_finer", payload["dispatches"]["b"]["hints"])

    def test_cross_cluster_penalty_triggers_fuse_hint(self):
        # b runs on CPU_E with a 2x penalty when its predecessor lived on
        # CPU_P. Nominal duration on CPU_E = 100; effective = 200; ratio 2.0
        # ≥ 1.5 threshold → consider_fuse_with_pred.
        op_a = Operation(processing_times=[100.0, 200.0], operation_name="a")
        op_b = Operation(
            processing_times=[100.0, 100.0],
            predecessors=[op_a],
            operation_name="b",
            processing_times_by_pred={(0, 1): 200.0},  # k_pred=CPU_P → k_curr=CPU_E
        )
        wl = _two_machine_workload([op_a, op_b])

        t = np.array([0.0, 100.0])
        alpha = np.array([[1.0, 0.0], [0.0, 1.0]])  # a on CPU_P, b on CPU_E

        payload = derive_dispatch_hints(wl, t, alpha)
        d = payload["dispatches"]["b"]
        self.assertIn("consider_fuse_with_pred", d["hints"])
        self.assertGreaterEqual(d["transfer_cost_ratio"], 1.5)

    def test_deadline_slack_triggers_prefer_coarser(self):
        # b finishes long before its deadline → suggest prefer_coarser.
        op = Operation(
            processing_times=[100.0, 200.0],
            operation_name="b",
            deadline_us=10_000.0,
        )
        wl = _two_machine_workload([op])
        t = np.array([0.0])
        alpha = np.array([[1.0, 0.0]])

        payload = derive_dispatch_hints(wl, t, alpha)
        self.assertIn("prefer_coarser", payload["dispatches"]["b"]["hints"])
        self.assertTrue(payload["model_signals"]["deadline_met"])

    def test_pin_target_when_current_is_slow(self):
        # CPU_E is 5x faster (20us) than CPU_P (100us). Solver picked CPU_P
        # → pin_target=CPU_E should fire.
        op = Operation(processing_times=[100.0, 20.0], operation_name="b")
        wl = _two_machine_workload([op])
        t = np.array([0.0])
        alpha = np.array([[1.0, 0.0]])  # picked CPU_P (the slow one)

        payload = derive_dispatch_hints(wl, t, alpha)
        hints = payload["dispatches"]["b"]["hints"]
        self.assertTrue(any(h.startswith("pin_target=") and "CPU_E" in h
                            for h in hints))

    def test_skipped_op_marked_in_model_signals(self):
        op_a = Operation(processing_times=[100.0, 200.0], operation_name="a")
        op_b = Operation(
            processing_times=[100.0, 200.0],
            predecessors=[op_a],
            operation_name="b",
            deadline_us=50.0,
            skip_allowed=True,
        )
        wl = _two_machine_workload([op_a, op_b])
        # Pretend the solver dropped op_b under deadline pressure.
        wl.skipped_op_indices = [1]
        wl.solver_state = {"problem_status": "optimal", "makespan": 100.0}

        t = np.array([0.0, 0.0])
        alpha = np.array([[1.0, 0.0], [1.0, 0.0]])

        payload = derive_dispatch_hints(wl, t, alpha)
        self.assertIn("b", payload["model_signals"]["skip_triggered"])
        self.assertFalse(payload["model_signals"]["deadline_met"])
        self.assertIn("prefer_finer", payload["dispatches"]["b"]["hints"])

    def test_solver_failure_returns_empty_payload(self):
        op = Operation(processing_times=[100.0, 200.0], operation_name="a")
        wl = _two_machine_workload([op])
        payload = derive_dispatch_hints(wl, t=None, alpha=None)
        self.assertEqual(payload["dispatches"], {})
        self.assertTrue(payload["model_signals"]["solver_failed"])

    def test_run_id_is_stamped_when_provided(self):
        op = Operation(processing_times=[100.0, 200.0], operation_name="a")
        wl = _two_machine_workload([op])
        t = np.array([0.0])
        alpha = np.array([[1.0, 0.0]])

        payload = derive_dispatch_hints(
            wl, t, alpha, run_id="custom_id_42",
            source_schedule="schedule.json",
        )
        self.assertEqual(payload["run_id"], "custom_id_42")
        self.assertEqual(payload["source_schedule"], "schedule.json")


if __name__ == "__main__":
    unittest.main(verbosity=2)
