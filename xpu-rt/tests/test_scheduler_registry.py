"""Regression tests for the scheduler registry introduced in milestone 1.

The registry's ``mosek`` entry must be a transparent indirection over the
existing ``scheduler.schedule`` function: any caller that flips from a direct
import to ``schedulers.get_scheduler("mosek")`` should observe bit-identical
``(t, alpha)`` output on the same workload + kwargs.

Run with:
    python -m pytest xpu-rt/tests/test_scheduler_registry.py -v
"""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))  # xpu-rt/ on the path

from workload import Operation, Workload  # noqa: E402
from schedulers import (  # noqa: E402
    available_schedulers,
    get_scheduler,
    register,
)
from metrics import compute_metrics  # noqa: E402


def _tiny_two_op_workload():
    """A->B chain on a 2-machine system, both ops cheap and distinct per machine."""
    op_a = Operation(processing_times=[10.0, 20.0], operation_name="a")
    op_b = Operation(
        processing_times=[15.0, 25.0],
        predecessors=[op_a],
        operation_name="b",
    )
    machines = ["CPU_P", "CPU_E"]
    transfer = np.zeros((2, 2), dtype=float)
    return Workload([op_a, op_b], machines, transfer)


class RegistryTests(unittest.TestCase):

    def test_mosek_registered_by_default(self):
        self.assertIn("mosek", available_schedulers())

    def test_unknown_scheduler_raises(self):
        with self.assertRaises(ValueError):
            get_scheduler("does_not_exist")

    def test_register_new_scheduler_roundtrip(self):
        marker = object()

        def fake(workload, **kw):
            return marker, kw

        register("__fake_for_test", fake)
        self.assertIn("__fake_for_test", available_schedulers())
        out = get_scheduler("__fake_for_test")(_tiny_two_op_workload(), foo=1)
        self.assertIs(out[0], marker)
        self.assertEqual(out[1], {"foo": 1})


class MosekIndirectionTests(unittest.TestCase):
    """Confirm `get_scheduler('mosek')` matches `scheduler.schedule` byte-for-byte."""

    @classmethod
    def setUpClass(cls):
        try:
            import cvxpy  # noqa: F401
            import mosek  # noqa: F401
        except ImportError as exc:
            raise unittest.SkipTest(f"MOSEK/cvxpy not available: {exc}")

    def test_direct_and_registry_paths_match(self):
        from scheduler import schedule as direct_schedule

        wl_direct = _tiny_two_op_workload()
        wl_registry = _tiny_two_op_workload()

        kwargs = dict(
            solver_verbosity=0,
            time_limit=5,
            restrict_makespan_to_nonperiodic=True,
            prune_cross_period_constraints=True,
        )

        t_d, a_d, _, _ = direct_schedule(wl_direct, **kwargs)
        t_r, a_r, _, _ = get_scheduler("mosek")(wl_registry, **kwargs)

        # The MILP is convex and deterministic given the same seed/inputs; the
        # registry is a pure forwarder, so outputs must match exactly.
        np.testing.assert_allclose(t_d, t_r, rtol=0, atol=1e-9)
        np.testing.assert_allclose(a_d, a_r, rtol=0, atol=1e-9)


class MetricsShapeTests(unittest.TestCase):
    """`compute_metrics` should run on (workload, t, alpha) without MOSEK."""

    def test_metrics_dict_contains_required_keys(self):
        wl = _tiny_two_op_workload()
        # Hand-build a feasible schedule: A at t=0 on machine 0, B at t=10 on machine 0.
        t = np.array([0.0, 10.0])
        alpha = np.array([[1.0, 0.0], [1.0, 0.0]])

        m = compute_metrics(wl, t, alpha, scheduler_name="test", solver_wall_time_s=0.123)

        for key in (
            "scheduler",
            "num_operations",
            "makespan_us",
            "deadline_miss_count",
            "deadline_miss_ratio",
            "per_machine_utilization",
            "cross_device_transitions",
            "critical_path_us",
            "solver_wall_time_s",
        ):
            self.assertIn(key, m)

        self.assertEqual(m["scheduler"], "test")
        self.assertEqual(m["num_operations"], 2)
        self.assertEqual(m["deadline_miss_count"], 0)
        self.assertEqual(m["cross_device_transitions"], 0)
        # Both ops on CPU_P → utilization on CPU_P is (10+15)/25 = 1.0, CPU_E = 0.
        self.assertAlmostEqual(m["per_machine_utilization"]["CPU_P"], 1.0)
        self.assertAlmostEqual(m["per_machine_utilization"]["CPU_E"], 0.0)
        # Critical path = 10 + 15 = 25 us.
        self.assertAlmostEqual(m["critical_path_us"], 25.0)

    def test_cross_device_transition_counted_when_pred_on_different_machine(self):
        wl = _tiny_two_op_workload()
        # A on machine 0, B on machine 1 → one cross-device transition (A→B).
        t = np.array([0.0, 10.0])
        alpha = np.array([[1.0, 0.0], [0.0, 1.0]])
        m = compute_metrics(wl, t, alpha, scheduler_name="test")
        self.assertEqual(m["cross_device_transitions"], 1)


if __name__ == "__main__":
    unittest.main()
