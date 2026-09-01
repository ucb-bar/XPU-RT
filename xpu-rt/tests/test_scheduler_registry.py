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


class MosekParameterTests(unittest.TestCase):

    def test_generic_parameter_parser_preserves_mosek_types(self):
        from scheduler import _parse_mosek_params

        parsed = _parse_mosek_params(
            "MSK_IPAR_MIO_MAX_NUM_SOLUTIONS=1;"
            "MSK_IPAR_MIO_MEMORY_EMPHASIS_LEVEL=1;"
            "MSK_IPAR_MIO_NODE_SELECTION=MSK_MIO_NODE_SELECTION_FIRST;"
            "MSK_DPAR_MIO_TOL_REL_GAP=0.05;"
            "MSK_SPAR_WRITE_DATA_PARAM=ignored")
        self.assertEqual(parsed["MSK_IPAR_MIO_MAX_NUM_SOLUTIONS"], 1)
        self.assertIsInstance(parsed["MSK_IPAR_MIO_MAX_NUM_SOLUTIONS"], int)
        self.assertEqual(parsed["MSK_IPAR_MIO_MEMORY_EMPHASIS_LEVEL"], 1)
        self.assertEqual(
            parsed["MSK_IPAR_MIO_NODE_SELECTION"],
            "MSK_MIO_NODE_SELECTION_FIRST")
        self.assertEqual(parsed["MSK_DPAR_MIO_TOL_REL_GAP"], 0.05)
        self.assertIsInstance(parsed["MSK_DPAR_MIO_TOL_REL_GAP"], float)
        self.assertEqual(parsed["MSK_SPAR_WRITE_DATA_PARAM"], "ignored")


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


class CpSatPrecisionTests(unittest.TestCase):
    """Fractional profile durations must survive CP-SAT's integer clock."""

    @classmethod
    def setUpClass(cls):
        try:
            import ortools  # noqa: F401
        except ImportError as exc:
            raise unittest.SkipTest(f"OR-Tools not available: {exc}")

    def test_fractional_predecessor_cannot_overlap_successor(self):
        first = Operation([1.292625], operation_name="first")
        second = Operation([0.711583], predecessors=[first],
                           operation_name="second")
        workload = Workload(
            [first, second], ["CPU_P#0"], np.zeros((1, 1), dtype=float))

        starts, alpha, _, _ = get_scheduler("cpsat")(
            workload, time_limit=10, solver_verbosity=0)

        self.assertEqual(int(np.argmax(alpha[0])), 0)
        self.assertEqual(int(np.argmax(alpha[1])), 0)
        self.assertGreaterEqual(starts[1], starts[0] + 1.292625)


class CpSatExactCycleObjectiveTests(unittest.TestCase):
    """The certificate must optimize the same per-instance quantities we report."""

    @classmethod
    def setUpClass(cls):
        try:
            import ortools  # noqa: F401
        except ImportError as exc:
            raise unittest.SkipTest(f"OR-Tools not available: {exc}")

    def test_sequential_certificate_is_per_instance_and_optimal(self):
        first = Operation(
            [6.0], operation_name="critical0_a", job_id=0,
            min_start_t=0.0, max_end_t=10.0)
        second = Operation(
            [6.0], predecessors=[first], operation_name="critical0_b",
            job_id=0, min_start_t=0.0, max_end_t=10.0)
        workload = Workload(
            [first, second], ["CPU_P#0"], np.zeros((1, 1)),
            job_names=["critical0"])

        starts, alpha, _, _ = get_scheduler("cpsat")(
            workload,
            time_limit=10,
            objective_mode="exact_cycle_worst_response",
            critical_models=["critical"],
            objective_stop_after="worst_critical_response",
        )

        self.assertIsNotNone(starts)
        self.assertEqual(int(np.argmax(alpha[0])), 0)
        cert = workload.solver_certificate
        self.assertTrue(cert["certified"])
        self.assertEqual(cert["jobs_modeled"], 1)
        self.assertEqual(
            [(p["name"], p["objective"], p["best_bound"])
             for p in cert["phases"]],
            [
                ("job_deadline_misses", 1.0, 1.0),
                ("max_job_lateness", 2000.0, 2000.0),
                ("worst_critical_response", 12000.0, 12000.0),
            ],
        )

    def test_critical_response_precedes_heavy_response(self):
        critical = Operation(
            [7.0], operation_name="critical0_a", job_id=0,
            min_start_t=0.0, max_end_t=20.0)
        heavy = Operation(
            [4.0], operation_name="heavy0_a", job_id=1,
            min_start_t=0.0, max_end_t=20.0)
        workload = Workload(
            [critical, heavy], ["CPU_P#0"], np.zeros((1, 1)),
            job_names=["critical0", "heavy0"])

        starts, _, _, _ = get_scheduler("cpsat")(
            workload,
            time_limit=10,
            objective_mode="exact_cycle_worst_response",
            critical_models=["critical"],
            heavy_model="heavy",
            objective_stop_after="heavy_max_response",
        )

        self.assertEqual(starts.tolist(), [0.0, 7.0])
        cert = workload.solver_certificate
        self.assertTrue(cert["certified"])
        self.assertEqual(
            [p["name"] for p in cert["phases"]],
            ["job_deadline_misses", "max_job_lateness",
             "worst_critical_response", "heavy_max_response"],
        )
        self.assertEqual(cert["phases"][2]["objective"], 7000.0)
        self.assertEqual(cert["phases"][3]["objective"], 11000.0)


if __name__ == "__main__":
    unittest.main()
