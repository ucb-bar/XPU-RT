"""Tests for left_shift_compact (compaction.py)."""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from compaction import left_shift_compact, compaction_savings_us  # noqa: E402
from workload import Operation, Workload  # noqa: E402


def _make_workload(ops_spec, machines):
    """Build a Workload from a list of (name, durations_per_combo,
    [pred_indices], min_start_t)."""
    operations = []
    for spec in ops_spec:
        name, durations, pred_ids, min_start_t = spec
        preds = [operations[p] for p in pred_ids]
        op = Operation(
            processing_times=list(durations),
            predecessors=preds,
            operation_name=name,
            min_start_t=min_start_t,
        )
        operations.append(op)
    n_m = len(machines)
    transfer_times = np.zeros((n_m, n_m))
    machine_combinations = [[m] for m in machines]
    wl = Workload(operations, machines, transfer_times,
                  machine_combinations=machine_combinations)
    return wl


class CompactionPlantedGapTest(unittest.TestCase):
    """A list scheduler placed op2 at t=5 when it could fit at t=1.
    Verify the pass slides it left and reduces makespan."""

    def test_planted_gap_eliminated(self):
        # op0 on machine A duration 1, op1 on machine B duration 1 (no deps),
        # op2 on machine B duration 1, depends on op0. List scheduler may
        # have placed op2 at t=5 by mistake; t=1 is the real earliest.
        wl = _make_workload(
            ops_spec=[
                ("op0", [1.0, 9.0], [], None),     # picks machine A
                ("op1", [9.0, 1.0], [], None),     # picks machine B
                ("op2", [9.0, 1.0], [0], None),    # picks machine B, dep on op0
            ],
            machines=["A", "B"],
        )
        # t/alpha as a sloppy list scheduler might have produced:
        t = np.array([0.0, 0.0, 5.0])
        alpha = np.array([
            [1, 0],  # op0 on A
            [0, 1],  # op1 on B
            [0, 1],  # op2 on B
        ], dtype=float)

        t2, alpha2 = left_shift_compact(t, alpha, wl)

        # op0 stays at 0 (no preds, no release).
        self.assertAlmostEqual(t2[0], 0.0)
        # op1 stays at 0 (no preds, on B, no prior B user).
        self.assertAlmostEqual(t2[1], 0.0)
        # op2 must start >= max(op0.end=1, op1.end=1 on B) = 1.0.
        self.assertAlmostEqual(t2[2], 1.0)

        savings = compaction_savings_us(t, t2, alpha2, wl)
        self.assertGreater(savings["delta_us"], 3.9)  # 5-1 = 4ms saved
        self.assertEqual(savings["ops_moved"], 1)


class CompactionIdempotentTest(unittest.TestCase):
    """A tight (e.g. MOSEK-produced) schedule should pass through unchanged."""

    def test_already_tight_unchanged(self):
        wl = _make_workload(
            ops_spec=[
                ("op0", [1.0, 9.0], [], None),
                ("op1", [9.0, 1.0], [0], None),
            ],
            machines=["A", "B"],
        )
        t = np.array([0.0, 1.0])  # op1 starts the instant op0 ends.
        alpha = np.array([[1, 0], [0, 1]], dtype=float)
        t2, _ = left_shift_compact(t, alpha, wl)
        np.testing.assert_allclose(t2, t, atol=1e-9)

    def test_running_twice_is_a_noop(self):
        wl = _make_workload(
            ops_spec=[
                ("op0", [1.0, 9.0], [], None),
                ("op1", [9.0, 1.0], [], None),
                ("op2", [9.0, 1.0], [0], None),
            ],
            machines=["A", "B"],
        )
        t = np.array([0.0, 0.0, 5.0])
        alpha = np.array([[1, 0], [0, 1], [0, 1]], dtype=float)
        t2, _ = left_shift_compact(t, alpha, wl)
        t3, _ = left_shift_compact(t2, alpha, wl)
        np.testing.assert_allclose(t3, t2, atol=1e-9)


class CompactionReleaseTimeTest(unittest.TestCase):
    """A periodic op with min_start_t must not slide before its release."""

    def test_release_time_honored(self):
        wl = _make_workload(
            ops_spec=[
                ("op0", [1.0], [], None),
                # min_start_t=10 — this is a periodic-release op.
                ("op_periodic", [1.0], [], 10.0),
            ],
            machines=["A"],
        )
        # Sloppy schedule put op_periodic at t=20.
        t = np.array([0.0, 20.0])
        alpha = np.array([[1], [1]], dtype=float)
        t2, _ = left_shift_compact(t, alpha, wl)
        # op_periodic slides left but only as far as max(release=10, prev_on_A=1).
        self.assertAlmostEqual(t2[1], 10.0)


class CompactionPrecedenceTest(unittest.TestCase):
    """Predecessor end time must be respected even when machine is free."""

    def test_dep_chain(self):
        wl = _make_workload(
            ops_spec=[
                ("op0", [3.0], [], None),
                ("op1", [1.0], [0], None),
                ("op2", [1.0], [1], None),
            ],
            machines=["A"],
        )
        # All three on machine A. Solver placed at 0/5/10.
        t = np.array([0.0, 5.0, 10.0])
        alpha = np.array([[1], [1], [1]], dtype=float)
        t2, _ = left_shift_compact(t, alpha, wl)
        self.assertAlmostEqual(t2[0], 0.0)
        self.assertAlmostEqual(t2[1], 3.0)  # after op0 ends
        self.assertAlmostEqual(t2[2], 4.0)  # after op1 ends


if __name__ == "__main__":
    unittest.main()
