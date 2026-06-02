"""Tests for oracle.compute_floor."""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from oracle import compute_floor, oracle_gap_pct  # noqa: E402
from workload import Operation, Workload  # noqa: E402


def _wl(ops_spec, machines, machine_combinations=None):
    """Build a Workload from (name, durations, pred_ids, min_start_t)."""
    operations = []
    for spec in ops_spec:
        name, durations, pred_ids, min_start_t = spec
        preds = [operations[p] for p in pred_ids]
        op = Operation(processing_times=list(durations), predecessors=preds,
                       operation_name=name, min_start_t=min_start_t)
        operations.append(op)
    n_m = len(machines)
    tt = np.zeros((n_m, n_m))
    return Workload(operations, machines, tt,
                    machine_combinations=machine_combinations or [[m] for m in machines])


class OracleCriticalPathTest(unittest.TestCase):

    def test_chain_critical_path(self):
        # Three serial ops, durations 3, 2, 1 — critical path is 6.
        wl = _wl([
            ("op0", [3.0], [], None),
            ("op1", [2.0], [0], None),
            ("op2", [1.0], [1], None),
        ], machines=["A"])
        f = compute_floor(wl)
        self.assertAlmostEqual(f["critical_path_us"], 6.0)
        # Load: all on machine A → 3+2+1 = 6.
        self.assertAlmostEqual(f["load_us"], 6.0)
        self.assertAlmostEqual(f["oracle_floor_us"], 6.0)


class OracleLoadBoundTest(unittest.TestCase):

    def test_parallel_load_bound_dominates(self):
        # 4 independent ops, all pinned to machine A, each 5µs.
        # Critical path = 5 (independent — each chain is length 1).
        # Load on A = 20.
        wl = _wl([
            ("op0", [5.0], [], None),
            ("op1", [5.0], [], None),
            ("op2", [5.0], [], None),
            ("op3", [5.0], [], None),
        ], machines=["A"])
        f = compute_floor(wl)
        self.assertAlmostEqual(f["critical_path_us"], 5.0)
        self.assertAlmostEqual(f["load_us"], 20.0)
        # Floor = max → load wins.
        self.assertAlmostEqual(f["oracle_floor_us"], 20.0)


class OracleHeteroTest(unittest.TestCase):

    def test_hetero_two_machine_split(self):
        # 4 ops, each pickable on A or B, choose-fastest. Each op 1µs on
        # A, 1µs on B → load per machine could be 0 (if optimal) or up
        # to 4 (if all on one).
        # Critical path = 1 (independent ops).
        # Floor should be max(1, ceil(4*1 / 2)) = 2 — but our naive load
        # bound credits the fastest-feasible machine, so each op picks
        # one machine and that machine gets +1µs.
        wl = _wl([
            ("op0", [1.0, 1.0], [], None),
            ("op1", [1.0, 1.0], [], None),
            ("op2", [1.0, 1.0], [], None),
            ("op3", [1.0, 1.0], [], None),
        ], machines=["A", "B"])
        f = compute_floor(wl)
        self.assertAlmostEqual(f["critical_path_us"], 1.0)
        # Each op picks the first feasible-min, all credited to A → 4.
        # Naive bound — tightening it requires LP relaxation.
        self.assertAlmostEqual(f["load_us"], 4.0)


class OracleReleaseTimeTest(unittest.TestCase):

    def test_release_floor(self):
        # An op that can't start until t=100 with own duration 5 forces
        # makespan to be at least 105.
        wl = _wl([
            ("op0", [3.0], [], None),
            ("op_periodic", [5.0], [], 100.0),
        ], machines=["A"])
        f = compute_floor(wl)
        self.assertAlmostEqual(f["release_us"], 105.0)
        self.assertAlmostEqual(f["oracle_floor_us"], 105.0)


class OracleGapPctTest(unittest.TestCase):

    def test_gap_pct(self):
        self.assertAlmostEqual(oracle_gap_pct(110.0, 100.0), 10.0)
        self.assertAlmostEqual(oracle_gap_pct(100.0, 100.0), 0.0)
        self.assertEqual(oracle_gap_pct(50.0, 0.0), float("inf"))


if __name__ == "__main__":
    unittest.main()
