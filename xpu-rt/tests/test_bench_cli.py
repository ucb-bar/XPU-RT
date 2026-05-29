"""Tests for xpurt.bench solver sweep CLI."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


_TINY_WORKLOAD_SCRIPT = """
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
from workload import Operation, Workload


def make_workload():
    op_a = Operation(processing_times=[10.0, 20.0], operation_name="a")
    op_b = Operation(processing_times=[15.0, 25.0], predecessors=[op_a], operation_name="b")
    return Workload([op_a, op_b], ["CPU_P", "CPU_E"], np.zeros((2, 2), dtype=float))
"""


def _has_solver(name: str) -> bool:
    try:
        import cvxpy as cp  # noqa: F401
        return name in cp.installed_solvers()
    except ImportError:
        return False


class BenchSweepTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # HIGHS ships with cvxpy; if neither HIGHS nor MOSEK is available
        # the bench has nothing to drive.
        if not (_has_solver("HIGHS") or _has_solver("MOSEK")):
            raise unittest.SkipTest("no MILP solver available")

    def test_sweep_emits_per_solver_reports(self):
        from xpurt.bench import sweep
        with tempfile.TemporaryDirectory() as td:
            script = os.path.join(td, "wl.py")
            with open(script, "w") as f:
                f.write(_TINY_WORKLOAD_SCRIPT)
            # Pick whichever solver is installed; prefer HIGHS (open source).
            solver = "HIGHS" if _has_solver("HIGHS") else "MOSEK"
            result = sweep(script, [solver], reps=2, time_limit=10)
        self.assertIn("sweep", result)
        self.assertIn(solver, result["sweep"])
        self.assertEqual(len(result["sweep"][solver]), 2)
        for rep in result["sweep"][solver]:
            if "error" in rep:
                self.fail(f"solve failed: {rep['error']}")
            self.assertEqual(rep["solver_name"], solver)
            self.assertGreater(rep["solve_wall_s"], 0)
            self.assertEqual(rep["n_operations"], 2)


if __name__ == "__main__":
    unittest.main()
