"""Tests for the axis-B backend availability/aggregation helpers (_sched_eval)."""

from __future__ import annotations

import os
import sys
import unittest

SCRIPTS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))
sys.path.insert(0, SCRIPTS)

import _sched_eval as ev


class SchedEvalTests(unittest.TestCase):

    def test_available_backends_finds_profiled_only(self):
        avail = ev.available_backends()
        # gemmini_q31 + V256D128_rvv are profiled for all 3 models; scalar is not.
        self.assertIn("gemmini_q31", avail)
        self.assertIn("V256D128_rvv", avail)
        self.assertNotIn("scalar", avail)

    def test_available_backends_guards_missing_model(self):
        # a nonexistent target yields nothing (no false positives)
        self.assertEqual(ev.available_backends(target="no_such_target"), [])

    def test_solver_tag_matches_runner_output_naming(self):
        self.assertEqual(ev.report_path("w", "decomposed", None),
                         os.path.join(ev.REPO, "schedules", "scheduled_w_decomposed_profiled_report.json"))
        self.assertEqual(ev.report_path("w", "greedy", None),
                         os.path.join(ev.REPO, "schedules", "scheduled_w_greedy_profiled_report.json"))
        # milp + registry scheduler tags with the scheduler name (non-mosek)
        self.assertEqual(ev.report_path("w", "milp", "heft"),
                         os.path.join(ev.REPO, "schedules", "scheduled_w_heft_profiled_report.json"))
        # milp + mosek keeps no infix
        self.assertEqual(ev.report_path("w", "milp", "mosek"),
                         os.path.join(ev.REPO, "schedules", "scheduled_w_profiled_report.json"))


if __name__ == "__main__":
    unittest.main()
