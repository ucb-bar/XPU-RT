"""Regression tests for strict repeated-board evidence aggregation."""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


_ROOT = Path(__file__).resolve().parents[2]
_SPEC = importlib.util.spec_from_file_location(
    "evaluate_exact_cycle_board",
    _ROOT / "scripts" / "evaluate_exact_cycle_board.py")
assert _SPEC and _SPEC.loader
board_eval = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(board_eval)


class BoardEvidenceTests(unittest.TestCase):

    def test_exact_rank_sum_and_pairwise_separation(self):
        self.assertAlmostEqual(
            board_eval._exact_rank_sum_less_p([1.0, 2.0], [3.0, 4.0]),
            1.0 / 6.0)
        comparison = board_eval._distribution_comparison(
            [10.0, 11.0], [7.0, 8.0])
        self.assertEqual(comparison["feedback_runs_below_original_min"], 2)
        self.assertEqual(comparison["pairwise_feedback_superiority_pct"], 100.0)

    def test_stdout_audit_requires_recorded_policy_and_golden(self):
        text = "\n".join([
            "xpurt_runner: sched_policy=SCHED_FIFO priority=80",
            "xpurt: observed_sched_policy=SCHED_FIFO priority=80",
            "xpurt: worker[0] kind=rvv pinned_hart=0 observed_cpu=0 "
            "claims_unbound=0 entries_done=1",
            "=== MODELBLASTER_VERIFY [m] === max_abs_err=0 "
            "max_rel_err=0 n=4 instance=0 ready=1",
        ])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stdout.txt"
            path.write_text(text)
            passed = board_eval._stdout_audit(
                path, 1, False, {"m"}, set(), "SCHED_FIFO")
            rejected = board_eval._stdout_audit(
                path, 1, False, {"m"}, set(), "SCHED_OTHER")
        self.assertTrue(passed["pass"])
        self.assertFalse(rejected["pass"])
        self.assertFalse(rejected["runner_policy_valid"])


if __name__ == "__main__":
    unittest.main()
