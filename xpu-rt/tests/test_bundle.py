"""Tests for the candidate-bundle proposer + fusion-hint contract (xpu-rt/bundle.py)."""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import bundle as B


class _Diag:
    """Minimal stand-in for advisor.Diagnosis."""
    def __init__(self, granularity_verdict, deadline_us=None):
        self.granularity_verdict = granularity_verdict
        self.deadline_us = deadline_us


def _report_with_tiny_chain():
    # network "mlp" with a chain of 3 tiny dispatches (0->1->2) + one big op (3)
    return {
        "schema_version": 2,
        "dispatches": [
            {"id": 0, "name": "mlp_control0_dispatch_0", "target": "CPU_P#0",
             "start_us": 0.0, "finish_us": 0.1, "duration_us": 0.1, "deps": []},
            {"id": 1, "name": "mlp_control0_dispatch_1", "target": "CPU_P#0",
             "start_us": 0.1, "finish_us": 0.2, "duration_us": 0.1, "deps": [0]},
            {"id": 2, "name": "mlp_control0_dispatch_2", "target": "CPU_P#0",
             "start_us": 0.2, "finish_us": 0.3, "duration_us": 0.1, "deps": [1]},
            {"id": 3, "name": "mlp_control0_dispatch_3", "target": "CPU_E#0",
             "start_us": 0.3, "finish_us": 5000.0, "duration_us": 4999.7, "deps": [2]},
        ],
        "granularity": {"buckets": {"lt_1k": 3, "lt_10k": 1, "lt_100k": 0, "lt_1M": 0, "ge_1M": 0}},
        "n_operations": 4,
    }


class BundleTests(unittest.TestCase):
    BASELINE = {"solver": "decomposed", "scheduler": None,
                "profile_hw": {"cpu_p": "gemmini_q31", "cpu_e": "V256D128_rvv"}}

    def test_axis_a_and_b_candidates_present_and_tagged(self):
        b = B.propose_bundle(_report_with_tiny_chain(), _Diag("balanced"),
                             baseline=self.BASELINE,
                             available_backends=["gemmini_q31", "V256D128_rvv"])
        kinds = {c["axis"] for c in b["candidates"]}
        self.assertIn("scheduler", kinds)   # axis A
        self.assertIn("backend", kinds)     # axis B
        for c in b["candidates"]:
            self.assertIn(c["realizable_by"], ("xpurt", "modelblaster"))
            if c["axis"] in ("scheduler", "backend"):
                self.assertEqual(c["realizable_by"], "xpurt")

    def test_baseline_config_is_not_proposed_again(self):
        b = B.propose_bundle(_report_with_tiny_chain(), _Diag("balanced"),
                             baseline=self.BASELINE,
                             available_backends=["gemmini_q31", "V256D128_rvv"])
        for c in b["candidates"]:
            same = (c["solver"] == "decomposed" and c.get("scheduler") is None
                    and c["profile_hw"] == self.BASELINE["profile_hw"])
            self.assertFalse(same, "baseline config must not be re-proposed")

    def test_fusion_candidate_only_when_too_fine(self):
        balanced = B.propose_bundle(_report_with_tiny_chain(), _Diag("balanced"),
                                    baseline=self.BASELINE, available_backends=["gemmini_q31"])
        self.assertFalse(any(c["axis"] == "fusion" for c in balanced["candidates"]))
        toofine = B.propose_bundle(_report_with_tiny_chain(), _Diag("too_fine"),
                                   baseline=self.BASELINE, available_backends=["gemmini_q31"])
        fus = [c for c in toofine["candidates"] if c["axis"] == "fusion"]
        self.assertEqual(len(fus), 1)
        self.assertEqual(fus[0]["realizable_by"], "modelblaster")

    def test_fusion_hint_groups_the_tiny_chain(self):
        hints = B.fusion_hints_from_diagnosis(_report_with_tiny_chain(), _Diag("too_fine"))
        self.assertEqual(hints["contract"], "modelblaster.fusion_hints/v1")
        nets = {n["network"]: n for n in hints["networks"]}
        self.assertIn("mlp_control", nets)
        groups = nets["mlp_control"]["fuse_groups"]
        # the 0->1->2 tiny chain should be one fuse group; the big op 3 excluded
        self.assertTrue(any(set(g) == {0, 1, 2} for g in groups), groups)
        self.assertFalse(any(3 in g for g in groups))

    def test_backend_assignments_include_homogeneous_and_hetero(self):
        asg = B.backend_assignments(["gemmini_q31", "V256D128_rvv"])
        self.assertIn({"cpu_p": "gemmini_q31", "cpu_e": "gemmini_q31"}, asg)
        self.assertIn({"cpu_p": "V256D128_rvv", "cpu_e": "V256D128_rvv"}, asg)
        self.assertIn({"cpu_p": "gemmini_q31", "cpu_e": "V256D128_rvv"}, asg)


if __name__ == "__main__":
    unittest.main()
