"""Tests for xpu-rt/fusion_advisor.py."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fusion_advisor import advise, FusionRecommendation  # noqa: E402


def _synthetic_report(*, n_ops=100, lt_1k=80, lt_10k=15, lt_100k=4, lt_1M=1, ge_1M=0,
                      p50=0.5, p99=20.0, util_p_busy=0.4, util_e_busy=0.3) -> dict:
    return {
        "schema_version": 1,
        "solver_name": "MOSEK",
        "solver_status": "optimal",
        "solve_wall_s": 0.5,
        "n_operations": n_ops,
        "n_combinations": 2,
        "n_resources_by_kind": {"CPU_P": 1, "CPU_E": 1},
        "makespan_cycles": 100.0,
        "utilization": {
            "CPU_P#0": {"busy_cycles": 100 * util_p_busy,
                        "idle_cycles": 100 * (1 - util_p_busy),
                        "frac_busy": util_p_busy},
            "CPU_E#0": {"busy_cycles": 100 * util_e_busy,
                        "idle_cycles": 100 * (1 - util_e_busy),
                        "frac_busy": util_e_busy},
        },
        "granularity": {
            "p50": p50, "p90": p50 * 2, "p95": p50 * 5, "p99": p99,
            "mean": p50, "max": p99,
            "buckets": {"lt_1k": lt_1k, "lt_10k": lt_10k,
                        "lt_100k": lt_100k, "lt_1M": lt_1M, "ge_1M": ge_1M},
        },
        "dispatch_durations": [p50] * n_ops,
        "critical_path": 50.0,
        "cross_device_transitions": 5,
        "deadline_miss_count": 0,
        "fusion_applied": False,
        "fusion_map": None,
        "git_sha": "deadbeef",
        "captured_at": "2026-05-29T20:00:00+00:00",
    }


class ThresholdRecTests(unittest.TestCase):

    def test_small_op_tail_gets_threshold_rec(self):
        # 80 of 100 ops in lt_1k → threshold should fire.
        rep = _synthetic_report(lt_1k=80, lt_10k=15, lt_100k=4)
        recs = advise(rep)
        kinds = [r.kind for r in recs]
        self.assertIn("threshold", kinds)
        thr = next(r for r in recs if r.kind == "threshold")
        self.assertIn("1000", thr.target)   # 1k threshold for >20% in lt_1k

    def test_long_ops_no_threshold(self):
        # All ops in lt_1M / ge_1M → no fusion possible.
        rep = _synthetic_report(lt_1k=0, lt_10k=0, lt_100k=0, lt_1M=10, ge_1M=90)
        recs = advise(rep)
        kinds = [r.kind for r in recs]
        self.assertNotIn("threshold", kinds)


class ChainFusionTests(unittest.TestCase):

    def test_high_idle_emits_chain_rec(self):
        # 70% idle on CPU_P → chain fusion recommendation.
        rep = _synthetic_report(p50=0.5, p99=2.0, util_p_busy=0.3, util_e_busy=0.3)
        recs = advise(rep)
        kinds = [r.kind for r in recs]
        self.assertIn("chain", kinds)

    def test_well_utilized_no_chain_rec(self):
        # 80% busy → no chain fusion needed.
        rep = _synthetic_report(util_p_busy=0.8, util_e_busy=0.8)
        recs = advise(rep)
        kinds = [r.kind for r in recs]
        self.assertNotIn("chain", kinds)


class APITests(unittest.TestCase):

    def test_top_k_limits_results(self):
        rep = _synthetic_report()
        recs = advise(rep, top_k=1)
        self.assertLessEqual(len(recs), 1)

    def test_dict_or_dataclass_input(self):
        rep = _synthetic_report()
        # Plain dict (as written by write_json).
        recs1 = advise(rep)
        # Re-wrapped (simulate the dataclass via a dict that has to_dict).
        class _W:
            def __init__(self, d): self._d = d
            def to_dict(self): return self._d
        recs2 = advise(_W(rep))
        self.assertEqual(len(recs1), len(recs2))


if __name__ == "__main__":
    unittest.main()
