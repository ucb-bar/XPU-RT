"""Regression tests for `gen_root`, which selects the profile tree timings come
from.

THE BUG: `hardware.profile.gen_root` was parsed out of the schedule JSON into a
config dict and then never read by anything. `find_profile_csv` hardcoded
`<repo>/gen/profile`. So a config naming an alternate profile tree silently read
the default one, and a run could be labelled with one timing basis while
actually using another -- the failure mode where the numbers look completely
reasonable and are answers to a different question.

It hid for so long because the canonical config's value is literally "gen",
identical to the hardcoded path, so the bug was invisible until a control
experiment pointed at `gen25/` (the same measured cycles converted at the real
25 MHz instead of an assumed 1 GHz) and got 1 GHz latencies back with 25 MHz
periods. A0 came out 2000.546 instead of 2421.843, which is what exposed it.

The scale-invariance test is the substantive one: it establishes that the
headline result does not depend on the fictional 1 GHz clock, because scaling
every duration and every period by the same factor leaves the schedule a pure
time-rescaling and every RATE unchanged.
"""

from __future__ import annotations

import json
import os
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_XPURT = os.path.dirname(_HERE)
_REPO = os.path.dirname(_XPURT)
sys.path.insert(0, _REPO)
sys.path.insert(0, _XPURT)

from profile_loader import find_profile_csv  # noqa: E402

CANON = os.path.join(_REPO, "data", "toplevel", "freshness_canon_300ms.json")
SCALED = os.path.join(_REPO, "data", "toplevel", "freshness_canon_25mhz.json")
K = 40.0  # 1000 MHz / 25 MHz

LOOKUP = dict(model="mlp_control", target="firesim_gemmini_opu", hw="gemmini",
              basename="mlp_control.int8", topo_tag="topo_0")


class GenRootSelectsTheTree(unittest.TestCase):
    """No `skipUnless`: both trees this class resolves against are COMMITTED.

    It used to skip on `gen/profile` being absent, dating from when the profile
    tree was generated locally. `gen/profile/gemmini/firesim_gemmini_opu/
    mlp_control/...` and its `gen25/` counterpart are both tracked now (six
    files each, a few kB), so the skip could only ever fire on a broken
    checkout -- and a regression test that skips itself when its data is missing
    is a regression test that reports success for the one situation it cannot
    check. `setUpClass` asserts the data is there instead, so a missing fixture
    is an error rather than a silence.

    The K1/`spacemit_x60` side of this loader -- a different directory depth and
    a `results.csv` with an extra column -- is covered by
    `test_k1_profile_fixture.py`, against a committed fixture added for the same
    reason.
    """

    @classmethod
    def setUpClass(cls):
        for tree in ("gen", "gen25"):
            path = os.path.join(_REPO, tree, "profile", "gemmini",
                                "firesim_gemmini_opu", "mlp_control",
                                "mlp_control.int8", "topo_0", "results.csv")
            assert os.path.exists(path), (
                f"committed profile fixture missing: {path}. It is tracked; "
                f"restore it rather than skipping these tests.")

    def test_default_resolves_under_gen(self):
        p = find_profile_csv(_REPO, **LOOKUP)
        self.assertIsNotNone(p)
        self.assertIn(os.path.join("gen", "profile"), p)

    def test_explicit_gen_root_is_honoured_not_ignored(self):
        """The regression itself: passing an alternate tree must change the
        resolved path. Before the fix this returned the gen/ path."""
        p = find_profile_csv(_REPO, gen_root="gen25", **LOOKUP)
        self.assertIsNotNone(p)
        self.assertIn(os.path.join("gen25", "profile"), p)
        self.assertNotIn(os.path.join("gen", "profile", "gemmini"), p)

    def test_nonexistent_gen_root_returns_none_rather_than_falling_back(self):
        """Silently falling back to the default tree is what made the original
        bug undetectable. Returning None lets strict mode raise."""
        self.assertIsNone(
            find_profile_csv(_REPO, gen_root="gen_definitely_not_here", **LOOKUP))

    def test_gen_root_is_threaded_through_every_loader_entry_point(self):
        import inspect
        import profile_loader as P
        for fn in (P.find_profile_csv, P.load_profiled_processing_times,
                   P._load_all_topo_profiles):
            self.assertIn(
                "gen_root", inspect.signature(fn).parameters,
                f"{fn.__name__} cannot forward gen_root, so some path still "
                f"resolves the hardcoded tree",
            )


@unittest.skipUnless(
    os.path.exists(os.path.join(_REPO, "gen25", "profile"))
    and os.path.exists(SCALED),
    "25 MHz control tree/config absent")
class ClockScaleInvariance(unittest.TestCase):
    """The 1 GHz clock is not the hardware's (bitstreams close at 25-30 MHz), so
    every millisecond figure is measured cycles under a counterfactual clock.

    That objection is weaker than it looks, and this pins why: if durations AND
    periods both scale by k, the schedule is a pure time-rescaling, so all rates
    are identical. The result therefore holds at the true frequency, describing a
    robot running k times slower -- 400 ms control period, 2 s perception. What
    the result depends on is the RATIO of compute time to period, not the clock.
    """

    def setUp(self):
        with open(CANON) as f:
            self.canon = json.load(f)
        with open(SCALED) as f:
            self.scaled = json.load(f)

    def test_every_time_quantity_scaled_by_exactly_k(self):
        self.assertAlmostEqual(self.scaled["epoch"]["length_ms"],
                               self.canon["epoch"]["length_ms"] * K, places=6)
        for name, info in self.canon["networks"].items():
            s = self.scaled["networks"][name]
            for key in ("period", "window_duration"):
                if info.get(key) is not None:
                    self.assertAlmostEqual(
                        float(s[key]), float(info[key]) * K, places=6,
                        msg=f"{name}.{key} is not scaled by {K}")
            # Instance counts must NOT scale -- the same number of releases over
            # a k-times-longer epoch is what makes it a pure rescaling.
            self.assertEqual(s.get("num_instances"), info.get("num_instances"))

    def test_scaled_config_reads_the_scaled_tree(self):
        self.assertEqual(self.scaled["hardware"]["profile"]["gen_root"], "gen25")

    def test_A0_scales_by_exactly_k(self):
        """A0 is the measured uncontended input-age ceiling and anchors the whole
        phi grid. If it did not scale exactly, the two experiments would not be
        at corresponding operating points and no comparison would be valid."""
        from benchmarks.freshness_eval.run import compute_a0
        from freshness import freshness_edges_from_config

        def a0(cfg):
            edge = freshness_edges_from_config(cfg, freshness_window_override=1.0)[0]
            return compute_a0(cfg, epoch_ms=float(cfg["epoch"]["length_ms"]),
                              edge=edge)

        base, scaled = a0(self.canon), a0(self.scaled)
        self.assertAlmostEqual(scaled["A0_realized"],
                               base["A0_realized"] * K, places=6)
        self.assertAlmostEqual(scaled["producer_latency_ms"],
                               base["producer_latency_ms"] * K, places=6)
        self.assertAlmostEqual(scaled["consumer_latency_ms"],
                               base["consumer_latency_ms"] * K, places=6)

    def test_the_scaled_workload_is_a_slower_robot_not_a_different_one(self):
        """Utilisation is what the schedule actually responds to, and it must be
        identical -- otherwise the rescaling changed the problem."""
        from benchmarks.freshness_eval.run import compute_a0
        from freshness import freshness_edges_from_config

        def util(cfg):
            edge = freshness_edges_from_config(cfg, freshness_window_override=1.0)[0]
            a = compute_a0(cfg, epoch_ms=float(cfg["epoch"]["length_ms"]), edge=edge)
            return (a["producer_latency_ms"] / a["producer_period_ms"],
                    a["consumer_latency_ms"] / a["consumer_period_ms"])

        for base_u, scaled_u in zip(util(self.canon), util(self.scaled)):
            self.assertAlmostEqual(base_u, scaled_u, places=9)


if __name__ == "__main__":
    unittest.main(verbosity=2)
