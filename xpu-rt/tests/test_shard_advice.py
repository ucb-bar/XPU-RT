"""Shard advice must be justified by measured scaling, not by op size.

`shard` was in the recommendation vocabulary from the start and no code path
could emit it, because the evidence it needs -- how a dispatch's cost changes
with core count -- was unreachable. `load_profiles` takes one `topo_tag` and
defaults to the 1-core profile, and `profile_loader` discards the `n_cores`
field the profiler already writes. The measurements were on disk the whole time
with no reader.

The discriminating case, and the reason op size alone is not enough evidence:
on this board DroNet's convolutions scale ~7.2x on eight cores at ~90%
efficiency, while every MLP dispatch gets SLOWER with more cores (0.066 ->
0.078 ms) because it is dominated by per-dispatch overhead rather than work. An
advisor that recommended sharding from "this op is big" would get the MLP
exactly backwards, and would do so while sounding confident.
"""

from __future__ import annotations

import os
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_XPURT = os.path.dirname(_HERE)
_REPO = os.path.dirname(_XPURT)
sys.path.insert(0, _REPO)
sys.path.insert(0, _XPURT)

from compile_advice import (  # noqa: E402
    load_profiles_by_cores, shard_advice,
)

_GEN = os.path.join(_REPO, "gen")


def _rec(did, ms, n_cores):
    return {"dispatch_id": did, "median_ms": ms, "n_cores": n_cores,
            "module_name": f"m$async_dispatch_{did}_conv_1x1x1"}


def _profiles(costs_by_cores):
    """{n_cores: {did: median_ms}} -> the shape shard_advice consumes."""
    return {n: {did: _rec(did, ms, n) for did, ms in d.items()}
            for n, d in costs_by_cores.items()}


class ScalingEvidenceDecides(unittest.TestCase):

    def test_an_op_that_scales_is_recommended(self):
        p = _profiles({1: {0: 20.0}, 2: {0: 10.5}, 4: {0: 5.4}})
        adv = shard_advice("m", p, free_slot_ms=10.0)
        self.assertEqual(len(adv), 1)
        a = adv[0]
        self.assertEqual(a.recommendation, "shard")
        self.assertEqual(a.constraints["n_cores"], 4)
        self.assertTrue(a.evidence.extra["fits_slot_after"])

    def test_an_op_that_does_not_scale_is_refused_with_its_numbers(self):
        """Refusal must carry the evidence, so a later round does not retry."""
        p = _profiles({1: {0: 20.0}, 2: {0: 19.5}, 4: {0: 21.0}})
        adv = shard_advice("m", p, free_slot_ms=10.0)
        self.assertEqual(len(adv), 1)
        a = adv[0]
        self.assertNotEqual(a.recommendation, "shard")
        self.assertIn("detail", a.evidence.extra)
        self.assertIn("19.5", a.evidence.extra["detail"])

    def test_speedup_bought_at_terrible_efficiency_is_refused(self):
        """8x the cores for 2x the speed is not a shard worth making.

        It occupies eight cores to save half the time of one, which on a
        multi-model workload takes those cores away from everything else.
        """
        p = _profiles({1: {0: 20.0}, 8: {0: 10.0}})   # 2x on 8 cores, eff 0.25
        adv = shard_advice("m", p, free_slot_ms=10.0)
        self.assertNotEqual(adv[0].recommendation, "shard")

    def test_an_op_that_already_fits_is_left_alone(self):
        p = _profiles({1: {0: 5.0}, 4: {0: 1.4}})
        self.assertEqual(shard_advice("m", p, free_slot_ms=10.0), [])

    def test_sync_overhead_is_reported_as_evidence(self):
        """n*cost(n) - cost(1): the extra total work the shard costs.

        This is exactly the quantity a summed-cycles objective would reject the
        change for, so the advisor has to surface it rather than hide it.
        """
        p = _profiles({1: {0: 20.0}, 4: {0: 6.0}})    # 4*6 - 20 = 4 ms extra
        a = shard_advice("m", p, free_slot_ms=10.0)[0]
        self.assertAlmostEqual(a.evidence.extra["sync_overhead_us"], 4000.0,
                               places=1)

    def test_the_fastest_qualifying_core_count_wins(self):
        p = _profiles({1: {0: 20.0}, 2: {0: 10.2}, 4: {0: 5.3}, 8: {0: 2.9}})
        a = shard_advice("m", p, free_slot_ms=10.0)[0]
        self.assertEqual(a.constraints["n_cores"], 8)

    def test_no_single_core_profile_means_no_advice(self):
        """Without a 1-core baseline there is no speedup to measure."""
        p = _profiles({4: {0: 5.0}})
        self.assertEqual(shard_advice("m", p, free_slot_ms=10.0), [])


class AgainstRealK1Measurements(unittest.TestCase):
    """Runs on whatever profiles are present; skips when they are not."""

    def _load(self, model, basename):
        p = load_profiles_by_cores(_GEN, "spacemit_x60", model, basename, "RVV")
        if 1 not in p or len(p) < 2:
            self.skipTest(f"no multi-core profile for {model} under {_GEN}")
        return p

    def test_dronet_convolutions_are_recommended(self):
        p = self._load("dronet", "dronet.q.int8")
        adv = shard_advice("dronet", p, free_slot_ms=10.0)
        sharded = [a for a in adv if a.recommendation == "shard"]
        self.assertGreaterEqual(len(sharded), 3,
                                "DroNet's heavy convs measure ~7x on 8 cores "
                                "and should be recommended")
        for a in sharded:
            self.assertGreater(a.evidence.extra["measured_speedup"], 1.5)

    def test_mlp_is_never_recommended_even_when_it_overruns(self):
        """The case op-size heuristics get wrong.

        A 0.05 ms slot forces every MLP dispatch past gate 1, so the decision
        rests entirely on the scaling evidence -- and the measurement says more
        cores make it slower.
        """
        p = self._load("mlp", "mlp.q.int8")
        adv = shard_advice("mlp", p, free_slot_ms=0.05)
        self.assertTrue(adv, "gate 1 should pass with a 0.05 ms slot")
        self.assertEqual([a for a in adv if a.recommendation == "shard"], [],
                         "MLP dispatches get SLOWER with more cores; "
                         "recommending a shard here would be wrong")


class CoreCountKeying(unittest.TestCase):

    def test_keys_come_from_the_recorded_n_cores(self):
        """Not from the directory name, which is a storage detail."""
        p = load_profiles_by_cores(_GEN, "spacemit_x60", "dronet",
                                   "dronet.q.int8", "RVV")
        if not p:
            self.skipTest("no dronet profiles present")
        for n, recs in p.items():
            for r in recs.values():
                if r.get("n_cores"):
                    self.assertEqual(int(r["n_cores"]), n)
                    break


if __name__ == "__main__":
    unittest.main()
