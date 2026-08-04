"""Tests for the upstream control signal, and the two claims it rests on.

Gate B's negative result had two tangled causes. This module's signal addresses
both, so the tests pin which property does what:

  * MONOTONICITY -- the downstream signal (max input age / phi) is flat at 1.124
    for B = 1..4 under every protective rung, so it cannot discriminate offered
    load. Offered utilisation is strictly increasing in the burst. That is the
    whole reason to change signals, and `test_upstream_signal_is_strictly_monotone`
    is the test that would catch it regressing.
  * NO LAG -- offered work for the epoch about to be scheduled is known at the
    boundary, because admission control happens there. This is not an oracle: it
    is the request count, not the resulting validity, makespan or input age.

The measured payoff is narrow and the tests say so: one soft instance per epoch
spent at B=3, the single burst where the cheap rung both suffices and fits the
epoch. Everything else it buys is SAFETY, which is not visible in a utility
number at all.
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

from benchmarks.freshness_eval.adaptive import (  # noqa: E402
    SELECTOR_RUNGS,
    CellTable,
    run_static,
)
from benchmarks.freshness_eval.upstream import (  # noqa: E402
    BASE_GEMMINI_MS,
    YOLO64_MS,
    candidate_thresholds,
    offered_utilisation,
    run_upstream,
)

PHI = 80.546074
EPOCH = 300.0
C1, C2, C3 = SELECTOR_RUNGS

# Measured cells at phi = A0+20 (see xpu-rt/tests/test_headroom.py for the same
# ladder as exact rationals). max_input_age is the DOWNSTREAM signal, included
# here so the monotonicity contrast can be asserted from data.
CELLS = {
    (C1, 0): (0.933, 0, 290.5, 60.55), (C1, 1): (0.900, 1, 290.5, 90.55),
    (C1, 2): (0.867, 2, 290.5, 90.55), (C1, 3): (0.833, 3, 297.2, 90.55),
    (C1, 4): (0.161, 3, 805.9, 410.55),
    (C2, 0): (0.933, 0, 290.5, 60.55), (C2, 1): (0.900, 1, 290.5, 90.55),
    (C2, 2): (0.867, 2, 290.5, 90.55), (C2, 3): (0.867, 2, 290.5, 90.55),
    (C2, 4): (0.867, 2, 290.5, 90.55),
    (C3, 0): (0.933, 0, 290.5, 60.55), (C3, 1): (0.900, 1, 290.5, 90.55),
    (C3, 2): (0.900, 1, 290.5, 90.55), (C3, 3): (0.900, 1, 290.5, 90.55),
    (C3, 4): (0.900, 1, 290.5, 90.55),
}


def _rows():
    out = []
    for (pol, b), (ovr, soft, mk, age) in CELLS.items():
        n = 30
        out.append({
            "policy": pol, "contention_level": str(b), "freshness_window": str(PHI),
            "delta": "20.0", "A0": "60.546074",
            "output_valid_rate": str(ovr), "valid_count": str(int(round(ovr * n))),
            "total_consumer_invocations": str(n),
            "freshness_success_rate": str(ovr), "deadline_success_rate": "1.0",
            "max_input_age": str(age), "soft_instances_completed": str(soft),
            "soft_instances_offered": str(b), "makespan_ms": str(mk),
            "fits_in_epoch": str(mk <= EPOCH), "seed": "0",
        })
    return out


def _table():
    return CellTable(_rows(), PHI)


class UpstreamSignalIsInformative(unittest.TestCase):
    def test_upstream_signal_is_strictly_monotone(self):
        """The property the downstream signal lacks, and the reason for this module."""
        vals = [offered_utilisation(b, EPOCH) for b in range(5)]
        self.assertEqual(vals, sorted(vals))
        for a, b in zip(vals, vals[1:]):
            self.assertGreater(b - a, 0.2,
                               "consecutive bursts must be clearly separable")

    def test_downstream_signal_is_flat_by_contrast(self):
        """Asserted from the same measured cells, so the contrast is not rhetorical."""
        for rung in (C2, C3):
            ages = {CELLS[(rung, b)][3] for b in (1, 2, 3, 4)}
            self.assertEqual(len(ages), 1,
                             f"{rung}: downstream age should be flat, got {ages}")

    def test_utilisation_matches_the_profile_table(self):
        """Against the plan's F1 figures: 43 / 65 / 88 / 110 / 132 percent."""
        expected = [0.427, 0.651, 0.875, 1.099, 1.323]
        for b, e in enumerate(expected):
            self.assertAlmostEqual(offered_utilisation(b, EPOCH), e, places=3)

    def test_utilisation_is_built_from_named_costs(self):
        self.assertAlmostEqual(offered_utilisation(1, EPOCH) * EPOCH,
                               BASE_GEMMINI_MS + YOLO64_MS, places=6)


class SelectionUsesTheCurrentEpoch(unittest.TestCase):
    def test_no_lag_the_current_burst_decides(self):
        """A step into high contention is handled in the epoch it arrives, which
        is the whole difference from the downstream selector."""
        res = run_upstream(_table(), [0, 0, 4, 4], "step", EPOCH,
                           escalate_at=(4, 5))
        self.assertEqual([e.candidate_id for e in res.epochs],
                         [C1, C1, C2, C2])
        self.assertTrue(all(e.makespan_ms <= EPOCH for e in res.epochs),
                        "no epoch may overrun -- that was the downstream failure")

    def test_the_downstream_failure_does_not_recur(self):
        """The concrete regression: the downstream selector ran C1 during the
        first B=4 epoch and took 805.9 ms in a 300 ms budget."""
        res = run_upstream(_table(), [0, 0, 4, 4, 4, 4, 0, 0, 0, 0], "step",
                           EPOCH, escalate_at=(4, 5))
        offenders = [(e.epoch, e.candidate_id, e.makespan_ms)
                     for e in res.epochs if e.makespan_ms > EPOCH]
        self.assertEqual(offenders, [])

    def test_thresholds_must_be_ascending(self):
        with self.assertRaises(ValueError):
            run_upstream(_table(), [0], "x", EPOCH, escalate_at=(4, 2))

    def test_switch_is_flagged_only_on_a_change(self):
        res = run_upstream(_table(), [0, 0, 4, 4], "x", EPOCH, escalate_at=(4, 5))
        self.assertEqual([e.switched for e in res.epochs],
                         [False, False, True, False])


class GainIsOnePerEpochAtBurst3(unittest.TestCase):
    """The measured payoff, stated in the form that actually generalises.

    headroom.py bounds the gain at +1 over the burst GRID, where each burst is
    visited once. On a trajectory the gain scales with the number of epochs spent
    at B=3 -- the one burst where the cheap rung both suffices and fits. The ramp
    visits B=3 twice and gains exactly +2, which is not a violation of the bound
    but the same bound applied per visit.
    """

    RAMP = [0, 1, 2, 3, 4, 4, 3, 2, 1, 0]

    def test_ramp_gains_two_because_it_visits_burst_3_twice(self):
        t = _table()
        up = run_upstream(t, self.RAMP, "ramp", EPOCH, escalate_at=(4, 5))
        st = run_static(t, C2, self.RAMP, "ramp", EPOCH)
        gain = (up.summary()["soft_completed"] - st.summary()["soft_completed"])
        self.assertEqual(gain, 2)
        self.assertEqual(sum(1 for b in self.RAMP if b == 3), 2)

    def test_the_gain_occurs_only_at_burst_3(self):
        t = _table()
        up = run_upstream(t, self.RAMP, "ramp", EPOCH, escalate_at=(4, 5))
        st = run_static(t, C2, self.RAMP, "ramp", EPOCH)
        for a, b in zip(up.epochs, st.epochs):
            d = a.soft_completed - b.soft_completed
            if a.offered_burst == 3:
                self.assertEqual(d, 1, f"epoch {a.epoch} at B=3")
            else:
                self.assertEqual(d, 0, f"epoch {a.epoch} at B={a.offered_burst}")

    def test_a_trajectory_without_burst_3_gains_nothing(self):
        """step, oscillate and sustained visit only B=0 and B=4, so there is no
        epoch where the cheap rung is both sufficient and admissible."""
        t = _table()
        for traj in ([0, 0, 4, 4, 4, 4, 0, 0, 0, 0],
                     [0, 4] * 5,
                     [4] * 10):
            up = run_upstream(t, traj, "x", EPOCH, escalate_at=(4, 5))
            st = run_static(t, C2, traj, "x", EPOCH)
            self.assertEqual(
                up.summary()["soft_completed"], st.summary()["soft_completed"],
                f"expected no gain on {traj}")


class DegenerateThresholdsCollapseToStatics(unittest.TestCase):
    """A selector can fail by never escalating or by always escalating, and the
    sweep must contain both so neither looks like adaptation."""

    RAMP = [0, 1, 2, 3, 4, 4, 3, 2, 1, 0]

    def test_never_escalating_is_the_unprotected_static_and_is_unsafe(self):
        t = _table()
        res = run_upstream(t, self.RAMP, "ramp", EPOCH, escalate_at=(5, 5))
        self.assertEqual({e.candidate_id for e in res.epochs}, {C1})
        self.assertTrue(any(e.makespan_ms > EPOCH for e in res.epochs),
                        "never escalating must reproduce C1's epoch overrun")

    def test_always_escalating_is_the_most_conservative_static(self):
        t = _table()
        res = run_upstream(t, self.RAMP, "ramp", EPOCH, escalate_at=(0, 0))
        self.assertEqual({e.candidate_id for e in res.epochs}, {C3})
        st = run_static(t, C3, self.RAMP, "ramp", EPOCH)
        self.assertEqual(res.summary()["soft_completed"],
                         st.summary()["soft_completed"])
        self.assertEqual(res.summary()["switch_count"], 0,
                         "a selector pinned at one rung is not adapting")

    def test_the_sweep_includes_both_degenerate_ends(self):
        ths = candidate_thresholds([0, 1, 2, 3, 4], len(SELECTOR_RUNGS))
        self.assertIn((0, 0), ths)
        self.assertIn((5, 5), ths)


class WinningThresholdIsNarrow(unittest.TestCase):
    """The honest caveat: the best setting is a two-point calibration.

    Only escalate-at-4 realises the gain. Escalating earlier sheds work that did
    not need shedding; escalating later reproduces C1's overrun. Reporting the
    best threshold without this would present a fitted parameter as a result.
    """

    RAMP = [0, 1, 2, 3, 4, 4, 3, 2, 1, 0]

    def test_only_escalating_at_four_is_both_safe_and_best(self):
        t = _table()
        best, results = None, {}
        for th in [(1, 5), (2, 5), (3, 5), (4, 5), (5, 5)]:
            res = run_upstream(t, self.RAMP, "ramp", EPOCH, escalate_at=th)
            ok = all(e.makespan_ms <= EPOCH for e in res.epochs)
            results[th] = (ok, res.summary()["soft_completed"])
        self.assertFalse(results[(5, 5)][0], "escalate-at-5 never fires -> unsafe")
        safe = {th: soft for th, (ok, soft) in results.items() if ok}
        self.assertEqual(max(safe, key=safe.get), (4, 5))
        # And it is strictly better than escalating one step earlier, so the
        # optimum is a single point on this grid rather than a plateau.
        self.assertGreater(safe[(4, 5)], safe[(3, 5)])
