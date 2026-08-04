"""Tests for the switching-headroom bound, and for the two disciplines it rests on.

The bound is the number the adaptive phase has to be interpreted against, so the
things worth pinning are not "the code runs" but the specific claims:

  1. Over bursts 0..3 switching gains EXACTLY ZERO at every validity target -- a
     single static rung is optimal everywhere. This is a negative result and it
     is easy to lose accidentally by adding a rung or changing a tie-break.
  2. Over bursts 0..4 switching gains EXACTLY ONE soft instance, and only for
     targets at or below C1's B=3 validity. The whole value of adaptation on this
     workload is that one instance.
  3. A rung whose schedule overruns the epoch is never selectable. Dropping that
     rule does not merely add a bad option -- it INFLATES the static baseline to
     9/10 on the strength of an 806 ms schedule in a 300 ms epoch, which would
     make every adaptive-vs-static comparison meaningless. Pinned by
     `test_ignoring_epoch_fit_would_corrupt_the_static_baseline`.
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

from benchmarks.freshness_eval.headroom import (  # noqa: E402
    DEFAULT_TARGETS,
    LADDER_RUNGS,
    Cell,
    bound,
    build_table,
    max_gain,
    monotonicity_violations,
)

# The measured canonical ladder at phi = A0+20 (delta 20), from
# results/freshness_cand/*/aggregate.csv. Inlined so these tests pin the CLAIMS
# rather than skipping when the results directory is absent -- a test that skips
# when its data is missing protects nothing.
#
# Written as EXACT RATIONALS, not rounded decimals, and that is not fussiness:
# transcribing 26/30 as 0.867 rounds it UP across the 0.867 target and silently
# changes which rung the bound reports as optimal. The first version of this file
# did exactly that and asserted a rung the measurement does not support.
# `test_inlined_table_matches_results_on_disk` compares to full float precision
# to keep it honest.
#   (admissible, validity, soft_completed, soft_offered)
MEASURED = {
    ("C0", 0): (True,  14 / 15, 0, 0), ("C0", 1): (True,  19 / 30, 1, 1),
    ("C0", 2): (True,   2 / 5,  2, 2), ("C0", 3): (False,  9 / 41, 3, 3),
    ("C0", 4): (False,  0.0,    4, 4),
    ("C1", 0): (True,  14 / 15, 0, 0), ("C1", 1): (True,   9 / 10, 1, 1),
    ("C1", 2): (True,  13 / 15, 2, 2), ("C1", 3): (True,   5 / 6,  3, 3),
    ("C1", 4): (False,  5 / 31, 3, 4),
    ("C2", 0): (True,  14 / 15, 0, 0), ("C2", 1): (True,   9 / 10, 1, 1),
    ("C2", 2): (True,  13 / 15, 2, 2), ("C2", 3): (True,  13 / 15, 2, 3),
    ("C2", 4): (True,  13 / 15, 2, 4),
    ("C3", 0): (True,  14 / 15, 0, 0), ("C3", 1): (True,   9 / 10, 1, 1),
    ("C3", 2): (True,   9 / 10, 1, 2), ("C3", 3): (True,   9 / 10, 1, 3),
    ("C3", 4): (True,   9 / 10, 1, 4),
}


def _table(overrides=None):
    t = {k: Cell(*v) for k, v in MEASURED.items()}
    for k, v in (overrides or {}).items():
        t[k] = Cell(*v)
    return t


LOW = (0, 1, 2, 3)
FULL = (0, 1, 2, 3, 4)


class SwitchingGainsNothingBelowB4(unittest.TestCase):
    """Finding 1: restricted to 0..3, one static rung is optimal at every target."""

    def test_gain_is_zero_at_every_target(self):
        for t in DEFAULT_TARGETS:
            r = bound(_table(), target=t, bursts=LOW)
            if r.gain is None:
                continue
            self.assertEqual(
                r.gain, 0,
                f"target {t}: adaptive {r.adaptive_utility} vs static "
                f"{r.static_utility} ({r.static_rung}) -- switching is supposed "
                f"to gain nothing over bursts 0..3")

    def test_max_gain_is_zero(self):
        self.assertEqual(max_gain(_table(), bursts=LOW), 0)

    def test_the_optimal_static_rung_tightens_as_the_target_rises(self):
        """Not a gain, but the mechanism is still doing something: a stricter
        target forces a higher rung and costs utility."""
        self.assertEqual(bound(_table(), target=0.50, bursts=LOW).static_rung, "C1")
        self.assertEqual(bound(_table(), target=0.85, bursts=LOW).static_rung, "C2")
        # Strictly above C2's 13/15, so only C3 (9/10) survives.
        self.assertEqual(bound(_table(), target=0.88, bursts=LOW).static_rung, "C3")
        utils = [bound(_table(), target=t, bursts=LOW).static_utility
                 for t in (0.50, 0.85, 0.88)]
        self.assertEqual(utils, sorted(utils, reverse=True),
                         "utility must fall as the validity target rises")


class SwitchingGainsExactlyOneInstanceOverTheFullRange(unittest.TestCase):
    """Finding 2: the entire value of adaptation here is one soft instance."""

    def test_max_gain_is_exactly_one(self):
        self.assertEqual(max_gain(_table(), bursts=FULL), 1)

    def test_the_gain_is_one_only_at_loose_targets(self):
        # C1's validity at B=3 is 0.833; at or below it, adaptive may run C1
        # there (3/3 soft) while static must fall back to C2 (2/3).
        for t in (0.50, 0.60, 0.75, 0.80, 0.833):
            self.assertEqual(bound(_table(), target=t, bursts=FULL).gain, 1,
                             f"expected +1 at target {t}")
        for t in (0.85, 0.867, 0.90):
            self.assertEqual(bound(_table(), target=t, bursts=FULL).gain, 0,
                             f"expected +0 at target {t}")

    def test_the_gain_comes_from_burst_3(self):
        r = bound(_table(), target=0.80, bursts=FULL)
        self.assertEqual(r.static_rung, "C2")
        self.assertEqual(r.adaptive_picks[3], "C1",
                         "the +1 is adaptive running C1 at B=3 where the safe "
                         "static choice must run C2")
        self.assertEqual(r.adaptive_utility, 8)
        self.assertEqual(r.static_utility, 7)

    def test_removing_b4_removes_the_entire_gain(self):
        """The gain exists only because B=4 disqualifies C1 as a static choice."""
        self.assertEqual(max_gain(_table(), bursts=FULL), 1)
        self.assertEqual(max_gain(_table(), bursts=LOW), 0)


class EpochOverrunIsNotASelectableOption(unittest.TestCase):
    """Finding 3, and the discipline the whole bound depends on."""

    def test_an_inadmissible_rung_is_never_picked(self):
        for t in DEFAULT_TARGETS:
            r = bound(_table(), target=t, bursts=FULL)
            self.assertNotEqual(
                r.adaptive_picks[4], "C1",
                f"target {t}: C1 overruns the epoch at B=4 (806 ms in a 300 ms "
                f"budget) and must not be selectable there")
            self.assertNotIn(r.static_rung, ("C0",),
                             "C0 overruns at B>=3 and cannot be a safe static choice")

    def test_ignoring_epoch_fit_would_corrupt_the_static_baseline(self):
        """The failure being prevented, stated as a measurement.

        With the epoch rule, the best safe static rung over 0..4 at a loose
        target is C2 at 7/10. If overrunning schedules counted as deployable, C1
        would qualify and report 9/10 -- a baseline built on a schedule that runs
        2.7x the epoch budget. Every adaptive-vs-static number would then be
        measured against a policy that cannot run.
        """
        honest = bound(_table(), target=0.10, bursts=FULL)
        self.assertEqual(honest.static_rung, "C2")
        self.assertEqual(honest.static_utility, 7)

        pretend_all_fit = _table({k: (True,) + MEASURED[k][1:] for k in MEASURED})
        corrupt = bound(pretend_all_fit, target=0.10, bursts=FULL)
        self.assertEqual(corrupt.static_rung, "C1")
        self.assertEqual(corrupt.static_utility, 9)
        self.assertGreater(corrupt.static_utility, honest.static_utility,
                           "ignoring epoch fit must inflate the baseline -- that "
                           "is why the rule exists")


class InfeasibleIsNotTheSameAsZeroGain(unittest.TestCase):
    def test_gain_is_none_when_no_rung_meets_the_target(self):
        r = bound(_table(), target=0.99, bursts=FULL)
        self.assertIsNone(r.adaptive_utility)
        self.assertIsNone(r.static_rung)
        self.assertIsNone(r.gain, "unreachable target must not report a gain of 0")
        self.assertIn("--", r.adaptive_picks)

    def test_a_target_reachable_at_some_bursts_only_is_still_infeasible(self):
        # 0.929 is above every rung's B>=1 value but below C0/C1's 0.933 at B=0.
        r = bound(_table(), target=0.929, bursts=FULL)
        self.assertEqual(r.adaptive_picks[0], "C3")
        self.assertIsNone(r.adaptive_utility)
        self.assertIsNone(r.gain)


class LadderIsMonotone(unittest.TestCase):
    def test_measured_ladder_has_no_violations(self):
        self.assertEqual(monotonicity_violations(_table(), bursts=FULL), [])

    def test_a_validity_inversion_is_detected(self):
        bad = _table({("C3", 2): (True, 0.50, 1, 2)})
        v = monotonicity_violations(bad, bursts=FULL)
        self.assertTrue(any("validity" in s for s in v), v)

    def test_a_utility_inversion_is_detected(self):
        bad = _table({("C3", 3): (True, 0.900, 3, 3)})
        v = monotonicity_violations(bad, bursts=FULL)
        self.assertTrue(any("utility" in s for s in v), v)

    def test_inadmissible_cells_are_not_reported_as_violations(self):
        """C1 at B=4 has a much lower rate than C0's -- but both are measured over
        different-length traces, so comparing them is meaningless, not a break."""
        v = monotonicity_violations(_table(), bursts=(4,))
        self.assertEqual(v, [], f"inadmissible cells must be skipped, got {v}")


class TableConstructionRefusesToGuess(unittest.TestCase):
    def _row(self, pol, b, ovr, soft, fits="True", delta=20.0):
        return {"policy": pol, "contention_level": str(b), "delta": str(delta),
                "output_valid_rate": str(ovr), "soft_instances_completed": str(soft),
                "soft_instances_offered": str(b), "fits_in_epoch": fits}

    def test_conflicting_duplicate_rows_raise(self):
        rows = [self._row("cand_c1_defer12", 2, 0.867, 2),
                self._row("cand_c1_defer12", 2, 0.500, 2)]
        with self.assertRaises(ValueError) as cm:
            build_table(rows, LADDER_RUNGS, delta=20.0, bursts=(2,))
        self.assertIn("refusing", str(cm.exception))

    def test_identical_duplicate_rows_are_accepted(self):
        """The candidate sweep is merged from several output dirs that each also
        re-measured the oracle, so exact duplicates are normal and benign."""
        rows = [self._row("cand_c1_defer12", 2, 0.867, 2)] * 3
        t = build_table(rows, LADDER_RUNGS, delta=20.0, bursts=(2,))
        self.assertAlmostEqual(t[("C1", 2)].validity, 0.867)

    def test_rows_at_another_delta_are_not_mixed_in(self):
        rows = [self._row("cand_c1_defer12", 2, 0.867, 2, delta=20.0),
                self._row("cand_c1_defer12", 2, 0.733, 2, delta=5.0)]
        t = build_table(rows, LADDER_RUNGS, delta=20.0, bursts=(2,))
        self.assertAlmostEqual(t[("C1", 2)].validity, 0.867)


class BoundMatchesTheMeasuredSweep(unittest.TestCase):
    """Guards the inlined MEASURED fixture against the real results drifting.

    Skipped when the results directory is absent so the suite stays runnable on a
    fresh clone -- but every claim above is asserted against the inlined table,
    so a skip here never means an unchecked claim.
    """

    def test_inlined_table_matches_results_on_disk(self):
        import glob
        from benchmarks.freshness_eval.headroom import load_rows
        pattern = os.path.join(_REPO, "results", "freshness_cand", "*",
                               "aggregate.csv")
        if not glob.glob(pattern):
            self.skipTest("results/freshness_cand not present")
        rows = load_rows(pattern)
        live = build_table(rows, LADDER_RUNGS, delta=20.0, bursts=FULL)
        self.assertEqual(set(live), set(MEASURED),
                         "the set of measured cells changed")
        for key, cell in live.items():
            exp = MEASURED[key]
            self.assertEqual(cell.admissible, exp[0], f"{key} admissibility")
            # places=9, not 3: a rounded fixture can sit on the other side of a
            # validity target from the real measurement and change the answer.
            self.assertAlmostEqual(cell.validity, exp[1], places=9,
                                   msg=f"{key} validity")
            self.assertEqual(cell.soft_completed, exp[2], f"{key} soft completed")
            self.assertEqual(cell.soft_offered, exp[3], f"{key} soft offered")


if __name__ == "__main__":
    unittest.main(verbosity=2)
