"""Tests for the epoch-level candidate selector.

The tests that matter most are the CAUSALITY ones. An adaptive policy compared
against static baselines is trivially made to win by letting it see the epoch it
is choosing for, and that mistake produces a beautiful, entirely fake result.
So: `decide()` must refuse an observation from its own epoch or later, `replay`
must refuse lag < 1, and the recorded row must carry the source epoch so an
auditor can check the lag independently rather than trusting the code.

The rest pin the anti-chatter brakes, each of which can silently make the
selector inert (never switching) or useless (switching constantly).
"""

from __future__ import annotations

import os
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_XPURT = os.path.dirname(_HERE)
sys.path.insert(0, _XPURT)

from selector import (  # noqa: E402
    BOOTSTRAP,
    CSV_COLUMNS,
    DEESCALATED,
    ESCALATED,
    HELD_AT_CEILING,
    HELD_COOLDOWN,
    HELD_HYSTERESIS,
    HELD_MIN_RESIDENCY,
    HELD_NO_TRIGGER,
    CandidateLevel,
    Selector,
    SelectorConfig,
    replay,
)

PHI = 100.0


def _cfg(min_residency=0, cooldown=0):
    return SelectorConfig(
        levels=(
            CandidateLevel("C0", 0, entry_risk=0.0, exit_risk=-1.0, intent="nominal"),
            CandidateLevel("C1", 1, entry_risk=0.7, exit_risk=0.5, intent="protect"),
            CandidateLevel("C2", 2, entry_risk=1.0, exit_risk=0.8, intent="degraded"),
        ),
        min_residency_epochs=min_residency,
        cooldown_epochs=cooldown,
    )


class Causality(unittest.TestCase):
    def test_observation_from_own_epoch_is_rejected(self):
        s = Selector(_cfg())
        with self.assertRaises(ValueError) as cm:
            s.decide(3, 90.0, PHI, observation_from_epoch=3)
        self.assertIn("oracle", str(cm.exception))

    def test_observation_from_a_future_epoch_is_rejected(self):
        s = Selector(_cfg())
        with self.assertRaises(ValueError):
            s.decide(3, 90.0, PHI, observation_from_epoch=4)

    def test_replay_rejects_zero_and_negative_lag(self):
        for lag in (0, -1):
            with self.assertRaises(ValueError) as cm:
                replay(_cfg(), [10.0, 20.0], PHI, lag=lag)
            self.assertIn("lag", str(cm.exception))

    def test_replay_lag_is_recorded_and_auditable(self):
        """An auditor must be able to verify the lag from the log alone."""
        ages = [10.0, 20.0, 30.0, 40.0]
        sel = replay(_cfg(), ages, PHI, lag=1)
        rows = sel.log
        self.assertIsNone(rows[0].observation_from_epoch, "epoch 0 has no history")
        self.assertEqual(rows[0].reason, BOOTSTRAP)
        for r in rows[1:]:
            self.assertEqual(r.observation_from_epoch, r.epoch - 1)
            self.assertEqual(r.observed_max_age, ages[r.epoch - 1])

    def test_replay_honours_a_longer_telemetry_lag(self):
        ages = [10.0, 20.0, 30.0, 40.0, 50.0]
        sel = replay(_cfg(), ages, PHI, lag=2)
        for r in sel.log[2:]:
            self.assertEqual(r.observation_from_epoch, r.epoch - 2)

    def test_bootstrap_epoch_is_nominal_and_labelled(self):
        """The first epoch cannot be protected -- nothing has been observed. It
        must be visibly a bootstrap, not a decision."""
        sel = replay(_cfg(), [200.0, 200.0], PHI)
        self.assertEqual(sel.log[0].candidate_after, "C0")
        self.assertEqual(sel.log[0].reason, BOOTSTRAP)
        self.assertFalse(sel.log[0].switched)


class Escalation(unittest.TestCase):
    def test_escalates_directly_to_the_level_the_risk_implies(self):
        """Danger gets the strongest applicable response at once; it does not
        walk up one rung per epoch while outputs are invalid."""
        s = Selector(_cfg())
        # risk = 1.5 exceeds C2's entry of 1.0, from level 0
        self.assertEqual(s.decide(1, 150.0, PHI, observation_from_epoch=0), "C2")
        self.assertEqual(s.log[-1].reason, ESCALATED)
        self.assertEqual(s.log[-1].level_after, 2)

    def test_intermediate_risk_selects_the_middle_level(self):
        s = Selector(_cfg())
        self.assertEqual(s.decide(1, 80.0, PHI, observation_from_epoch=0), "C1")

    def test_low_risk_stays_nominal(self):
        s = Selector(_cfg())
        self.assertEqual(s.decide(1, 10.0, PHI, observation_from_epoch=0), "C0")
        self.assertEqual(s.log[-1].reason, HELD_NO_TRIGGER)

    def test_saturation_at_the_top_is_reported_distinctly(self):
        """Still in danger at maximum protection is NOT "nothing triggered": it
        means the candidate set is inadequate. Conflating the two would let a
        run that never had a working candidate look like a calm run."""
        s = Selector(_cfg())
        s.decide(1, 150.0, PHI, observation_from_epoch=0)   # -> C2
        s.decide(2, 500.0, PHI, observation_from_epoch=1)   # still over threshold
        self.assertEqual(s.log[-1].reason, HELD_AT_CEILING)
        self.assertFalse(s.log[-1].switched)

    def test_calm_at_the_top_is_not_reported_as_saturation(self):
        """At C2 but with risk inside C2's hysteresis band, the selector is
        holding for hysteresis -- not saturated."""
        s = Selector(_cfg())
        s.decide(1, 150.0, PHI, observation_from_epoch=0)   # -> C2
        s.decide(2, 90.0, PHI, observation_from_epoch=1)    # 0.9: below entry 1.0,
        self.assertNotEqual(s.log[-1].reason, HELD_AT_CEILING)  # above exit 0.8


class Hysteresis(unittest.TestCase):
    def test_does_not_step_down_inside_the_hysteresis_band(self):
        """At C1 (entry 0.7, exit 0.5), risk 0.6 implies target C0 but sits
        inside the band, so protection is retained."""
        s = Selector(_cfg())
        s.decide(1, 80.0, PHI, observation_from_epoch=0)     # -> C1
        self.assertEqual(s.decide(2, 60.0, PHI, observation_from_epoch=1), "C1")
        self.assertEqual(s.log[-1].reason, HELD_HYSTERESIS)
        self.assertIn(HELD_HYSTERESIS, s.log[-1].blocked_by)

    def test_steps_down_once_below_the_exit_threshold(self):
        s = Selector(_cfg())
        s.decide(1, 80.0, PHI, observation_from_epoch=0)     # -> C1
        self.assertEqual(s.decide(2, 40.0, PHI, observation_from_epoch=1), "C0")
        self.assertEqual(s.log[-1].reason, DEESCALATED)

    def test_deescalation_is_one_rung_per_decision(self):
        """From C2, a single quiet epoch must not drop straight to nominal."""
        s = Selector(_cfg())
        s.decide(1, 150.0, PHI, observation_from_epoch=0)    # -> C2
        self.assertEqual(s.decide(2, 0.0, PHI, observation_from_epoch=1), "C1")
        self.assertEqual(s.decide(3, 0.0, PHI, observation_from_epoch=2), "C0")

    def test_equal_entry_and_exit_thresholds_are_rejected(self):
        with self.assertRaises(ValueError) as cm:
            CandidateLevel("X", 1, entry_risk=0.7, exit_risk=0.7)
        self.assertIn("oscillate", str(cm.exception))


class Brakes(unittest.TestCase):
    def test_min_residency_blocks_an_early_switch(self):
        s = Selector(_cfg(min_residency=3))
        s.decide(1, 150.0, PHI, observation_from_epoch=0)  # bootstrap level 0 -> C2?
        # epochs_at_current_level was 0 at the first decision, so residency binds
        self.assertEqual(s.log[-1].reason, HELD_MIN_RESIDENCY)
        self.assertFalse(s.log[-1].switched)

    def test_cooldown_blocks_a_second_switch_immediately_after_the_first(self):
        s = Selector(_cfg(cooldown=2))
        s.decide(1, 80.0, PHI, observation_from_epoch=0)   # -> C1 (no prior switch)
        self.assertTrue(s.log[-1].switched)
        s.decide(2, 150.0, PHI, observation_from_epoch=1)  # wants C2, cooled down
        self.assertEqual(s.log[-1].reason, HELD_COOLDOWN)
        self.assertFalse(s.log[-1].switched)

    def test_both_brakes_are_recorded_when_both_bind(self):
        s = Selector(_cfg(min_residency=5, cooldown=5))
        s.decide(1, 80.0, PHI, observation_from_epoch=0)
        s.decide(2, 150.0, PHI, observation_from_epoch=1)
        blocked = s.log[-1].blocked_by.split("|")
        self.assertIn(HELD_MIN_RESIDENCY, blocked)

    def test_brakes_off_permits_switching_every_epoch(self):
        """Sanity: with no brakes the selector is maximally reactive. If this
        fails, a passing brake test might just mean the selector never
        switches at all."""
        s = Selector(_cfg())
        s.decide(1, 150.0, PHI, observation_from_epoch=0)  # -> C2
        s.decide(2, 0.0, PHI, observation_from_epoch=1)    # -> C1
        s.decide(3, 0.0, PHI, observation_from_epoch=2)    # -> C0
        self.assertEqual(s.state.switch_count, 3)


class ConfigValidation(unittest.TestCase):
    def test_gapped_protection_levels_are_rejected(self):
        with self.assertRaises(ValueError) as cm:
            SelectorConfig(levels=(
                CandidateLevel("C0", 0, 0.0, -1.0),
                CandidateLevel("C2", 2, 1.0, 0.8),
            ))
        self.assertIn("no gaps", str(cm.exception))

    def test_non_increasing_entry_thresholds_are_rejected(self):
        """A higher level whose entry threshold is not above the level below it
        can never be selected, so a candidate would be silently dead."""
        with self.assertRaises(ValueError) as cm:
            SelectorConfig(levels=(
                CandidateLevel("C0", 0, 0.0, -1.0),
                CandidateLevel("C1", 1, 0.7, 0.5),
                CandidateLevel("C2", 2, 0.7, 0.5),
            ))
        self.assertIn("unreachable", str(cm.exception))

    def test_empty_level_set_is_rejected(self):
        with self.assertRaises(ValueError):
            SelectorConfig(levels=())

    def test_zero_freshness_window_is_rejected(self):
        s = Selector(_cfg())
        with self.assertRaises(ValueError):
            s.decide(1, 10.0, 0.0, observation_from_epoch=0)


class Reporting(unittest.TestCase):
    def test_summary_reports_switch_count_and_residency(self):
        ages = [0.0, 150.0, 150.0, 0.0, 0.0, 0.0]
        sel = replay(_cfg(), ages, PHI)
        s = sel.summary()
        self.assertEqual(s["n_epochs"], len(ages))
        self.assertEqual(s["switch_count"], sel.state.switch_count)
        self.assertAlmostEqual(sum(s["fraction_in_candidate"].values()), 1.0)
        self.assertEqual(sum(s["epochs_in_candidate"].values()), len(ages))

    def test_csv_columns_match_the_row_fields(self):
        sel = replay(_cfg(), [10.0, 20.0], PHI)
        self.assertEqual(set(CSV_COLUMNS), set(sel.rows()[0]))

    def test_switch_epochs_are_recorded_for_transition_analysis(self):
        """Phase 11 needs the switch boundaries to check whether any invocation
        straddles one."""
        sel = replay(_cfg(), [0.0, 150.0, 150.0, 150.0], PHI)
        self.assertEqual(
            sel.summary()["switch_epochs"],
            [r.epoch for r in sel.log if r.switched],
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
