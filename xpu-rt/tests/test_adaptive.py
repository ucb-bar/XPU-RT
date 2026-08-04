"""Tests for the Phase 11 adaptive-vs-static harness.

The harness composes measured per-epoch cells, and the things most likely to
quietly produce a flattering result are:

  * an "oracle" that is not actually contention-aware, making adaptive look
    closer to optimal than it is;
  * an adaptive loop that is not a loop -- if the observation fed back does not
    depend on the candidate that was chosen, a bad choice never degrades its own
    next input and adaptive is measured on easy mode;
  * `transition_violations: 0` reported because nothing was checked rather than
    because boundaries are clean;
  * hard validity computed as a mean of per-epoch RATES, which silently
    reweights epochs that have different consumer counts;
  * a missing (candidate, contention) cell filled in by interpolation instead
    of raising, inventing data the sweep never measured.
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
    CellTable,
    TRAJECTORIES,
    check_clean_boundaries,
    default_selector_config,
    evaluate_trajectories,
    run_adaptive,
    run_oracle_contention_aware,
    run_static,
)

PHI = 100.0
EPOCH = 300.0


def _cell(policy, burst, ovr, *, age=50.0, soft=0, makespan=290.5,
          n_inv=30, phi=PHI):
    valid = int(round(ovr * n_inv))
    return {
        "policy": policy,
        "contention_level": str(burst),
        "freshness_window": str(phi),
        "output_valid_rate": str(ovr),
        "valid_count": str(valid),
        "total_consumer_invocations": str(n_inv),
        "freshness_success_rate": str(ovr),
        "deadline_success_rate": "1.0",
        "max_input_age": str(age),
        "soft_instances_completed": str(soft),
        "makespan_ms": str(makespan),
        "seed": "0",
    }


def _grid():
    """C0 degrades hard with contention; C1 holds up but sheds soft work."""
    rows = []
    for b, (o0, o1) in enumerate([(0.93, 0.93), (0.63, 0.85), (0.40, 0.80),
                                  (0.22, 0.75), (0.00, 0.70)]):
        rows.append(_cell("C0", b, o0, age=60.0 + 30 * b, soft=b))
        rows.append(_cell("C1", b, o1, age=60.0 + 5 * b, soft=max(0, b - 1)))
    return rows


class Composition(unittest.TestCase):
    def test_static_reads_the_cell_for_each_epochs_contention(self):
        t = CellTable(_grid(), PHI)
        r = run_static(t, "C0", [0, 2, 4], "custom", EPOCH)
        self.assertEqual([e.offered_burst for e in r.epochs], [0, 2, 4])
        self.assertAlmostEqual(r.epochs[1].output_valid_rate, 0.40)
        self.assertAlmostEqual(r.epochs[2].output_valid_rate, 0.00)
        self.assertTrue(all(e.candidate_id == "C0" for e in r.epochs))

    def test_missing_cell_raises_rather_than_interpolating(self):
        rows = [_cell("C0", b, 0.5) for b in (0, 1, 2)]  # no B=3
        t = CellTable(rows, PHI)
        with self.assertRaises(KeyError) as cm:
            run_static(t, "C0", [0, 3], "custom", EPOCH)
        self.assertIn("never scheduled", str(cm.exception))

    def test_rows_at_a_different_phi_are_not_used(self):
        rows = _grid() + [_cell("C0", 0, 0.0, phi=999.0)]
        t = CellTable(rows, PHI)
        self.assertAlmostEqual(
            run_static(t, "C0", [0], "x", EPOCH).epochs[0].output_valid_rate, 0.93)

    def test_hard_validity_is_invocation_weighted_not_a_mean_of_rates(self):
        """Two epochs, 30 and 10 consumer invocations, rates 1.0 and 0.0.
        Mean-of-rates says 0.50; the correct answer is 30/40 = 0.75."""
        rows = [_cell("C0", 0, 1.0, n_inv=30), _cell("C0", 1, 0.0, n_inv=10)]
        t = CellTable(rows, PHI)
        s = run_static(t, "C0", [0, 1], "x", EPOCH).summary()
        self.assertAlmostEqual(s["hard_output_valid_rate"], 0.75)
        self.assertEqual(s["consumer_invocations"], 40)


class OracleIsContentionAware(unittest.TestCase):
    def test_oracle_picks_the_best_candidate_at_each_contention_level(self):
        t = CellTable(_grid(), PHI)
        r = run_oracle_contention_aware(t, ["C0", "C1"], [0, 1, 4], "x", EPOCH)
        # At B=0 both are 0.93 -> tie goes to the LOWER protection level.
        self.assertEqual(r.epochs[0].candidate_id, "C0")
        # At B=1 and B=4, C1 is better.
        self.assertEqual(r.epochs[1].candidate_id, "C1")
        self.assertEqual(r.epochs[2].candidate_id, "C1")

    def test_tie_breaks_to_lower_protection_so_it_gets_no_free_shedding(self):
        """If ties went to the protective candidate, the oracle would shed soft
        work it did not need to shed and its utility number would be
        pessimistic -- flattering adaptive by comparison."""
        rows = [_cell("C0", 0, 0.9, soft=3), _cell("C1", 0, 0.9, soft=0)]
        t = CellTable(rows, PHI)
        r = run_oracle_contention_aware(t, ["C0", "C1"], [0], "x", EPOCH)
        self.assertEqual(r.epochs[0].candidate_id, "C0")
        self.assertEqual(r.summary()["soft_completed"], 3)

    def test_oracle_beats_or_matches_every_static_on_hard_validity(self):
        """An upper bound that a static policy can beat is not an upper bound."""
        t = CellTable(_grid(), PHI)
        traj = TRAJECTORIES["ramp"]
        orc = run_oracle_contention_aware(t, ["C0", "C1"], traj, "ramp", EPOCH)
        for cid in ("C0", "C1"):
            st = run_static(t, cid, traj, "ramp", EPOCH)
            self.assertGreaterEqual(
                orc.summary()["hard_output_valid_rate"] + 1e-12,
                st.summary()["hard_output_valid_rate"],
                f"oracle must not be beaten by static_{cid}",
            )


class AdaptiveIsAClosedLoop(unittest.TestCase):
    def test_observation_depends_on_the_candidate_that_ran(self):
        """The fed-back age must come from the cell actually executed. If it
        came from a fixed sequence, a bad choice would not degrade its own next
        input and the loop would be fake."""
        t = CellTable(_grid(), PHI)
        cfg = default_selector_config(["C0", "C1"], entry_risks=(0.0, 0.85),
                                      exit_risks=(-1.0, 0.70))
        r = run_adaptive(t, cfg, [4, 4, 4, 4], PHI, "x", EPOCH)
        ages = [e.max_input_age for e in r.epochs]
        cands = [e.candidate_id for e in r.epochs]
        # Epoch 0 is bootstrap on C0 (age 180 at B=4 -> risk 1.8), so the
        # selector escalates and the age recorded afterwards must be C1's (80),
        # not C0's. Identical ages throughout would mean the loop is open.
        self.assertEqual(cands[0], "C0")
        self.assertAlmostEqual(ages[0], 180.0)
        self.assertIn("C1", cands[1:])
        self.assertLess(min(ages[1:]), ages[0],
                        "escalating must change the observed age")

    def test_first_epoch_is_unprotected_bootstrap(self):
        t = CellTable(_grid(), PHI)
        cfg = default_selector_config(["C0", "C1"], entry_risks=(0.0, 0.85),
                                      exit_risks=(-1.0, 0.70))
        r = run_adaptive(t, cfg, [4, 4], PHI, "x", EPOCH)
        self.assertEqual(r.epochs[0].candidate_id, "C0")
        self.assertEqual(r.epochs[0].selector_reason, "bootstrap_no_observation")

    def test_adaptive_never_exceeds_the_contention_aware_oracle(self):
        t = CellTable(_grid(), PHI)
        cfg = default_selector_config(["C0", "C1"], entry_risks=(0.0, 0.85),
                                      exit_risks=(-1.0, 0.70))
        for name, traj in TRAJECTORIES.items():
            a = run_adaptive(t, cfg, traj, PHI, name, EPOCH).summary()
            o = run_oracle_contention_aware(t, ["C0", "C1"], traj, name,
                                            EPOCH).summary()
            self.assertLessEqual(
                a["hard_output_valid_rate"], o["hard_output_valid_rate"] + 1e-12,
                f"{name}: adaptive beat the full-knowledge oracle, which means "
                f"the oracle is not an upper bound or adaptive is peeking",
            )

    def test_selector_log_is_emitted(self):
        t = CellTable(_grid(), PHI)
        cfg = default_selector_config(["C0", "C1"], entry_risks=(0.0, 0.85),
                                      exit_risks=(-1.0, 0.70))
        r = run_adaptive(t, cfg, TRAJECTORIES["step"], PHI, "step", EPOCH)
        self.assertEqual(len(r.selector_log), len(TRAJECTORIES["step"]))
        self.assertIn("observation_from_epoch", r.selector_log[0])


class BoundaryHonesty(unittest.TestCase):
    def test_overrunning_makespan_is_detected(self):
        rows = [_cell("C0", 0, 0.9, makespan=290.5),
                _cell("C0", 1, 0.9, makespan=310.0)]
        t = CellTable(rows, PHI)
        problems = check_clean_boundaries(t, ["C0"], [0, 1], EPOCH)
        self.assertEqual(len(problems), 1)
        self.assertIn("exceeds", problems[0])

    def test_clean_boundaries_report_no_problems(self):
        t = CellTable(_grid(), PHI)
        self.assertEqual(
            check_clean_boundaries(t, ["C0", "C1"], [0, 1, 2, 3, 4], EPOCH), [])

    def test_transition_violations_counts_overrunning_epochs(self):
        rows = [_cell("C0", 0, 0.9, makespan=310.0)]
        t = CellTable(rows, PHI)
        s = run_static(t, "C0", [0, 0], "x", EPOCH).summary()
        self.assertEqual(s["transition_violations"], 2)

    def test_evaluate_trajectories_surfaces_boundary_warnings(self):
        # C0 overruns everywhere; C1 fits, so the oracle is still definable and
        # the warning under test is about the boundary, not a missing oracle.
        rows = ([_cell("C0", b, 0.9, makespan=310.0) for b in range(5)]
                + [_cell("C1", b, 0.8, makespan=290.5) for b in range(5)])
        results, warnings = evaluate_trajectories(
            rows, ["C0", "C1"], phi=PHI, epoch_ms=EPOCH,
            trajectories={"t": [0, 1]})
        self.assertTrue(warnings)
        self.assertTrue(any("NOT meaningful" in w for w in warnings))
        self.assertIn("oracle_contention_aware", {r.strategy for r in results})

    def test_an_undefinable_oracle_is_skipped_with_a_warning_not_raised(self):
        """When nothing fits at some burst there is no attainable bound. The
        static and adaptive rows are still valid, so losing them to an exception
        would discard good data to report a missing reference."""
        rows = [_cell("C0", b, 0.9, makespan=310.0) for b in range(5)]
        results, warnings = evaluate_trajectories(
            rows, ["C0"], phi=PHI, epoch_ms=EPOCH,
            trajectories={"t": [0, 1]})
        strategies = {r.strategy for r in results}
        self.assertNotIn("oracle_contention_aware", strategies)
        self.assertIn("static_C0", strategies)
        self.assertIn("adaptive", strategies)
        self.assertTrue(any("NO oracle bound" in w for w in warnings), warnings)

    def test_the_oracle_will_not_pick_an_overrunning_cell(self):
        """The defect this guards against, reproduced from the real numbers: at
        B=3 the best rate belonged to a 495 ms schedule (0.952) and the best
        epoch-respecting one to a 297 ms schedule (0.933)."""
        rows = [_cell("FAST", 3, 0.952, makespan=495.4, n_inv=42),
                _cell("FITS", 3, 0.933, makespan=297.2, n_inv=30)]
        t = CellTable(rows, PHI)
        res = run_oracle_contention_aware(t, ["FAST", "FITS"], [3], "x", EPOCH)
        self.assertEqual(res.epochs[0].candidate_id, "FITS")
        self.assertAlmostEqual(res.epochs[0].output_valid_rate, 0.933)


class EndToEnd(unittest.TestCase):
    def test_every_trajectory_yields_every_strategy(self):
        results, warnings = evaluate_trajectories(
            _grid(), ["C0", "C1"], phi=PHI, epoch_ms=EPOCH)
        self.assertEqual(warnings, [])
        names = {(r.trajectory_name, r.strategy) for r in results}
        for traj in TRAJECTORIES:
            for strat in ("static_C0", "static_C1", "adaptive",
                          "oracle_contention_aware"):
                self.assertIn((traj, strat), names)

    def test_selector_overhead_is_reported_per_epoch(self):
        results, _ = evaluate_trajectories(
            _grid(), ["C0", "C1"], phi=PHI, epoch_ms=EPOCH,
            trajectories={"ramp": TRAJECTORIES["ramp"]})
        ad = next(r for r in results if r.strategy == "adaptive")
        s = ad.summary()
        self.assertGreater(s["selector_overhead_us_per_epoch"], 0.0)
        # Statics do not run a selector, so their overhead must be zero rather
        # than an unmeasured blank.
        st = next(r for r in results if r.strategy == "static_C0")
        self.assertEqual(st.summary()["selector_overhead_us_total"], 0.0)

    def test_trajectories_are_fixed_in_code(self):
        """Guards against tuning a trajectory until adaptive wins."""
        self.assertEqual(set(TRAJECTORIES), {"ramp", "step", "oscillate", "sustained"})
        self.assertEqual(TRAJECTORIES["sustained"], [4] * 10)


class SaturatedObservableDefeatsTheSelector(unittest.TestCase):
    """The measured cause of adaptation's failure on this workload.

    Under any protective rung, max_input_age is FLAT at 90.55 ms for B = 1, 2, 3
    and 4 (risk 1.124 at phi = A0+20), against 60.55 ms at B=0. So the selector's
    only input cannot distinguish 65% offered load from 131%. The values that do
    discriminate B=4 belong to schedules that overrun the epoch, and are therefore
    observable only after the overrun has already happened.

    These tests use a synthetic grid with that saturation property rather than the
    measured CSVs, so they pin the CONSEQUENCE of a saturated observable -- which
    is a property of reactive selection, not of one results directory.
    """

    #      B:      0      1      2      3      4
    AGES = {"P1": [60.0, 90.0, 90.0, 90.0, 410.0],   # protective, overruns at B=4
            "P2": [60.0, 90.0, 90.0, 90.0, 90.0]}    # protective, always fits
    MK = {"P1": [290.5, 290.5, 290.5, 290.5, 805.9],
          "P2": [290.5] * 5}
    OVR = {"P1": [0.93, 0.90, 0.87, 0.83, 0.16],
           "P2": [0.93, 0.90, 0.87, 0.87, 0.87]}
    SOFT = {"P1": [0, 1, 2, 3, 3], "P2": [0, 1, 2, 2, 2]}

    def _rows(self):
        rows = []
        for cid in ("P1", "P2"):
            for b in range(5):
                rows.append(_cell(cid, b, self.OVR[cid][b], age=self.AGES[cid][b],
                                  soft=self.SOFT[cid][b], makespan=self.MK[cid][b]))
        return rows

    def _table(self):
        return CellTable(self._rows(), PHI)

    def test_the_observable_does_not_discriminate_contention(self):
        """The premise. If this ever stops holding the findings below are void."""
        t = self._table()
        ages = [float(t.get("P2", b)["max_input_age"]) for b in (1, 2, 3, 4)]
        self.assertEqual(len(set(ages)), 1,
                         f"expected a saturated signal, got {ages}")
        self.assertNotEqual(ages[0], float(t.get("P2", 0)["max_input_age"]),
                            "B=0 must still be distinguishable, or the signal "
                            "carries no information at all")

    def test_a_step_to_max_contention_costs_one_overrunning_epoch(self):
        t = self._table()
        cfg = default_selector_config(["P1", "P2"], entry_risks=(0.0, 0.85),
                                      exit_risks=(-1.0, 0.70))
        res = run_adaptive(t, cfg, [0, 0, 4, 4, 4], PHI, "step", EPOCH, lag=1)
        overruns = [e.epoch for e in res.epochs if e.makespan_ms > EPOCH]
        self.assertEqual(overruns, [2],
                         "the first high-contention epoch is entered on the "
                         "previous epoch's low-risk observation and must overrun")
        self.assertEqual(res.epochs[2].candidate_id, "P1")
        self.assertEqual(res.epochs[3].candidate_id, "P2",
                         "it should escalate immediately afterwards -- the "
                         "failure is the lag, not a broken selector")

    def test_a_threshold_low_enough_to_be_safe_stops_adapting(self):
        """The finding that corrects the obvious first guess.

        Thresholds that survive the step DO exist -- but the only observation
        available before contention arrives is the B=0 one (risk 0.60 here), so
        such a selector escalates at ZERO contention and never returns. It becomes
        safe by degenerating into the conservative static policy, which is not
        adaptation.
        """
        t = self._table()
        safe = default_selector_config(["P1", "P2"], entry_risks=(0.0, 0.55),
                                       exit_risks=(-1.0, 0.40))
        res = run_adaptive(t, safe, [0, 0, 4, 4, 0, 0], PHI, "step", EPOCH, lag=1)
        self.assertTrue(all(e.makespan_ms <= EPOCH for e in res.epochs),
                        "a threshold below the B=0 risk must be safe")
        chosen = [e.candidate_id for e in res.epochs]
        self.assertEqual(chosen[1:], ["P2"] * 5,
                         f"it should pin to the protective rung forever, got {chosen}")
        static = run_static(t, "P2", [0, 0, 4, 4, 0, 0], "step", EPOCH)
        self.assertEqual(res.summary()["soft_completed"],
                         static.summary()["soft_completed"],
                         "a selector that never de-escalates must reproduce the "
                         "static policy's utility exactly")

    def test_a_gradual_ramp_is_survivable_because_it_warns_first(self):
        """Contrast case: an intermediate burst raises risk before B=4 arrives, so
        the same selector and thresholds are admissible on a ramp."""
        t = self._table()
        cfg = default_selector_config(["P1", "P2"], entry_risks=(0.0, 0.85),
                                      exit_risks=(-1.0, 0.70))
        res = run_adaptive(t, cfg, [0, 1, 2, 3, 4], PHI, "ramp", EPOCH, lag=1)
        self.assertTrue(all(e.makespan_ms <= EPOCH for e in res.epochs),
                        f"ramp should be safe; got "
                        f"{[(e.epoch, e.candidate_id, e.makespan_ms) for e in res.epochs]}")


class AdaptiveIsNotCreditedForUnmodellableEpochs(unittest.TestCase):
    def test_a_strategy_that_overruns_is_flagged_inadmissible(self):
        from benchmarks.freshness_eval.adaptive import admissible_strategies
        t = CellTable(
            [_cell("P1", 4, 0.16, age=410.0, soft=3, makespan=805.9),
             _cell("P1", 0, 0.93, age=60.0, soft=0, makespan=290.5)], PHI)
        bad = run_static(t, "P1", [0, 4], "x", EPOCH)
        good = run_static(t, "P1", [0, 0], "y", EPOCH)
        ok = admissible_strategies([bad, good])
        self.assertFalse(ok["static_P1@x"])
        self.assertTrue(ok["static_P1@y"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
