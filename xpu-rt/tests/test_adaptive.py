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
        rows = [_cell("C0", b, 0.9, makespan=310.0) for b in range(5)]
        _, warnings = evaluate_trajectories(
            rows, ["C0"], phi=PHI, epoch_ms=EPOCH,
            trajectories={"t": [0, 1]})
        self.assertTrue(warnings)
        self.assertTrue(any("NOT meaningful" in w for w in warnings))


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


if __name__ == "__main__":
    unittest.main(verbosity=2)
