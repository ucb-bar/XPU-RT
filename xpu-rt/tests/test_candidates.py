"""Tests for the candidate-set validation gate.

The gate exists because of a measured negative finding: the protection mechanism
the plan proposed (reserve the fast accelerator for the perception producer)
performed WORSE than doing nothing at every contention level. So the gate's job
is to make that outcome *fatal* rather than merely visible — a selector built on
a candidate that harms its own objective would generate a confident
adaptive-vs-static comparison whose conclusion is about the candidate, not about
adaptation, and nothing in the numbers would say so.

The most important test here is `test_gate_fails_on_the_measured_regression`: it
replays the real Gate A numbers and asserts the gate rejects them. If that test
ever passes trivially, the gate has stopped protecting anything.
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

from benchmarks.freshness_eval.candidates import (  # noqa: E402
    Candidate,
    build_candidate_set,
    validate_candidate_set,
)

PHI = 80.546


def _row(policy, burst, ovr, soft=0, phi=PHI):
    return {
        "policy": policy,
        "contention_level": str(burst),
        "freshness_window": str(phi),
        "output_valid_rate": str(ovr),
        "soft_instances_completed": str(soft),
        "seed": "0",
    }


def _nominal(**kw):
    return Candidate(
        candidate_id="C0", protection_level=0, intent="nominal",
        intended_bursts=(0, 1, 2, 3, 4), solver="greedy", scheduler="mosek",
        mutations={}, **kw
    )


def _protect(cid="C1", bursts=(1, 2, 3), mutations=None, level=1):
    return Candidate(
        candidate_id=cid, protection_level=level, intent="protect perception",
        intended_bursts=bursts, solver="greedy", scheduler="mosek",
        mutations=mutations if mutations is not None else {"window_duration": {"dronet": 25.0}},
    )


class CandidateConstruction(unittest.TestCase):
    def test_unknown_mutation_is_rejected(self):
        with self.assertRaises(ValueError) as cm:
            _protect(mutations={"reserve_the_whole_accelerator": True})
        self.assertIn("reserve_the_whole_accelerator", str(cm.exception))

    def test_empty_intended_region_is_rejected(self):
        with self.assertRaises(ValueError) as cm:
            _protect(bursts=())
        self.assertIn("region", str(cm.exception))

    def test_describe_marks_unvalidated_candidates(self):
        d = _protect().describe()
        self.assertEqual(d["status"], "precomputed, NOT yet validated")
        self.assertNotIn("certified", str(d).lower(),
                         "the word 'certified' must never appear: no validation "
                         "or proof procedure establishes a guarantee here")


class GateRejectsRegressions(unittest.TestCase):
    def test_gate_fails_on_the_measured_regression(self):
        """The real Gate A numbers: static_conservative vs static_nominal at
        phi = A0+20. Nominal 0.633/0.400/0.220 at B=1/2/3; the 'protective'
        candidate measured 0.146 at B=3 and worse throughout."""
        rows = [
            _row("C0", 1, 0.633), _row("C0", 2, 0.400), _row("C0", 3, 0.220),
            _row("C1", 1, 0.500), _row("C1", 2, 0.300), _row("C1", 3, 0.146),
        ]
        gate = validate_candidate_set(
            [_nominal(), _protect()], rows, phi=PHI, nominal_id="C0")
        self.assertFalse(gate.admissible)
        self.assertEqual(gate.per_candidate["C1"]["verdict"], "FAIL")
        self.assertLess(gate.per_candidate["C1"]["margin"], 0)
        self.assertTrue(any("does not protect" in f for f in gate.findings))

    def test_build_raises_in_strict_mode_on_a_failing_set(self):
        rows = [_row("C0", b, 0.5) for b in (1, 2, 3)] + \
               [_row("C1", b, 0.2) for b in (1, 2, 3)]
        with self.assertRaises(ValueError) as cm:
            build_candidate_set([_nominal(), _protect()], rows,
                                phi=PHI, run_label="probe")
        self.assertIn("refusing to build a selector", str(cm.exception))

    def test_non_strict_mode_returns_the_failing_set_for_inspection(self):
        rows = [_row("C0", b, 0.5) for b in (1, 2, 3)] + \
               [_row("C1", b, 0.2) for b in (1, 2, 3)]
        cands, gate = build_candidate_set(
            [_nominal(), _protect()], rows, phi=PHI, run_label="probe", strict=False)
        self.assertFalse(gate.admissible)
        self.assertEqual(len(cands), 2)

    def test_a_candidate_that_helps_outside_its_region_still_fails(self):
        """Declaring a region is a commitment. Doing well only at B=4 does not
        validate a candidate that claims B=1..3."""
        rows = [
            _row("C0", 1, 0.60), _row("C0", 2, 0.40), _row("C0", 3, 0.20), _row("C0", 4, 0.00),
            _row("C1", 1, 0.50), _row("C1", 2, 0.30), _row("C1", 3, 0.10), _row("C1", 4, 0.90),
        ]
        gate = validate_candidate_set(
            [_nominal(), _protect(bursts=(1, 2, 3))], rows, phi=PHI, nominal_id="C0")
        self.assertFalse(gate.admissible)

    def test_missing_measurement_in_region_is_unscorable_not_a_pass(self):
        """Absent data must never read as success."""
        rows = [_row("C0", b, 0.5) for b in (1, 2, 3)] + \
               [_row("C1", 1, 0.9), _row("C1", 2, 0.9)]   # no B=3
        gate = validate_candidate_set(
            [_nominal(), _protect(bursts=(1, 2, 3))], rows, phi=PHI, nominal_id="C0")
        self.assertFalse(gate.admissible)
        self.assertIn("UNSCORABLE", gate.per_candidate["C1"]["verdict"])


class GateAcceptsImprovements(unittest.TestCase):
    def test_gate_passes_a_candidate_that_helps_in_its_region(self):
        rows = [
            _row("C0", 1, 0.633), _row("C0", 2, 0.400), _row("C0", 3, 0.220),
            _row("C1", 1, 0.800), _row("C1", 2, 0.700), _row("C1", 3, 0.500),
        ]
        gate = validate_candidate_set(
            [_nominal(), _protect()], rows, phi=PHI, nominal_id="C0")
        self.assertTrue(gate.admissible)
        self.assertEqual(gate.per_candidate["C1"]["verdict"], "PASS")
        self.assertGreater(gate.per_candidate["C1"]["margin"], 0)

    def test_equal_performance_passes_the_weakest_bar_but_is_flagged(self):
        rows = [_row("C0", b, 0.5) for b in (1, 2, 3)] + \
               [_row("C1", b, 0.5) for b in (1, 2, 3)]
        gate = validate_candidate_set(
            [_nominal(), _protect()], rows, phi=PHI, nominal_id="C0")
        self.assertTrue(gate.admissible)
        self.assertTrue(any("buys nothing" in f for f in gate.findings))

    def test_min_margin_can_demand_a_real_improvement(self):
        rows = [_row("C0", b, 0.50) for b in (1, 2, 3)] + \
               [_row("C1", b, 0.52) for b in (1, 2, 3)]
        self.assertTrue(validate_candidate_set(
            [_nominal(), _protect()], rows, phi=PHI, nominal_id="C0").admissible)
        self.assertFalse(validate_candidate_set(
            [_nominal(), _protect()], rows, phi=PHI, nominal_id="C0",
            min_margin=0.10).admissible)

    def test_measurements_and_provenance_are_attached(self):
        rows = [_row("C0", b, 0.5, soft=b) for b in (1, 2, 3)] + \
               [_row("C1", b, 0.9, soft=0) for b in (1, 2, 3)]
        cands, _ = build_candidate_set(
            [_nominal(), _protect()], rows, phi=PHI, run_label="probe_v1")
        c1 = next(c for c in cands if c.candidate_id == "C1")
        self.assertEqual(c1.measured_in, "probe_v1")
        self.assertAlmostEqual(c1.expected_output_valid[3], 0.9)
        self.assertEqual(c1.describe()["status"],
                         "precomputed and empirically validated")


class LadderShape(unittest.TestCase):
    def test_gapped_protection_levels_are_rejected(self):
        rows = [_row("C0", b, 0.5) for b in (1, 2, 3)]
        with self.assertRaises(ValueError) as cm:
            build_candidate_set([_nominal(), _protect(level=2)], rows,
                                phi=PHI, run_label="x")
        self.assertIn("no gaps", str(cm.exception))

    def test_two_nominals_are_rejected_by_the_level_check(self):
        """Duplicate level 0 is caught as a level-shape error, which is also
        what guarantees exactly one nominal exists."""
        rows = [_row("C0", b, 0.5) for b in (1, 2, 3)]
        with self.assertRaises(ValueError) as cm:
            build_candidate_set([_nominal(), _protect(cid="C1", level=0)], rows,
                                phi=PHI, run_label="x")
        msg = str(cm.exception)
        self.assertIn("[0, 0]", msg)
        self.assertIn("nominal", msg, "the message should explain the consequence")

    def test_absent_nominal_rows_are_reported_not_silently_passed(self):
        gate = validate_candidate_set(
            [_nominal(), _protect()], [_row("C1", 1, 0.9)], phi=PHI, nominal_id="C0")
        self.assertFalse(gate.admissible)
        self.assertTrue(any("nothing to validate against" in f for f in gate.findings))

    def test_rows_at_other_phi_are_ignored(self):
        """phi selects the operating point; mixing windows would average away
        the very axis the sweep varies."""
        rows = [_row("C0", b, 0.5, phi=PHI) for b in (1, 2, 3)] + \
               [_row("C1", b, 0.9, phi=PHI) for b in (1, 2, 3)] + \
               [_row("C1", b, 0.0, phi=999.0) for b in (1, 2, 3)]
        gate = validate_candidate_set(
            [_nominal(), _protect()], rows, phi=PHI, nominal_id="C0")
        self.assertAlmostEqual(gate.per_candidate["C1"]["candidate_output_valid"], 0.9)


if __name__ == "__main__":
    unittest.main(verbosity=2)
