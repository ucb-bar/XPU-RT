"""The acceptance objective must prefer meeting deadlines over saving cycles.

These tests encode the two worked examples the objective exists for, and pin the
specific failure of the criterion it replaces: accepting on summed standalone
cycles rejects every parallelism win by construction, because splitting an op
across cores makes the *sum* worse even when it makes the *deadline*.
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

from candidate_objective import (  # noqa: E402
    GATE_CORRECTNESS, GATE_LEGALITY, GATE_PROFILE,
    CandidateOutcome, Ineligible, ModelOutcome, Tolerances,
    accept, compare, from_trace_summary, rank,
)


def _model(instances=10, misses=0, worst=0.0, p99=0.0, hz=30.0, req=30.0):
    return ModelOutcome(instances=instances, deadline_misses=misses,
                        worst_lateness_ms=worst, response_p99_ms=p99,
                        achieved_hz=hz, required_hz=req)


class WorkedExamples(unittest.TestCase):

    def test_a_split_that_costs_cycles_but_meets_the_deadline_is_a_win(self):
        """5% more cycles, but DroNet goes from 20% misses to none."""
        base = CandidateOutcome(
            label="baseline",
            per_model={"dronet": _model(instances=10, misses=2, worst=8.0,
                                        hz=24.0, req=30.0)},
            makespan_ms=400.0, standalone_cycles=1_000_000)
        cand = CandidateOutcome(
            label="split",
            per_model={"dronet": _model(instances=10, misses=0, worst=0.0,
                                        hz=30.0, req=30.0)},
            makespan_ms=400.0, standalone_cycles=1_050_000)  # 5% WORSE
        ok, why = accept(cand, base)
        self.assertTrue(ok, why)
        self.assertIn("deadline misses", why)

    def test_a_fusion_that_is_faster_alone_but_breaks_a_cofrunning_loop_loses(self):
        """10% fewer cycles, but an 8 ms dispatch wrecks a 100 Hz MLP."""
        base = CandidateOutcome(
            label="baseline",
            per_model={"mlp": _model(instances=100, misses=0, worst=0.0,
                                     hz=100.0, req=100.0),
                       "dronet": _model(instances=30, misses=0, hz=30.0,
                                        req=30.0)},
            makespan_ms=300.0, standalone_cycles=1_000_000)
        cand = CandidateOutcome(
            label="fused",
            per_model={"mlp": _model(instances=100, misses=40, worst=8.0,
                                     hz=62.0, req=100.0),
                       "dronet": _model(instances=30, misses=0, hz=30.0,
                                        req=30.0)},
            makespan_ms=290.0, standalone_cycles=900_000)  # 10% BETTER
        ok, why = accept(cand, base)
        self.assertFalse(ok, why)
        self.assertIn("deadline misses", why)


class SummedCyclesMustNotDecide(unittest.TestCase):

    def test_worse_summed_cycles_alone_never_rejects_a_real_time_win(self):
        """The precise defect of the criterion this replaces.

        An OC=8 tile costs 76% of the OC=16 op, so a 2-way split inflates total
        work ~53%. Under a summed-cycles objective that is a guaranteed
        rejection even when it halves the critical path.
        """
        base = CandidateOutcome(
            label="unsharded",
            per_model={"dronet": _model(instances=10, misses=10, worst=108.0,
                                        hz=22.6, req=30.0)},
            makespan_ms=441.0, standalone_cycles=1_000_000)
        cand = CandidateOutcome(
            label="sharded",
            per_model={"dronet": _model(instances=10, misses=10, worst=27.0,
                                        hz=29.0, req=30.0)},
            makespan_ms=413.0, standalone_cycles=1_530_000)  # +53%
        ok, why = accept(cand, base)
        self.assertTrue(ok, why)
        self.assertIn("lateness", why,
                      "misses tie at 10/10, so lateness should decide")

    def test_cycles_decide_only_when_everything_else_ties(self):
        common = dict(per_model={"m": _model(instances=10, misses=0)},
                      makespan_ms=100.0)
        a = CandidateOutcome(label="a", standalone_cycles=900_000, **common)
        b = CandidateOutcome(label="b", standalone_cycles=1_000_000, **common)
        order, why = compare(a, b)
        self.assertEqual(order, -1)
        self.assertIn("standalone kernel cycles", why)


class NoiseMustNotDecide(unittest.TestCase):

    def test_a_miss_difference_within_tolerance_falls_through(self):
        """Measured: 7-9 misses of 38 across identical runs of one schedule.

        Treating 9-vs-7 as decisive would make the highest-priority term the
        noisiest one, with nothing downstream able to correct it.
        """
        a = CandidateOutcome(
            label="run_a",
            per_model={"mlp": _model(instances=38, misses=7)},
            makespan_ms=413.0)
        b = CandidateOutcome(
            label="run_b",
            per_model={"mlp": _model(instances=38, misses=9)},
            makespan_ms=450.0)  # clearly worse, and should be what decides
        order, why = compare(a, b)
        self.assertEqual(order, -1)
        self.assertIn("makespan", why,
                      f"a 2-instance miss difference in 38 is within noise; "
                      f"makespan should decide. got: {why}")

    def test_a_large_miss_difference_still_decides(self):
        """The tolerance must not blunt the signal it exists to protect."""
        a = CandidateOutcome(
            label="good", per_model={"m": _model(instances=10, misses=2)},
            makespan_ms=999.0)
        b = CandidateOutcome(
            label="bad", per_model={"m": _model(instances=10, misses=10)},
            makespan_ms=100.0)
        order, why = compare(a, b)
        self.assertEqual(order, -1)
        self.assertIn("deadline misses", why)

    def test_tolerances_are_caller_overridable(self):
        a = CandidateOutcome(label="a",
                             per_model={"m": _model(instances=100, misses=2)},
                             makespan_ms=100.0)
        b = CandidateOutcome(label="b",
                             per_model={"m": _model(instances=100, misses=4)},
                             makespan_ms=100.0)
        # Default: 2 of 100 is within max(1, 5%) = 5 -> tie -> falls through.
        self.assertEqual(compare(a, b)[0], 0)
        # A caller who measured a tight spread can say so.
        strict = Tolerances(miss_instances=1, miss_rate_frac=0.0)
        self.assertEqual(compare(a, b, strict)[0], -1)


class EligibilityGates(unittest.TestCase):

    def _base(self):
        return CandidateOutcome(label="base",
                                per_model={"m": _model(instances=10, misses=5)},
                                makespan_ms=500.0)

    def test_a_correctness_failure_is_rejected_however_fast(self):
        cand = CandidateOutcome(
            label="fast_but_wrong",
            per_model={"m": _model(instances=10, misses=0)},
            makespan_ms=1.0,
            ineligible=Ineligible(GATE_CORRECTNESS, "max_abs_err=5"))
        ok, why = accept(cand, self._base())
        self.assertFalse(ok)
        self.assertIn("correctness", why)

    def test_a_legality_violation_is_rejected(self):
        cand = CandidateOutcome(
            label="ime_on_cluster1",
            per_model={"m": _model(instances=10, misses=0)},
            makespan_ms=1.0,
            ineligible=Ineligible(GATE_LEGALITY,
                                  "smt.vmadot placed on CPU_E, which traps"))
        self.assertFalse(accept(cand, self._base())[0])

    def test_an_invalid_profile_is_rejected(self):
        """A single cold sample, or a unit mismatch, is not a measurement."""
        cand = CandidateOutcome(
            label="n1",
            per_model={"m": _model(instances=10, misses=0)},
            makespan_ms=1.0,
            ineligible=Ineligible(GATE_PROFILE,
                                  "n=1 sample; median over warm reps required"))
        self.assertFalse(accept(cand, self._base())[0])

    def test_gates_are_checked_before_any_term(self):
        a = CandidateOutcome(label="a", per_model={"m": _model(misses=0)},
                             ineligible=Ineligible(GATE_CORRECTNESS, "x"))
        b = CandidateOutcome(label="b", per_model={"m": _model(misses=99)})
        self.assertEqual(compare(a, b)[0], 1, "ineligible always loses")


class TiesAndRanking(unittest.TestCase):

    def test_a_tie_is_a_rejection_not_an_acceptance(self):
        """A change that cannot be shown to help is not worth taking.

        It costs a rebuild, regenerated kernels, and a known-good baseline.
        """
        c = CandidateOutcome(label="same",
                             per_model={"m": _model(instances=10, misses=1)},
                             makespan_ms=100.0)
        b = CandidateOutcome(label="base",
                             per_model={"m": _model(instances=10, misses=1)},
                             makespan_ms=100.0)
        ok, why = accept(c, b)
        self.assertFalse(ok)
        self.assertIn("tie", why.lower())

    def test_rank_orders_best_first(self):
        mk = lambda n, misses, ms: CandidateOutcome(  # noqa: E731
            label=n, per_model={"m": _model(instances=20, misses=misses)},
            makespan_ms=ms)
        got = [c.label for c in rank([mk("mid", 5, 200.0),
                                      mk("best", 0, 300.0),
                                      mk("worst", 15, 100.0)])]
        self.assertEqual(got, ["best", "mid", "worst"])


class TraceSummaryAdapter(unittest.TestCase):

    def test_builds_from_trace_metrics_output(self):
        summary = {
            "makespan_us": 413_500.0,
            "per_model": {
                "dronet": {"instances": 12, "instance_deadline_misses": 12,
                           "worst_lateness_ms": 27.15,
                           "response_p99_ms": 60.45,
                           "achieved_frequency_hz": 29.02,
                           "required_frequency_hz": 30.0},
                "mlp": {"instances": 38, "instance_deadline_misses": 9,
                        "worst_lateness_ms": 2.68, "response_p99_ms": 12.68,
                        "achieved_frequency_hz": 100.21,
                        "required_frequency_hz": 100.0},
            },
            "per_cluster_utilization_pct": {"CPU_P": 51.6, "CPU_E": 54.2},
        }
        c = from_trace_summary("B4", summary, heavy_model="dronet")
        self.assertAlmostEqual(c.makespan_ms, 413.5)
        self.assertEqual(c.total_misses(), 21)
        self.assertAlmostEqual(c.worst_lateness(), 27.15)
        self.assertAlmostEqual(c.heavy_throughput_hz, 29.02)
        self.assertAlmostEqual(c.utilization_pct, 52.9, places=4)
        # dronet is 29.02 of 30 required -> ~3.3% short; mlp exceeds its rate.
        self.assertGreater(c.worst_frequency_shortfall(), 0.03)
        self.assertLess(c.worst_frequency_shortfall(), 0.04)

    def test_critical_models_restricts_the_hard_set(self):
        summary = {
            "makespan_us": 1000.0,
            "per_model": {
                "mlp": {"instances": 10, "instance_deadline_misses": 0,
                        "achieved_frequency_hz": 100.0,
                        "required_frequency_hz": 100.0},
                "yolo": {"instances": 2, "instance_deadline_misses": 2,
                         "worst_lateness_ms": 500.0,
                         "achieved_frequency_hz": 0.3,
                         "required_frequency_hz": 5.0},
            },
        }
        both = from_trace_summary("all", summary)
        hard_only = from_trace_summary("hard", summary,
                                       critical_models=("mlp",))
        self.assertEqual(both.total_misses(), 2)
        self.assertEqual(hard_only.total_misses(), 0,
                         "a soft-deadline background model's misses must not "
                         "dominate term 1")


if __name__ == "__main__":
    unittest.main()
