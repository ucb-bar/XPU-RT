"""Boundary-robustness tests for the freshness evaluator.

All three of the evaluator's comparisons are documented as INCLUSIVE (producer
eligibility, consumer deadline, freshness window) but were implemented as exact
float comparisons. That is not a hypothetical problem for this experiment: phi is
anchored on A0, the measured uncontended input-age ceiling, and the uncontended
ages ARE A0 -- so at phi = A0 + delta there exist consumer instances whose age
equals phi to the last bit, and their verdict was decided by whatever rounding
happened along the way.

How it was found: the same schedule evaluated on two arithmetically equivalent
timing bases (the same measured cycles converted at 1 GHz and at 25 MHz, with
periods scaled to match) disagreed on one invocation in thirty.

    1 GHz : age = 70.54607400000002  phi = 70.546074   -> stale
    25 MHz: age = 2821.84296         phi = 2821.84296   -> valid

A 1.4e-14 disagreement moved a reported rate by 0.033. Without a tolerance the
published number depends on the order floating-point operations were performed
in, which is not a property any result should have.
"""

from __future__ import annotations

import os
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_XPURT = os.path.dirname(_HERE)
sys.path.insert(0, _XPURT)

from freshness import (  # noqa: E402
    BOUNDARY_RTOL,
    NO_COMPLETED_PRODUCER,
    STALE_INPUT,
    VALID,
    FreshnessEdge,
    Invocation,
    _lte,
    evaluate_freshness,
    select_producer,
)

PROD, CONS = "dronet", "mlp_control"


def _edge(window):
    return FreshnessEdge(producer_task=PROD, consumer_task=CONS,
                         freshness_window=window, criticality="hard")


def _trace(producer_release, consumer_start, consumer_end, deadline=None):
    return [
        Invocation(task=PROD, instance=0, release_time=producer_release,
                   start_time=producer_release, end_time=producer_release + 1.0),
        Invocation(task=CONS, instance=0, release_time=consumer_start,
                   start_time=consumer_start, end_time=consumer_end,
                   deadline=deadline),
    ]


def _reason(trace, window):
    ev = evaluate_freshness(trace, dependency_edges=[_edge(window)], time_unit="ms")
    rows = [r for r in ev.rows() if r["consumer_task"] == CONS]
    return rows[0]["invalid_reason"]


class LteHelper(unittest.TestCase):
    def test_exact_equality_satisfies(self):
        self.assertTrue(_lte(70.546074, 70.546074))

    def test_one_ulp_over_still_satisfies(self):
        """The measured failure: 1.4e-14 above the window must not flip."""
        self.assertTrue(_lte(70.54607400000002, 70.546074))

    def test_a_genuinely_larger_value_does_not_satisfy(self):
        """The tolerance must not swallow real differences. The smallest real age
        gap in this workload is the 10 ms control period."""
        self.assertFalse(_lte(70.546074 + 1e-3, 70.546074))
        self.assertFalse(_lte(80.546074, 70.546074))

    def test_tolerance_is_relative_so_it_scales_with_magnitude(self):
        self.assertTrue(_lte(2821.84296 * (1 + BOUNDARY_RTOL / 2), 2821.84296))
        self.assertFalse(_lte(2821.84296 * (1 + 1e-6), 2821.84296))

    def test_tolerance_is_tiny_relative_to_the_experiment_grid(self):
        """A tolerance big enough to reclassify a real invocation would be worse
        than the bug. 1e-9 relative at ~100 ms is ~1e-7 ms, four orders of
        magnitude below the 1e-3 ms resolution of any measured duration."""
        self.assertLess(BOUNDARY_RTOL * 100.0, 1e-6)


class FreshnessWindowBoundary(unittest.TestCase):
    def test_age_exactly_equal_to_the_window_is_valid(self):
        # producer released at 0, consumer output at t=70.546074 -> age == window
        self.assertEqual(_reason(_trace(0.0, 60.0, 70.546074), 70.546074), VALID)

    def test_age_one_ulp_over_the_window_is_still_valid(self):
        self.assertEqual(
            _reason(_trace(0.0, 60.0, 70.54607400000002), 70.546074), VALID)

    def test_age_meaningfully_over_the_window_is_stale(self):
        self.assertEqual(_reason(_trace(0.0, 60.0, 70.6), 70.546074), STALE_INPUT)

    def test_verdict_is_identical_under_a_40x_time_rescaling(self):
        """The invariance the bug broke: scaling every time by k must not change
        any verdict, since the schedule is then a pure time-rescaling."""
        K = 40.0
        for end, window in ((70.546074, 70.546074),
                            (70.54607400000002, 70.546074),
                            (70.6, 70.546074)):
            base = _reason(_trace(0.0, 60.0, end), window)
            scaled = _reason(_trace(0.0, 60.0 * K, end * K), window * K)
            self.assertEqual(base, scaled,
                             f"verdict changed under rescaling for age={end}")


class ProducerEligibilityBoundary(unittest.TestCase):
    @staticmethod
    def _producers(*ends):
        return [Invocation(task=PROD, instance=i, release_time=0.0,
                           start_time=0.0, end_time=e) for i, e in enumerate(ends)]

    def test_producer_finishing_exactly_at_consumer_start_is_eligible(self):
        p = select_producer(self._producers(50.0), consumer_start_time=50.0)
        self.assertIsNotNone(p)

    def test_producer_finishing_one_ulp_after_is_still_eligible(self):
        p = select_producer(self._producers(50.000000000000007),
                            consumer_start_time=50.0)
        self.assertIsNotNone(p,
                             "a 7e-15 overshoot must not make the producer "
                             "invisible; that would report no_completed_producer")

    def test_producer_finishing_meaningfully_after_is_not_eligible(self):
        self.assertIsNone(
            select_producer(self._producers(50.001), consumer_start_time=50.0))

    def test_no_producer_case_is_still_reachable(self):
        """The tolerance must not accidentally make every producer eligible."""
        trace = [
            Invocation(task=PROD, instance=0, release_time=0.0, start_time=0.0,
                       end_time=90.0),
            Invocation(task=CONS, instance=0, release_time=10.0, start_time=10.0,
                       end_time=11.0),
        ]
        self.assertEqual(_reason(trace, 100.0), NO_COMPLETED_PRODUCER)


class DeadlineBoundary(unittest.TestCase):
    def test_finishing_exactly_on_the_deadline_meets_it(self):
        r = _reason(_trace(0.0, 60.0, 70.0, deadline=70.0), 1000.0)
        self.assertEqual(r, VALID)

    def test_finishing_one_ulp_late_still_meets_it(self):
        r = _reason(_trace(0.0, 60.0, 70.00000000000001, deadline=70.0), 1000.0)
        self.assertEqual(r, VALID)

    def test_finishing_meaningfully_late_misses(self):
        r = _reason(_trace(0.0, 60.0, 70.1, deadline=70.0), 1000.0)
        self.assertNotEqual(r, VALID)


if __name__ == "__main__":
    unittest.main(verbosity=2)
