"""Tests for the three consumption policies, and what distinguishes them.

`latest_completed` was the only policy this project ever evaluated, and the plan
lists that as an open limitation: the headline divergence could in principle be an
artifact of it. These tests pin the semantics; benchmarks/freshness_eval/consumption.py
measures whether the finding survives.

The policies answer "which readable producer output IS the input":

    latest_completed   the most recently WRITTEN sample (max end_time)
    newest_version     the freshest SAMPLE present (max instance index)
    release_matched    strictly the CURRENT frame, no substitution

All three share the physical constraint that a sample must be written before it
can be read. They diverge in two situations that this workload actually produces,
so the distinction is measurable rather than theoretical:

  * OUT-OF-ORDER COMPLETION separates latest_completed from newest_version. With
    heterogeneous backends a perception instance placed on the slow cluster can
    finish after a later instance placed on the fast one, so "most recently
    written" and "newest sample" are different instances.
  * A LATE CURRENT FRAME separates release_matched from both. When the current
    frame has not completed, release_matched reports NO input while the other two
    substitute an older completed sample and report a STALE one. That moves
    invocations between the two failure categories the Gate A summary decomposes,
    which is exactly why it has to be measured rather than assumed harmless.
"""

from __future__ import annotations

import os
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_XPURT = os.path.dirname(_HERE)
sys.path.insert(0, _XPURT)

from freshness import (  # noqa: E402
    CONSUMPTION_POLICIES,
    LATEST_COMPLETED,
    NEWEST_VERSION,
    NO_COMPLETED_PRODUCER,
    RELEASE_MATCHED,
    STALE_INPUT,
    FreshnessEdge,
    Invocation,
    evaluate_freshness,
    select_producer,
)

PROD, CONS = "dronet", "mlp_control"


def _p(instance, release, start, end):
    return Invocation(task=PROD, instance=instance, release_time=release,
                      start_time=start, end_time=end)


class PolicyRegistry(unittest.TestCase):
    def test_latest_completed_is_first_and_default(self):
        """Reordering would silently restate every result reported before the
        alternatives existed."""
        self.assertEqual(CONSUMPTION_POLICIES[0], LATEST_COMPLETED)
        p = select_producer([_p(0, 0.0, 0.0, 10.0)], 20.0)
        self.assertIsNotNone(p)

    def test_unknown_policy_raises(self):
        with self.assertRaises(ValueError) as cm:
            select_producer([], 0.0, "most_convenient")
        self.assertIn("unknown consumption_policy", str(cm.exception))

    def test_release_matched_demands_the_consumer_release_time(self):
        """It selects by frame, which the start time cannot express."""
        with self.assertRaises(ValueError) as cm:
            select_producer([_p(0, 0.0, 0.0, 10.0)], 20.0, RELEASE_MATCHED)
        self.assertIn("consumer_release_time", str(cm.exception))


class OutOfOrderCompletionSeparatesTwoPolicies(unittest.TestCase):
    """The heterogeneous case: instance 0 on the slow backend finishes last.

        instance 0  released 0,  runs 0 -> 90   (slow cluster)
        instance 1  released 50, runs 50 -> 60  (fast cluster)

    At consumer start 100 both are readable. "Most recently written" is instance
    0 (end 90); "newest sample" is instance 1 (index 1).
    """

    PRODUCERS = [_p(0, 0.0, 0.0, 90.0), _p(1, 50.0, 50.0, 60.0)]

    def test_latest_completed_takes_the_last_written(self):
        p = select_producer(self.PRODUCERS, 100.0, LATEST_COMPLETED)
        self.assertEqual(p.instance, 0)

    def test_newest_version_takes_the_freshest_sample(self):
        p = select_producer(self.PRODUCERS, 100.0, NEWEST_VERSION)
        self.assertEqual(p.instance, 1)

    def test_the_two_policies_really_do_differ_here(self):
        a = select_producer(self.PRODUCERS, 100.0, LATEST_COMPLETED)
        b = select_producer(self.PRODUCERS, 100.0, NEWEST_VERSION)
        self.assertNotEqual(a.instance, b.instance,
                            "if these agree the fixture no longer exercises "
                            "out-of-order completion and proves nothing")

    def test_they_agree_when_completion_follows_release_order(self):
        in_order = [_p(0, 0.0, 0.0, 20.0), _p(1, 50.0, 50.0, 70.0)]
        a = select_producer(in_order, 100.0, LATEST_COMPLETED)
        b = select_producer(in_order, 100.0, NEWEST_VERSION)
        self.assertEqual(a.instance, b.instance)


class ReleaseMatchedRefusesSubstitution(unittest.TestCase):
    def test_it_picks_the_frame_matching_the_consumer_release(self):
        producers = [_p(0, 0.0, 0.0, 20.0), _p(1, 50.0, 50.0, 70.0)]
        p = select_producer(producers, 80.0, RELEASE_MATCHED,
                            consumer_release_time=60.0)
        self.assertEqual(p.instance, 1, "release 60 belongs to the frame released at 50")

    def test_a_late_current_frame_yields_no_input_not_a_stale_one(self):
        """The distinguishing behaviour. The current frame (released 50) has not
        finished; an older completed sample exists and is deliberately NOT used."""
        producers = [_p(0, 0.0, 0.0, 20.0), _p(1, 50.0, 50.0, 200.0)]
        self.assertIsNone(
            select_producer(producers, 80.0, RELEASE_MATCHED,
                            consumer_release_time=60.0))
        # ...whereas the substituting policies hand back the older sample.
        for pol in (LATEST_COMPLETED, NEWEST_VERSION):
            got = select_producer(producers, 80.0, pol,
                                  consumer_release_time=60.0)
            self.assertIsNotNone(got, pol)
            self.assertEqual(got.instance, 0, pol)

    def test_a_consumer_released_before_any_producer_has_no_frame(self):
        producers = [_p(0, 50.0, 50.0, 70.0)]
        self.assertIsNone(
            select_producer(producers, 100.0, RELEASE_MATCHED,
                            consumer_release_time=10.0))


class DecompositionShiftsBetweenPolicies(unittest.TestCase):
    """End to end: the same trace, scored three ways, moves invocations between
    stale_input and no_completed_producer. That is the reason this matters -- the
    Gate A summary reports those two separately."""

    def _trace(self):
        # Producer frame 1 (released 50) is very late; frame 0 completed early.
        return [
            _p(0, 0.0, 0.0, 20.0),
            _p(1, 50.0, 50.0, 250.0),
            Invocation(task=CONS, instance=0, release_time=60.0,
                       start_time=60.0, end_time=60.5),
        ]

    def _ev(self, policy, window=1000.0):
        edge = FreshnessEdge(producer_task=PROD, consumer_task=CONS,
                             freshness_window=window, criticality="hard",
                             consumption_policy=policy)
        return evaluate_freshness(self._trace(), dependency_edges=[edge],
                                  time_unit="ms")

    def test_substituting_policies_report_an_input(self):
        for pol in (LATEST_COMPLETED, NEWEST_VERSION):
            a = self._ev(pol).aggregate
            self.assertEqual(a["no_producer_count"], 0, pol)

    def test_release_matched_reports_no_input(self):
        a = self._ev(RELEASE_MATCHED).aggregate
        self.assertEqual(a["no_producer_count"], 1)
        self.assertEqual(a["stale_input_count"], 0)

    def test_a_tight_window_makes_the_substituted_input_stale(self):
        """With a window of 30 ms the substituted 60 ms-old sample is stale, so
        latest_completed records a STALE input where release_matched records NO
        input. Same trace, same schedule, different failure attributed."""
        stale = self._ev(LATEST_COMPLETED, window=30.0)
        none_ = self._ev(RELEASE_MATCHED, window=30.0)
        rows_s = [r for r in stale.rows() if r["consumer_task"] == CONS]
        rows_n = [r for r in none_.rows() if r["consumer_task"] == CONS]
        self.assertEqual(rows_s[0]["invalid_reason"], STALE_INPUT)
        self.assertEqual(rows_n[0]["invalid_reason"], NO_COMPLETED_PRODUCER)
        # Both are invalid: the headline validity number is unchanged, only its
        # decomposition moves. Worth asserting so nobody reads the policy choice
        # as changing how much output is valid.
        self.assertEqual(stale.aggregate["output_valid_rate"],
                         none_.aggregate["output_valid_rate"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
