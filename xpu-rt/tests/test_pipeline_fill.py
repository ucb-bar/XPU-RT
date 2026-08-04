"""Tests for separating "acted on stale input" from "had no input at all".

`output_valid_rate` merges two failures that are not the same claim, and the merge
was materially misleading here. Measured at phi = A0+20 on the canonical workload,
static_nominal at B=1 lost 11 of 30 consumer invocations:

    stale_input             2      acted on an input that was too old
    no_completed_producer   9      had no input whatsoever

Only the first is the phenomenon this project studies. Reporting 0.367 divergence
implies eleven controller outputs computed from stale perception; nine of them had
no perception to compute from, which a real controller would handle by holding or
faulting rather than by actuating on garbage.

Of those 9, two are unavoidable at ANY contention level -- they are present at
B=0. The pipeline starts empty: a consumer released before the producer could have
finished even with the machine to itself is unservable by construction. The other
7 exist because the producer's first instance lost the t=0 race to the soft burst
and finished at 87 ms instead of 17.7 ms, which IS a scheduling outcome.

Hence the discipline this file pins: the pipeline-fill threshold is a WORKLOAD
constant (uncontended producer release + uncontended producer latency), never each
policy's own first producer completion. A per-policy threshold would excuse a
policy for exactly the starvation being measured -- the same failure as scoring a
candidate's deadline compliance against its own tightened window.
"""

from __future__ import annotations

import os
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_XPURT = os.path.dirname(_HERE)
sys.path.insert(0, _XPURT)

from freshness import (  # noqa: E402
    NO_COMPLETED_PRODUCER,
    STALE_INPUT,
    VALID,
    FreshnessEdge,
    Invocation,
    aggregate_metrics,
    evaluate_freshness,
)

PROD, CONS = "dronet", "mlp_control"
LP = 17.69          # uncontended producer latency (ms)
PHI = 80.55


def _edge(window=PHI):
    return FreshnessEdge(producer_task=PROD, consumer_task=CONS,
                         freshness_window=window, criticality="hard")


def _trace(producer_ends, consumer_starts, *, consumer_dur=0.55):
    """Producer instances finishing at `producer_ends`; consumers at 10 ms grid."""
    invs = []
    for i, e in enumerate(producer_ends):
        invs.append(Invocation(task=PROD, instance=i, release_time=i * 50.0,
                               start_time=e - LP, end_time=e))
    for j, s in enumerate(consumer_starts):
        invs.append(Invocation(task=CONS, instance=j, release_time=s,
                               start_time=s, end_time=s + consumer_dur))
    return invs


def _ev(producer_ends, consumer_starts, fill=LP, window=PHI):
    return evaluate_freshness(
        _trace(producer_ends, consumer_starts),
        dependency_edges=[_edge(window)], time_unit="ms",
        pipeline_fill_ms=fill,
    )


class StaleIsSeparatedFromMissing(unittest.TestCase):
    def test_the_two_reasons_are_reported_as_distinct_rates(self):
        # consumers at 0 and 10 have no producer (first finishes at 17.69);
        # a consumer at 200 reading a producer that finished at 100 is stale.
        ev = _ev([17.69, 100.0], [0.0, 10.0, 20.0, 200.0])
        a = ev.aggregate
        self.assertEqual(a["no_producer_count"], 2)
        self.assertGreaterEqual(a["stale_input_count"], 1)
        self.assertAlmostEqual(a["no_producer_rate"], 2 / 4)
        self.assertAlmostEqual(a["stale_input_rate"], 1 / 4)

    def test_the_rates_plus_valid_account_for_everything(self):
        ev = _ev([17.69, 100.0], [0.0, 10.0, 20.0, 200.0])
        a = ev.aggregate
        self.assertAlmostEqual(
            a["output_valid_rate"] + a["stale_input_rate"] + a["no_producer_rate"]
            + a["deadline_miss_count"] / a["total_consumer_invocations"],
            1.0,
            msg="the reasons must partition the invocations")

    def test_a_missing_producer_is_not_counted_as_stale(self):
        ev = _ev([17.69], [0.0])
        rows = [r for r in ev.rows() if r["consumer_task"] == CONS]
        self.assertEqual(rows[0]["invalid_reason"], NO_COMPLETED_PRODUCER)
        self.assertEqual(ev.aggregate["stale_input_count"], 0)


class SteadyStateExcludesOnlyTheUnservable(unittest.TestCase):
    def test_consumers_before_the_fill_time_are_excluded(self):
        ev = _ev([17.69], [0.0, 10.0, 20.0, 30.0])
        a = ev.aggregate
        self.assertEqual(a["structurally_unservable_count"], 2)
        self.assertEqual(a["steady_total_consumer_invocations"], 2)

    def test_a_consumer_exactly_at_the_fill_time_is_included(self):
        ev = _ev([17.69], [LP])
        self.assertEqual(ev.aggregate["structurally_unservable_count"], 0)
        self.assertEqual(ev.aggregate["steady_total_consumer_invocations"], 1)

    def test_steady_rates_are_over_the_steady_subset(self):
        # 4 consumers, 2 unservable, of the remaining 2 one is valid one is stale
        ev = _ev([17.69], [0.0, 10.0, 20.0, 200.0])
        a = ev.aggregate
        self.assertEqual(a["steady_total_consumer_invocations"], 2)
        self.assertAlmostEqual(a["steady_output_valid_rate"], 0.5)
        self.assertAlmostEqual(a["steady_stale_input_rate"], 0.5)

    def test_full_trace_rates_are_still_emitted_unchanged(self):
        """The steady numbers ADD to the report; they do not replace it. A reader
        must be able to see the cold start rather than have it silently removed."""
        ev = _ev([17.69], [0.0, 10.0, 20.0, 200.0])
        a = ev.aggregate
        self.assertEqual(a["total_consumer_invocations"], 4)
        self.assertAlmostEqual(a["output_valid_rate"], 0.25)
        self.assertNotAlmostEqual(a["output_valid_rate"],
                                  a["steady_output_valid_rate"])

    def test_no_fill_time_means_no_exclusion(self):
        ev = evaluate_freshness(
            _trace([17.69], [0.0, 10.0, 20.0]),
            dependency_edges=[_edge()], time_unit="ms")
        a = ev.aggregate
        self.assertIsNone(a["pipeline_fill_ms"])
        self.assertEqual(a["structurally_unservable_count"], 0)
        self.assertEqual(a["steady_total_consumer_invocations"], 3)
        self.assertAlmostEqual(a["steady_output_valid_rate"],
                               a["output_valid_rate"])

    def test_empty_record_set_is_handled(self):
        a = aggregate_metrics([], pipeline_fill_ms=LP)
        self.assertEqual(a["structurally_unservable_count"], 0)
        self.assertIsNone(a["steady_output_valid_rate"])
        self.assertEqual(a["pipeline_fill_ms"], LP)


class ThresholdIsAWorkloadConstant(unittest.TestCase):
    """The central discipline: the threshold must not adapt to the policy."""

    def test_a_starving_policy_gets_no_credit_from_the_fixed_threshold(self):
        """Two policies, same consumers. Policy A finishes the producer at 17.69
        (uncontended); policy B starves it until 87.17. With the FIXED workload
        threshold, B is penalised for the 7 extra unserved invocations."""
        consumers = [i * 10.0 for i in range(10)]
        good = _ev([17.69], consumers).aggregate
        bad = _ev([87.17], consumers).aggregate

        self.assertEqual(good["structurally_unservable_count"],
                         bad["structurally_unservable_count"],
                         "the exclusion must be identical across policies")
        self.assertEqual(good["structurally_unservable_count"], 2)
        # B still carries its starvation, in the steady window, as missing input.
        self.assertAlmostEqual(bad["steady_no_producer_rate"], 7 / 8)
        self.assertAlmostEqual(good["steady_no_producer_rate"], 0.0)

    def test_a_self_relative_threshold_would_have_erased_the_difference(self):
        """Demonstrates the bug being prevented: threshold = each policy's own
        first producer completion makes the starving policy look identical to the
        uncontended one."""
        consumers = [i * 10.0 for i in range(10)]
        good = _ev([17.69], consumers, fill=17.69).aggregate
        bad_self = _ev([87.17], consumers, fill=87.17).aggregate
        self.assertAlmostEqual(bad_self["steady_no_producer_rate"], 0.0)
        self.assertAlmostEqual(good["steady_no_producer_rate"], 0.0)
        self.assertNotEqual(good["structurally_unservable_count"],
                            bad_self["structurally_unservable_count"],
                            "a self-relative threshold hides the starvation by "
                            "excusing more invocations for the worse policy")

    def test_fill_time_is_derived_from_the_uncontended_latency(self):
        import json
        from benchmarks.freshness_eval.run import compute_a0
        from freshness import freshness_edges_from_config
        repo = os.path.dirname(_XPURT)
        cfg_path = os.path.join(repo, "data", "toplevel",
                                "freshness_canon_300ms.json")
        with open(cfg_path) as f:
            base = json.load(f)
        edge = freshness_edges_from_config(base)[0]
        info = compute_a0(base, epoch_ms=300.0, edge=edge)
        self.assertAlmostEqual(
            info["pipeline_fill_ms"],
            float(base["networks"][edge.producer_task].get("start_time", 0.0))
            + info["producer_latency_ms"])
        # And it must be strictly inside the first producer period, or every
        # invocation would be excluded.
        self.assertLess(info["pipeline_fill_ms"], info["producer_period_ms"])
        self.assertGreater(info["pipeline_fill_ms"], 0.0)

    def test_fill_time_excludes_exactly_the_b0_floor(self):
        """On the canonical workload the B=0 floor of unservable consumers is 2
        (consumers at t=0 and t=10; the producer finishes at ~17.7). If the
        threshold ever drifts past a consumer boundary this count changes, which
        is a silent restatement of every steady rate."""
        import json
        from benchmarks.freshness_eval.run import compute_a0
        from freshness import freshness_edges_from_config
        repo = os.path.dirname(_XPURT)
        with open(os.path.join(repo, "data", "toplevel",
                              "freshness_canon_300ms.json")) as f:
            base = json.load(f)
        edge = freshness_edges_from_config(base)[0]
        info = compute_a0(base, epoch_ms=300.0, edge=edge)
        tc = info["consumer_period_ms"]
        n_excluded = sum(1 for k in range(100)
                         if k * tc < info["pipeline_fill_ms"])
        self.assertEqual(n_excluded, 2,
                         f"fill={info['pipeline_fill_ms']} excludes {n_excluded} "
                         f"consumers on a {tc} ms grid, not 2")


if __name__ == "__main__":
    unittest.main(verbosity=2)
