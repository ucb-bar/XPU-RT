"""Unit tests for producer-consumer freshness semantics.

Every fixture is hand-constructed so the right answer is known by inspection —
no solver, no profile data, no randomness. Cases 1-7 are the specified
correctness contract; Case 2 (stale but on time) is the counterexample the whole
evaluation exists to detect, and Cases 6-7 pin the two decisions most likely to
be silently wrong: which producer instance is selected, and whether the
completion/start boundary is inclusive.
"""

from __future__ import annotations

import os
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))  # xpu-rt/ on path

from freshness import (  # noqa: E402
    DEADLINE_AND_STALE,
    DEADLINE_MISS,
    INFERRED,
    NO_COMPLETED_PRODUCER,
    STALE_INPUT,
    VALID,
    FreshnessEdge,
    Invocation,
    analytic_age_ceiling_realized,
    analytic_age_supremum,
    aggregate_metrics,
    criticality_from_config,
    evaluate_freshness,
    freshness_edges_from_config,
    select_producer,
    split_instance_name,
)

PHI = 50.0


def _edge(**kw):
    base = dict(
        producer_task="dronet",
        consumer_task="mlp_control",
        freshness_window=PHI,
    )
    base.update(kw)
    return FreshnessEdge(**base)


def _producer(instance=0, release=0.0, start=0.0, end=10.0, sample=None):
    return Invocation(
        task="dronet",
        instance=instance,
        release_time=release,
        start_time=start,
        end_time=end,
        sample_time=sample,
    )


def _consumer(instance=0, release=20.0, start=20.0, end=21.0, deadline=30.0):
    return Invocation(
        task="mlp_control",
        instance=instance,
        release_time=release,
        start_time=start,
        end_time=end,
        deadline=deadline,
    )


def _one(trace, edge=None):
    """Evaluate and return the single expected record."""
    ev = evaluate_freshness(trace, dependency_edges=[edge or _edge()])
    assert len(ev.records) == 1, f"expected 1 record, got {len(ev.records)}"
    return ev.records[0]


class SpecifiedCases(unittest.TestCase):
    """The seven required cases."""

    def test_case1_fresh_and_on_time(self):
        # producer samples at 0, done at 10; consumer 20 -> 21, deadline 30.
        # age_at_output = 21 - 0 = 21 <= 50, and 21 <= 30.
        r = _one([_producer(), _consumer()])
        self.assertTrue(r.deadline_valid)
        self.assertTrue(r.freshness_valid)
        self.assertTrue(r.output_valid)
        self.assertEqual(r.invalid_reason, VALID)
        self.assertEqual(r.input_age_at_output, 21.0)
        self.assertEqual(r.input_age_at_start, 20.0)
        self.assertEqual(r.producer_instance, 0)

    def test_case2_stale_but_on_time(self):
        """The central paper counterexample: deadline met, output invalid.

        The consumer does everything right — released at 100, finished at 101,
        well inside its 110 deadline — yet acts on a sample taken at t=0, so the
        command it emits is 101 units stale against a 50 window.
        """
        r = _one(
            [
                _producer(release=0.0, start=0.0, end=10.0),
                _consumer(release=100.0, start=100.0, end=101.0, deadline=110.0),
            ]
        )
        self.assertTrue(r.deadline_valid, "consumer met its own deadline")
        self.assertFalse(r.freshness_valid)
        self.assertFalse(r.output_valid)
        self.assertEqual(r.invalid_reason, STALE_INPUT)
        self.assertEqual(r.input_age_at_output, 101.0)

    def test_case3_fresh_but_late(self):
        # age 21 <= 50 (fresh) but end 21 > deadline 15.
        r = _one([_producer(), _consumer(deadline=15.0)])
        self.assertFalse(r.deadline_valid)
        self.assertTrue(r.freshness_valid)
        self.assertFalse(r.output_valid)
        self.assertEqual(r.invalid_reason, DEADLINE_MISS)

    def test_case4_late_and_stale(self):
        r = _one(
            [
                _producer(release=0.0, start=0.0, end=10.0),
                _consumer(release=100.0, start=100.0, end=101.0, deadline=95.0),
            ]
        )
        self.assertFalse(r.deadline_valid)
        self.assertFalse(r.freshness_valid)
        self.assertFalse(r.output_valid)
        self.assertEqual(r.invalid_reason, DEADLINE_AND_STALE)

    def test_case5_no_completed_producer(self):
        # Producer still running at the consumer's start (ends 25 > starts 20).
        r = _one([_producer(release=0.0, start=0.0, end=25.0), _consumer()])
        self.assertIsNone(r.producer_instance)
        self.assertIsNone(r.producer_sample_time)
        self.assertIsNone(r.input_age_at_output)
        self.assertIsNone(r.input_age_at_start)
        self.assertFalse(r.freshness_valid)
        self.assertFalse(r.output_valid)
        self.assertEqual(r.invalid_reason, NO_COMPLETED_PRODUCER)
        # The consumer still met its own deadline; that is reported honestly
        # rather than being masked by the missing input.
        self.assertTrue(r.deadline_valid)

    def test_case6_selects_most_recently_completed_not_released(self):
        """Out-of-order completion must select by end_time, not release order.

        instance 0: released 0,  runs long, ends 100  <- most recently COMPLETED
        instance 1: released 10, runs short, ends 20
        Consumer starts at 150. The most recently *released* producer is
        instance 1; the most recently *completed* is instance 0. Selecting by
        release would report a 140 age instead of the correct 150.
        """
        trace = [
            _producer(instance=0, release=0.0, start=0.0, end=100.0),
            _producer(instance=1, release=10.0, start=10.0, end=20.0),
            _consumer(release=150.0, start=150.0, end=151.0, deadline=200.0),
        ]
        r = _one(trace, _edge(freshness_window=1000.0))
        self.assertEqual(r.producer_instance, 0)
        self.assertEqual(r.producer_end_time, 100.0)
        self.assertEqual(r.producer_sample_time, 0.0)
        self.assertEqual(r.input_age_at_output, 151.0)

    def test_case6b_picks_latest_among_several_in_order(self):
        """With in-order completion it must still pick the newest, not the first."""
        trace = [
            _producer(instance=0, release=0.0, start=0.0, end=10.0),
            _producer(instance=1, release=50.0, start=50.0, end=60.0),
            _producer(instance=2, release=100.0, start=100.0, end=110.0),
            # instance 3 has not completed by the consumer's start.
            _producer(instance=3, release=150.0, start=150.0, end=160.0),
            _consumer(release=120.0, start=120.0, end=121.0, deadline=130.0),
        ]
        r = _one(trace, _edge(freshness_window=1000.0))
        self.assertEqual(r.producer_instance, 2)
        self.assertEqual(r.input_age_at_output, 21.0)  # 121 - 100

    def test_case7_producer_ending_exactly_at_consumer_start_is_eligible(self):
        """Inclusive boundary: producer_end_time <= consumer_start_time.

        Equality is the normal case in solver output, not a measure-zero edge
        case, so excluding it would reclassify a whole class of tight schedules
        as no_completed_producer.
        """
        r = _one(
            [
                _producer(release=0.0, start=0.0, end=20.0),
                _consumer(release=20.0, start=20.0, end=21.0, deadline=30.0),
            ]
        )
        self.assertEqual(
            r.producer_instance, 0, "producer finishing exactly at start is eligible"
        )
        self.assertEqual(r.input_age_at_output, 21.0)
        self.assertEqual(r.invalid_reason, VALID)

    def test_case7b_producer_ending_just_after_start_is_not_eligible(self):
        """The other side of the boundary, so the test above pins a boundary
        rather than merely passing for both answers."""
        r = _one(
            [
                _producer(release=0.0, start=0.0, end=20.000001),
                _consumer(release=20.0, start=20.0, end=21.0, deadline=30.0),
            ]
        )
        self.assertIsNone(r.producer_instance)
        self.assertEqual(r.invalid_reason, NO_COMPLETED_PRODUCER)


class FreshnessBoundary(unittest.TestCase):
    def test_age_exactly_at_window_is_fresh(self):
        # age_at_output == phi exactly -> valid (freshness_valid = age <= phi).
        r = _one(
            [
                _producer(release=0.0, start=0.0, end=10.0),
                _consumer(release=49.0, start=49.0, end=PHI, deadline=100.0),
            ]
        )
        self.assertEqual(r.input_age_at_output, PHI)
        self.assertTrue(r.freshness_valid)

    def test_age_just_over_window_is_stale(self):
        r = _one(
            [
                _producer(release=0.0, start=0.0, end=10.0),
                _consumer(release=49.0, start=49.0, end=PHI + 1e-6, deadline=100.0),
            ]
        )
        self.assertFalse(r.freshness_valid)
        self.assertEqual(r.invalid_reason, STALE_INPUT)

    def test_freshness_is_judged_at_output_not_at_start(self):
        """A consumer can start fresh and finish stale. The window is applied to
        age_at_output, because the actuation command is only emitted at the end;
        age_at_start is still recorded for the alternative reading."""
        r = _one(
            [
                _producer(release=0.0, start=0.0, end=10.0),
                _consumer(release=45.0, start=45.0, end=60.0, deadline=100.0),
            ]
        )
        self.assertEqual(r.input_age_at_start, 45.0)  # fresh at start
        self.assertEqual(r.input_age_at_output, 60.0)  # stale at output
        self.assertFalse(r.freshness_valid)


class SampleTimeSemantics(unittest.TestCase):
    def test_sample_at_release_is_the_default(self):
        r = _one(
            [
                _producer(release=5.0, start=12.0, end=20.0),
                _consumer(release=30.0, start=30.0, end=31.0, deadline=40.0),
            ]
        )
        self.assertEqual(r.producer_sample_time, 5.0)
        self.assertEqual(r.input_age_at_output, 26.0)

    def test_sample_at_start_uses_producer_start(self):
        r = _one(
            [
                _producer(release=5.0, start=12.0, end=20.0),
                _consumer(release=30.0, start=30.0, end=31.0, deadline=40.0),
            ],
            _edge(sample_time_semantics="producer_start"),
        )
        self.assertEqual(r.producer_sample_time, 12.0)
        self.assertEqual(r.input_age_at_output, 19.0)

    def test_explicit_sample_time_wins(self):
        """A real sensor timestamp beats deriving one from release."""
        r = _one(
            [
                _producer(release=5.0, start=12.0, end=20.0, sample=1.5),
                _consumer(release=30.0, start=30.0, end=31.0, deadline=40.0),
            ]
        )
        self.assertEqual(r.producer_sample_time, 1.5)
        self.assertEqual(r.input_age_at_output, 29.5)

    def test_rejects_unknown_semantics(self):
        with self.assertRaises(ValueError):
            _edge(sample_time_semantics="whenever")


class Provenance(unittest.TestCase):
    def test_records_mark_producer_attribution_as_inferred(self):
        """Nothing in the stack records which buffer a consumer read, so every
        record must say the attribution was inferred."""
        r = _one([_producer(), _consumer()])
        self.assertEqual(r.producer_instance_provenance, INFERRED)

    def test_context_carries_unit_and_provenance(self):
        ev = evaluate_freshness(
            [_producer(), _consumer()],
            dependency_edges=[_edge()],
            time_unit="us",
            provenance={"timing_source": "firesim_measured", "scaling_factor": 1e-6},
        )
        self.assertEqual(ev.context["time_unit"], "us")
        self.assertEqual(ev.context["producer_instance_provenance"], INFERRED)
        self.assertEqual(
            ev.context["timing_provenance"]["timing_source"], "firesim_measured"
        )


class Aggregates(unittest.TestCase):
    def test_rates_and_reason_partition(self):
        """One of each outcome; rates and counts must agree and the reason
        buckets must partition the total."""
        trace = [
            _producer(instance=0, release=0.0, start=0.0, end=10.0),
            _consumer(instance=0, release=20.0, start=20.0, end=21.0, deadline=30.0),
            _consumer(instance=1, release=100.0, start=100.0, end=101.0, deadline=110.0),
            _consumer(instance=2, release=30.0, start=30.0, end=31.0, deadline=30.5),
            _consumer(instance=3, release=200.0, start=200.0, end=201.0, deadline=200.5),
        ]
        ev = evaluate_freshness(trace, dependency_edges=[_edge()])
        agg = ev.aggregate
        self.assertEqual(agg["total_consumer_invocations"], 4)
        self.assertEqual(agg["valid_count"], 1)            # inst 0
        self.assertEqual(agg["stale_input_count"], 1)      # inst 1
        self.assertEqual(agg["deadline_miss_count"], 1)    # inst 2
        self.assertEqual(agg["deadline_and_stale_count"], 1)  # inst 3
        self.assertEqual(agg["no_producer_count"], 0)
        self.assertEqual(agg["deadline_success_rate"], 0.5)
        self.assertEqual(agg["freshness_success_rate"], 0.5)
        self.assertEqual(agg["output_valid_rate"], 0.25)
        counts = ev.reason_counts()
        self.assertEqual(sum(counts.values()), 4)

    def test_divergence_is_representable(self):
        """The shape the experiment is looking for: every consumer meets its
        deadline, most emit stale output."""
        trace = [_producer(instance=0, release=0.0, start=0.0, end=10.0)]
        for j in range(10):
            t = 100.0 + 10.0 * j
            trace.append(
                _consumer(instance=j, release=t, start=t, end=t + 1.0, deadline=t + 9.0)
            )
        agg = evaluate_freshness(trace, dependency_edges=[_edge()]).aggregate
        self.assertEqual(agg["deadline_success_rate"], 1.0)
        self.assertEqual(agg["freshness_success_rate"], 0.0)
        self.assertEqual(agg["output_valid_rate"], 0.0)

    def test_age_percentiles_exclude_missing_producer(self):
        """no_completed_producer has no age; imputing one would move the
        percentiles, so those records are counted separately."""
        trace = [
            _producer(instance=0, release=0.0, start=0.0, end=100.0),
            # starts before any producer completed -> no age
            _consumer(instance=0, release=10.0, start=10.0, end=11.0, deadline=20.0),
            _consumer(instance=1, release=100.0, start=100.0, end=101.0, deadline=110.0),
        ]
        agg = evaluate_freshness(
            trace, dependency_edges=[_edge(freshness_window=1000.0)]
        ).aggregate
        self.assertEqual(agg["total_consumer_invocations"], 2)
        self.assertEqual(agg["no_producer_count"], 1)
        self.assertEqual(agg["n_with_age"], 1)
        self.assertEqual(agg["max_input_age"], 101.0)
        self.assertEqual(agg["p50_input_age"], 101.0)

    def test_empty_records_do_not_crash(self):
        agg = aggregate_metrics([])
        self.assertEqual(agg["total_consumer_invocations"], 0)
        self.assertIsNone(agg["output_valid_rate"])
        self.assertIsNone(agg["max_input_age"])


class ProducerSelection(unittest.TestCase):
    def test_select_producer_returns_none_when_none_eligible(self):
        self.assertIsNone(select_producer([_producer(end=100.0)], 50.0))

    def test_ties_break_toward_higher_instance(self):
        """Same end_time: the higher instance index is the fresher sample."""
        p = select_producer(
            [
                _producer(instance=0, release=0.0, end=20.0),
                _producer(instance=1, release=5.0, end=20.0),
            ],
            50.0,
        )
        self.assertEqual(p.instance, 1)

    def test_unknown_policy_raises(self):
        with self.assertRaises(ValueError):
            select_producer([_producer()], 50.0, policy="latest_released")


class Validation(unittest.TestCase):
    def test_missing_consumer_task_is_an_error_not_zero_records(self):
        """A declared edge whose consumer never appears is a workload/trace
        mismatch and must fail loudly, not silently score nothing."""
        with self.assertRaises(ValueError) as cm:
            evaluate_freshness([_producer()], dependency_edges=[_edge()])
        self.assertIn("mlp_control", str(cm.exception))

    def test_nonpositive_window_rejected(self):
        with self.assertRaises(ValueError):
            _edge(freshness_window=0.0)

    def test_epoch_is_derived_from_release_and_epoch_length(self):
        trace = [
            _producer(instance=0, release=0.0, start=0.0, end=10.0),
            _consumer(instance=0, release=20.0, start=20.0, end=21.0, deadline=30.0),
            _consumer(instance=1, release=350.0, start=350.0, end=351.0, deadline=360.0),
        ]
        ev = evaluate_freshness(
            trace, dependency_edges=[_edge(freshness_window=1000.0)], epoch_length=300.0
        )
        self.assertEqual([r.epoch for r in ev.records], [0, 1])


class ConfigPlumbing(unittest.TestCase):
    CANON = {
        "networks": {
            "mlp_control": {"identifier": "mlp_control", "criticality": "hard"},
            "dronet": {"identifier": "dronet", "criticality": "hard"},
            "yolov8_nano_64": {"identifier": "yolov8_nano_64", "criticality": "soft"},
        },
        "edges": [],
        "freshness_edges": [
            {
                "producer_task": "dronet",
                "consumer_task": "mlp_control",
                "freshness_window": 70.5,
                "sample_time_semantics": "producer_release",
                "consumption_policy": "latest_completed",
                "criticality": "hard",
            }
        ],
    }

    def test_loads_the_dronet_to_control_edge(self):
        edges = freshness_edges_from_config(self.CANON)
        self.assertEqual(len(edges), 1)
        e = edges[0]
        self.assertEqual(e.producer_task, "dronet")
        self.assertEqual(e.consumer_task, "mlp_control")
        self.assertEqual(e.freshness_window, 70.5)
        self.assertEqual(e.criticality, "hard")

    def test_window_override_drives_the_phi_sweep(self):
        edges = freshness_edges_from_config(self.CANON, freshness_window_override=95.0)
        self.assertEqual(edges[0].freshness_window, 95.0)

    def test_missing_window_with_no_override_is_an_error(self):
        cfg = {"freshness_edges": [{"producer_task": "a", "consumer_task": "b"}]}
        with self.assertRaises(ValueError) as cm:
            freshness_edges_from_config(cfg)
        self.assertIn("freshness_window", str(cm.exception))

    def test_precedence_and_freshness_on_the_same_pair_is_rejected(self):
        """A precedence edge makes the consumer wait, which makes staleness
        impossible. Declaring both on one pair silently destroys the
        measurement, so it must fail loudly."""
        cfg = {
            "edges": [{"from": "dronet", "to": "mlp_control"}],
            "freshness_edges": [
                {
                    "producer_task": "dronet",
                    "consumer_task": "mlp_control",
                    "freshness_window": 50.0,
                }
            ],
        }
        with self.assertRaises(ValueError) as cm:
            freshness_edges_from_config(cfg)
        msg = str(cm.exception)
        self.assertIn("precedence", msg)
        self.assertIn("impossible", msg)

    def test_absent_freshness_edges_yields_none(self):
        self.assertEqual(freshness_edges_from_config({"networks": {}}), [])

    def test_criticality_map(self):
        crit = criticality_from_config(self.CANON)
        self.assertEqual(crit["dronet"], "hard")
        self.assertEqual(crit["mlp_control"], "hard")
        self.assertEqual(crit["yolov8_nano_64"], "soft")

    def test_criticality_defaults_to_soft(self):
        """Defaulting to soft cannot inflate the hard-validity denominator."""
        crit = criticality_from_config(
            {"networks": {"x": {"identifier": "x"}}}
        )
        self.assertEqual(crit["x"], "soft")

    def test_bad_criticality_rejected(self):
        with self.assertRaises(ValueError):
            criticality_from_config(
                {"networks": {"x": {"identifier": "x", "criticality": "critical"}}}
            )


class InstanceNameSplitting(unittest.TestCase):
    TASKS = ("dronet", "mlp_control", "yolov8_nano_64")

    def test_periodic_instance_suffix(self):
        self.assertEqual(split_instance_name("dronet0", self.TASKS), ("dronet", 0))
        self.assertEqual(split_instance_name("dronet5", self.TASKS), ("dronet", 5))
        self.assertEqual(
            split_instance_name("mlp_control29", self.TASKS), ("mlp_control", 29)
        )

    def test_bare_name_is_instance_zero(self):
        self.assertEqual(
            split_instance_name("yolov8_nano_64", self.TASKS), ("yolov8_nano_64", 0)
        )

    def test_model_name_ending_in_digits_is_not_mis_split(self):
        """A trailing-digit regex would read this as ('yolov8_nano_', 64)."""
        task, inst = split_instance_name("yolov8_nano_64", self.TASKS)
        self.assertEqual(task, "yolov8_nano_64")
        self.assertEqual(inst, 0)

    def test_longest_prefix_wins_on_ambiguity(self):
        tasks = ("yolov8_nano", "yolov8_nano_64")
        self.assertEqual(
            split_instance_name("yolov8_nano_640", tasks), ("yolov8_nano_64", 0)
        )
        self.assertEqual(
            split_instance_name("yolov8_nano3", tasks), ("yolov8_nano", 3)
        )

    def test_unknown_name_raises(self):
        with self.assertRaises(ValueError):
            split_instance_name("fastdepth0", self.TASKS)


class AnalyticCrossCheck(unittest.TestCase):
    """The evaluator must agree with closed-form arithmetic. If these fail, the
    evaluator or an adapter is wrong -- not the arithmetic."""

    def test_supremum_formula(self):
        self.assertAlmostEqual(
            analytic_age_supremum(50.0, 18.614, 0.546), 69.16, places=6
        )

    def test_realized_ceiling_on_the_canonical_grid(self):
        """DroNet T=50/L=18.614 feeding MLP T=10/L=0.546.

        Because the consumer period divides the producer period, the supremum
        (69.16) is NOT attained; the realized ceiling is 60.546. This is exactly
        why the freshness window must be anchored on the realized A0 rather than
        on the producer period: phi below 60.546 reports staleness that comes
        from the sampling rate, not from contention.
        """
        r = analytic_age_ceiling_realized(
            producer_period=50.0,
            producer_latency=18.614,
            consumer_period=10.0,
            consumer_latency=0.546,
            horizon=300.0,
        )
        self.assertAlmostEqual(r["A0_realized"], 60.546, places=6)
        self.assertAlmostEqual(r["A0_supremum"], 69.16, places=6)
        self.assertEqual(
            [round(a, 3) for a in r["distinct_ages"]],
            [20.546, 30.546, 40.546, 50.546, 60.546],
        )
        # Consumers at t=0 and t=10 precede the first producer completion.
        self.assertEqual(r["no_producer_count"], 2)

    def test_realized_ceiling_reaches_supremum_on_a_coprime_grid(self):
        """With periods that are not commensurate the realized ceiling
        approaches the supremum, confirming the two formulas are consistent."""
        r = analytic_age_ceiling_realized(
            producer_period=50.0,
            producer_latency=5.0,
            consumer_period=7.0,
            consumer_latency=1.0,
            horizon=5000.0,
        )
        self.assertLessEqual(r["A0_realized"], r["A0_supremum"])
        self.assertGreater(r["A0_realized"], r["A0_supremum"] - 7.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestDigitSuffixNetworksAreNotMisSplit(unittest.TestCase):
    """A network whose own name ends in a digit must survive the split.

    The regression this guards: several call sites split
    `<network><instance>` with a trailing-digit rule -- a `\\d*` regex group or
    `.rstrip("0123456789")`. Both read "vision_v3" as instance 3 of a network
    "vision_v", so every hint, per-model row and candidate comparison for a
    `*_v3` model was addressed to a network that does not exist.
    """

    TASKS = ("vision_v3", "smolvlm_vision_v3", "dronet", "mlp_control",
             "yolov8_nano", "yolov8_nano_64")

    def test_bare_digit_suffix_name_is_instance_zero(self):
        self.assertEqual(split_instance_name("vision_v3", self.TASKS),
                         ("vision_v3", 0))
        self.assertEqual(split_instance_name("smolvlm_vision_v3", self.TASKS),
                         ("smolvlm_vision_v3", 0))

    def test_instances_of_a_digit_suffix_name_still_split(self):
        self.assertEqual(split_instance_name("vision_v3" + "7", self.TASKS),
                         ("vision_v3", 7))

    def test_ordinary_instances_are_unaffected(self):
        self.assertEqual(split_instance_name("dronet0", self.TASKS),
                         ("dronet", 0))
        self.assertEqual(split_instance_name("mlp_control29", self.TASKS),
                         ("mlp_control", 29))

    def test_longest_prefix_beats_the_shorter_registered_name(self):
        # Both "yolov8_nano" and "yolov8_nano_64" are registered; the more
        # specific one wins, so this is instance 0 of the _64 variant and not
        # instance 64 of the base.
        self.assertEqual(split_instance_name("yolov8_nano_64", self.TASKS),
                         ("yolov8_nano_64", 0))

    def test_the_old_trailing_digit_rule_would_have_failed_these(self):
        import re
        bad = re.compile(r"^(?P<net>.+?)(?P<instance>\d*)_dispatch_\d+$")
        self.assertEqual(bad.match("vision_v3_dispatch_6").group("net"),
                         "vision_v")   # the bug, pinned
        self.assertEqual("vision_v3".rstrip("0123456789"), "vision_v")
        # and what the shared splitter does instead
        self.assertEqual(split_instance_name("vision_v3", self.TASKS)[0],
                         "vision_v3")
