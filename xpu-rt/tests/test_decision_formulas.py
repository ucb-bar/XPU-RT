"""Unit tests for analytical decision formulas (Phase B7).

Adversarial fixtures: each test feeds a hand-constructed scenario where
the right answer is known by closed-form arithmetic, then asserts the
formula returns it. Tests cover each of B1-B6 with at least one
positive case (improvement / feasible) and one negative case (no
improvement / infeasible / blocked).
"""

from __future__ import annotations

import math
import os
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))  # xpu-rt/ on path

from decision_formulas import (  # noqa: E402
    frequency_feasibility,
    shard_benefit,
    fuse_benefit,
    unfuse_benefit,
    compaction_eligible,
    critical_path,
)


class TestFrequencyFeasibility(unittest.TestCase):
    def test_single_class_feasible(self):
        """One op @ 5ms on class A, period 10ms => feasible with 5ms slack."""
        ops = {"op0": {"A": 5.0, "B": 8.0}}
        r = frequency_feasibility(ops, period=10.0, machine_classes=["A", "B"])
        self.assertTrue(r.feasible)
        self.assertAlmostEqual(r.per_class_load["A"], 5.0)
        self.assertAlmostEqual(r.per_class_load["B"], 8.0)
        self.assertAlmostEqual(r.min_class_load, 5.0)
        self.assertEqual(r.bottleneck_class, "A")
        self.assertAlmostEqual(r.slack, 5.0)

    def test_single_class_infeasible(self):
        """3 ops each 4ms on A, period 10ms => 12ms load, infeasible."""
        ops = {f"op{i}": {"A": 4.0, "B": 6.0} for i in range(3)}
        r = frequency_feasibility(ops, period=10.0, machine_classes=["A", "B"])
        self.assertFalse(r.feasible)
        self.assertAlmostEqual(r.per_class_load["A"], 12.0)
        self.assertAlmostEqual(r.per_class_load["B"], 18.0)
        self.assertAlmostEqual(r.min_class_load, 12.0)
        self.assertAlmostEqual(r.slack, -2.0)

    def test_multiclass_better_partition(self):
        """op1 cheap on A, op2 cheap on B. Sum of cheapest < either single class."""
        ops = {
            "op1": {"A": 1.0, "B": 9.0},
            "op2": {"A": 9.0, "B": 1.0},
        }
        r = frequency_feasibility(ops, period=10.0, machine_classes=["A", "B"])
        # Single class A: 10ms; single class B: 10ms; multiclass: 2ms.
        self.assertAlmostEqual(r.multiclass_partition_load, 2.0)
        self.assertEqual(r.multiclass_partition["op1"], "A")
        self.assertEqual(r.multiclass_partition["op2"], "B")

    def test_op_only_on_one_class(self):
        """An op missing on class A => per_class_load[A]=inf, A infeasible."""
        ops = {"op0": {"B": 5.0}}
        r = frequency_feasibility(ops, period=10.0, machine_classes=["A", "B"])
        self.assertTrue(math.isinf(r.per_class_load["A"]))
        self.assertAlmostEqual(r.per_class_load["B"], 5.0)
        self.assertTrue(r.feasible)
        self.assertEqual(r.bottleneck_class, "B")


class TestShardBenefit(unittest.TestCase):
    def test_symmetric_speeds_harmonic_mean(self):
        """home=alt=10ms, no contention: optimal f=0.5, finish=5ms."""
        r = shard_benefit(home_cost=10.0, alt_cost=10.0, alt_machine="rvv")
        self.assertAlmostEqual(r.best_fraction, 0.5)
        self.assertAlmostEqual(r.optimal_finish_no_contention, 5.0)
        # parallel_finish at f=0.5 = max(5, 5) = 5.
        # expected_delta = 5 - 10 = -5 (improvement of 5).
        self.assertAlmostEqual(r.expected_delta, -5.0)

    def test_asymmetric_speeds(self):
        """home=2, alt=8: optimal f=2/10=0.2, finish=2*8/10=1.6."""
        r = shard_benefit(home_cost=2.0, alt_cost=8.0, alt_machine="rvv")
        self.assertAlmostEqual(r.best_fraction, 0.2)
        self.assertAlmostEqual(r.optimal_finish_no_contention, 1.6)
        self.assertAlmostEqual(r.expected_delta, -0.4)  # 1.6 - 2.0

    def test_contention_eats_benefit(self):
        """home=2, alt=8, contention=2: f*=(2-2)/10=0 => no shard."""
        r = shard_benefit(home_cost=2.0, alt_cost=8.0, alt_machine="rvv",
                          alt_soonest_free=2.0, op_ready=0.0)
        self.assertAlmostEqual(r.best_fraction, 0.0)
        self.assertAlmostEqual(r.expected_delta, 0.0)
        self.assertIn("contended", (r.rejection_reason or ""))

    def test_partial_contention(self):
        """home=2, alt=8, contention=1: f*=(2-1)/10=0.1
        home_branch = 0.9*2 = 1.8; alt_branch = 1+0.1*8=1.8; parallel_finish=1.8.
        expected_delta = 1.8 - 2 = -0.2."""
        r = shard_benefit(home_cost=2.0, alt_cost=8.0, alt_machine="rvv",
                          alt_soonest_free=1.0, op_ready=0.0)
        self.assertAlmostEqual(r.best_fraction, 0.1)
        self.assertAlmostEqual(r.expected_delta, -0.2, places=5)


class TestFuseBenefit(unittest.TestCase):
    def test_same_machine_pure_dispatch_save(self):
        """Both on same machine, no reuse. expected_delta = -dispatch_overhead."""
        r = fuse_benefit(op1_cost=10.0, op2_cost=5.0,
                          op1_machine="A", op2_machine="A",
                          dispatch_overhead=2.0)
        # fused_cost defaulted to op1+op2 = 15
        # expected = 15 - 15 - 2 - 0 + 0 = -2
        self.assertAlmostEqual(r.expected_delta, -2.0)
        self.assertEqual(r.parallelism_cost, 0.0)

    def test_cross_machine_parallelism_cost(self):
        """op1 on A, op2 on B. Fusing serializes them.
        fused_cost = 15 (no kernel speedup), op1+op2=15, dispatch=2, reuse=0,
        parallelism_cost = min(10,5) = 5.
        expected_delta = 15 - 15 - 2 + 5 = +3 (loss)."""
        r = fuse_benefit(op1_cost=10.0, op2_cost=5.0,
                          op1_machine="A", op2_machine="B",
                          dispatch_overhead=2.0)
        self.assertAlmostEqual(r.parallelism_cost, 5.0)
        self.assertAlmostEqual(r.expected_delta, 3.0)

    def test_kernel_speedup_overrides_loss(self):
        """fused_cost=8 instead of 15 (kernel fuses inner loops).
        Same-machine, no reuse, dispatch=2.
        expected = 8 - 15 - 2 = -9 (big win)."""
        r = fuse_benefit(op1_cost=10.0, op2_cost=5.0,
                          op1_machine="A", op2_machine="A",
                          fused_cost=8.0, dispatch_overhead=2.0)
        self.assertAlmostEqual(r.expected_delta, -9.0)

    def test_data_reuse_save(self):
        """intermediate=1024 bytes, bw=1024 bytes/us => save 1us."""
        r = fuse_benefit(op1_cost=10.0, op2_cost=5.0,
                          op1_machine="A", op2_machine="A",
                          dispatch_overhead=0.0,
                          intermediate_bytes=1024.0,
                          mem_bw_per_us=1024.0)
        # expected = 15 - 15 - 0 - 1 + 0 = -1
        self.assertAlmostEqual(r.expected_delta, -1.0)


class TestUnfuseBenefit(unittest.TestCase):
    def test_no_alt_machine_pure_loss(self):
        """No alt for op2 => only dispatch overhead added."""
        r = unfuse_benefit(fused_cost=10.0, op1_cost=6.0, op2_cost=4.0,
                            on_same_machine_now=True,
                            alt_machine_for_op2_cost=None,
                            dispatch_overhead=2.0)
        self.assertAlmostEqual(r.expected_delta, 2.0)  # pure loss

    def test_parallel_alt_wins(self):
        """fused=10, alt cost for op2=3. parallel_finish=max(6,3)=6.
        gain=10-6=4. expected = -4 + dispatch(1) + 0 = -3 (win)."""
        r = unfuse_benefit(fused_cost=10.0, op1_cost=6.0, op2_cost=4.0,
                            on_same_machine_now=True,
                            alt_machine_for_op2_cost=3.0,
                            dispatch_overhead=1.0)
        self.assertAlmostEqual(r.parallelism_cost, 4.0)
        self.assertAlmostEqual(r.expected_delta, -3.0)


class TestCompactionEligible(unittest.TestCase):
    def test_gap_present_release_blocks(self):
        """op_start=10, release=10, dep_finishes empty, machine_busy=0.
        earliest=10, gap=0 => not applicable."""
        r = compaction_eligible(op_start=10.0, op_duration=1.0,
                                 op_release=10.0)
        self.assertFalse(r.applicable)
        self.assertEqual(r.gap, 0.0)

    def test_gap_present_full_shift(self):
        """op_start=10, release=2, dep_finishes=[5], machine_busy=3.
        earliest=max(2,5,3)=5, gap=5 => applicable."""
        r = compaction_eligible(op_start=10.0, op_duration=1.0,
                                 op_release=2.0,
                                 dep_finishes=[5.0],
                                 machine_last_busy_before=3.0)
        self.assertTrue(r.applicable)
        self.assertAlmostEqual(r.gap, 5.0)

    def test_machine_busy_dominates(self):
        """op_start=10, release=0, dep_finishes=[2], machine_busy=8.
        earliest=8, gap=2, blocked_by=machine_busy."""
        r = compaction_eligible(op_start=10.0, op_duration=1.0,
                                 op_release=0.0,
                                 dep_finishes=[2.0],
                                 dep_names=["d0"],
                                 machine_last_busy_before=8.0)
        self.assertTrue(r.applicable)
        self.assertAlmostEqual(r.gap, 2.0)

    def test_own_deadline_caps_shift(self):
        """op_start=10, duration=1, op_max_end=11. New finish would be
        10-gap+1 = stays ≤11 even at gap=0. Test that 'applicable' works
        when there IS a gap but shift respects band."""
        r = compaction_eligible(op_start=10.0, op_duration=1.0,
                                 op_release=2.0, op_max_end=11.0,
                                 dep_finishes=[5.0])
        # earliest=5, gap=5. After shift: new_finish = 10-5+1 = 6 ≤ 11. OK.
        self.assertTrue(r.applicable)
        self.assertAlmostEqual(r.gap, 5.0)


class TestCriticalPath(unittest.TestCase):
    def test_linear_chain(self):
        """a -> b -> c with durations 2,3,1. CP = a->b->c, length=6."""
        r = critical_path(
            ops=["a", "b", "c"],
            durations={"a": 2.0, "b": 3.0, "c": 1.0},
            edges=[("a", "b"), ("b", "c")],
        )
        self.assertEqual(r.path, ["a", "b", "c"])
        self.assertAlmostEqual(r.length, 6.0)
        for op in ["a", "b", "c"]:
            self.assertTrue(r.on_path[op])

    def test_diamond_picks_heavier(self):
        """s -> {a,b} -> t. a=5, b=2. Path s->a->t."""
        r = critical_path(
            ops=["s", "a", "b", "t"],
            durations={"s": 1.0, "a": 5.0, "b": 2.0, "t": 1.0},
            edges=[("s", "a"), ("s", "b"), ("a", "t"), ("b", "t")],
        )
        self.assertEqual(r.path, ["s", "a", "t"])
        self.assertAlmostEqual(r.length, 7.0)
        self.assertFalse(r.on_path["b"])

    def test_two_independent_chains(self):
        """Two disjoint chains. Heavier chain wins."""
        r = critical_path(
            ops=["a1", "a2", "b1"],
            durations={"a1": 1.0, "a2": 2.0, "b1": 4.0},
            edges=[("a1", "a2")],
        )
        # path could be "b1" alone (length 4) or "a1 -> a2" (length 3).
        self.assertAlmostEqual(r.length, 4.0)
        self.assertEqual(r.path, ["b1"])

    def test_empty(self):
        r = critical_path(ops=[], durations={}, edges=[])
        self.assertEqual(r.path, [])
        self.assertEqual(r.length, 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
