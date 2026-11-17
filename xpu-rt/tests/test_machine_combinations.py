#!/usr/bin/env python3
"""Topologies the solver can choose from (scheduler.machine_combination_mode).

`shard` must let one dispatch take an aligned power-of-two block of cores and
let disjoint blocks run at the same time; `prefix` keeps its historical shape
so specs that never set the mode schedule exactly as before.

    python3 xpu-rt/tests/test_machine_combinations.py
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from workload_factory import (aligned_core_blocks, build_machine_combinations,  # noqa: E402
                              expand_combinations_with_impls)


def _overlap(a, b):
    return bool(set(a) & set(b))


class Shard(unittest.TestCase):
    def setUp(self):
        _, self.combos = build_machine_combinations({"CPU_P": 8, "CPU_E": 8}, mode="shard")

    def test_widths_are_powers_of_two_and_kinds_never_mix(self):
        for c in self.combos:
            self.assertIn(len(c), (1, 2, 4, 8))
            self.assertEqual(len({x.split("#")[0] for x in c}), 1, c)

    def test_blocks_are_aligned_and_disjoint_or_nested(self):
        p = [c for c in self.combos if c[0].startswith("CPU_P")]
        for a in p:
            idx = [int(x.split("#")[1]) for x in a]
            self.assertEqual(idx[0] % len(a), 0, a)
            self.assertEqual(idx, list(range(idx[0], idx[0] + len(a))), a)
            for b in p:
                if _overlap(a, b):
                    self.assertTrue(set(a) <= set(b) or set(b) <= set(a), (a, b))

    def test_two_four_core_groups_can_run_side_by_side(self):
        fours = [c for c in self.combos if len(c) == 4 and c[0].startswith("CPU_P")]
        self.assertEqual(fours, [[f"CPU_P#{i}" for i in range(4)],
                                 [f"CPU_P#{i}" for i in range(4, 8)]])
        self.assertFalse(_overlap(*fours))

    def test_count(self):
        # per 8-core kind: 8 singles + 4 pairs + 2 quads + 1 octet
        self.assertEqual(len(self.combos), 2 * 15)

    def test_odd_core_count_keeps_every_core(self):
        self.assertEqual(aligned_core_blocks(["a", "b", "c"]),
                         [["a"], ["b"], ["c"], ["a", "b"]])


class PrefixAndSingletons(unittest.TestCase):
    def test_prefix_is_the_default_and_unchanged(self):
        _, default = build_machine_combinations({"CPU_P": 4})
        _, prefix = build_machine_combinations({"CPU_P": 4}, mode="prefix")
        self.assertEqual(default, prefix)
        self.assertEqual(prefix, [["CPU_P#0"], ["CPU_P#0", "CPU_P#1"],
                                  ["CPU_P#0", "CPU_P#1", "CPU_P#2"],
                                  ["CPU_P#0", "CPU_P#1", "CPU_P#2", "CPU_P#3"],
                                  ["CPU_P#1"], ["CPU_P#2"], ["CPU_P#3"]])

    def test_singletons(self):
        _, c = build_machine_combinations({"CPU_P": 3}, mode="singletons")
        self.assertEqual(c, [["CPU_P#0"], ["CPU_P#1"], ["CPU_P#2"]])

    def test_unknown_mode_is_refused(self):
        with self.assertRaises(ValueError):
            build_machine_combinations({"CPU_P": 2}, mode="every_subset")


class Implementations(unittest.TestCase):
    """One core group, several implementations (the K3 A100: IME or RVV)."""

    def setUp(self):
        _, combos = build_machine_combinations({"CPU_P": 2, "CPU_E": 2}, mode="shard")
        hw = ["rvv_k3_x100" if c[0].startswith("CPU_P") else "rvv_k3_a100" for c in combos]
        self.base = combos
        self.c, self.hw, self.impl = expand_combinations_with_impls(
            combos, hw, {"cpu_e": {"ime": "ime_k3_a100", "rvv": "rvv_k3_a100"}})

    def test_each_group_of_an_impl_kind_appears_once_per_impl(self):
        e = [c for c in self.base if c[0].startswith("CPU_E")]
        p = [c for c in self.base if c[0].startswith("CPU_P")]
        self.assertEqual(len(self.c), len(p) + 2 * len(e))
        for c, hw, impl in zip(self.c, self.hw, self.impl):
            if c[0].startswith("CPU_E"):
                self.assertEqual(hw, {"ime": "ime_k3_a100", "rvv": "rvv_k3_a100"}[impl])
            else:
                self.assertIsNone(impl)
                self.assertEqual(hw, "rvv_k3_x100")

    def test_copies_keep_the_same_cores(self):
        pairs = {}
        for c, impl in zip(self.c, self.impl):
            if impl:
                pairs.setdefault(tuple(c), set()).add(impl)
        self.assertTrue(pairs)
        self.assertTrue(all(v == {"ime", "rvv"} for v in pairs.values()))


if __name__ == "__main__":
    unittest.main()
