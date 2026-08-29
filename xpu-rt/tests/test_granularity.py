"""Tests for the merge-vs-split decision logic (scripts/granularity_loop.py)."""

from __future__ import annotations

import os
import sys
import unittest

SCRIPTS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))
sys.path.insert(0, SCRIPTS)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import granularity_loop as G


def _merge(name, dmake, ddisp, obj):
    return {"id": name, "type": "fuse_linear_chain", "affected": [f"{name}_dispatch_0", f"{name}_dispatch_1"],
            "makespan_delta_us": dmake, "dispatch_delta": ddisp, "obj_delta_us": obj}


def _split(name, dmake):
    return {"id": name, "type": "split_heavy_dispatch", "affected": [f"{name}_dispatch_0"],
            "makespan_delta_us": dmake, "dispatch_delta": 1, "obj_delta_us": dmake + 1}


class DecideTests(unittest.TestCase):
    def test_split_wins_when_it_lowers_makespan(self):
        merges = [_merge("mlp_control0", 0.0, -5, -5.0)]
        splits = [_split("yolov8_nano0", -3.0)]
        decision, chosen, _ = G.decide(merges, splits, "too_fine")
        self.assertEqual(decision, "split")
        self.assertEqual(chosen["id"], "yolov8_nano0")

    def test_merge_wins_when_too_fine_and_no_useful_split(self):
        merges = [_merge("mlp_control3", -5.0, -5, -10.0)]
        splits = [_split("yolov8_nano0", 0.0)]   # split doesn't help
        decision, chosen, _ = G.decide(merges, splits, "too_fine")
        self.assertEqual(decision, "merge")
        self.assertEqual(chosen["id"], "mlp_control3")

    def test_none_when_balanced(self):
        merges = [_merge("mlp_control0", 0.0, -2, -2.0)]
        splits = [_split("yolov8_nano0", 0.0)]
        decision, chosen, _ = G.decide(merges, splits, "balanced")
        self.assertEqual(decision, "none")
        self.assertIsNone(chosen)

    def test_merge_hint_uses_local_dispatch_ids_per_network(self):
        chosen = {"affected": ["mlp_control3_dispatch_0", "mlp_control3_dispatch_1",
                               "mlp_control3_dispatch_2"], "rationale": "x"}
        hint = G.build_hint("merge", chosen)
        self.assertEqual(hint["contract"], "modelblaster.fusion_hints/v1")
        net = hint["networks"][0]
        self.assertEqual(net["network"], "mlp_control")          # instance index stripped
        self.assertEqual(net["fuse_groups"], [[0, 1, 2]])        # local dispatch ids

    def test_split_hint_shape(self):
        chosen = {"affected": ["dronet0_dispatch_4"], "rationale": "x"}
        hint = G.build_hint("split", chosen)
        self.assertEqual(hint["contract"], "modelblaster.split_hints/v1")
        self.assertEqual(hint["networks"][0]["split_ops"], [{"op": 4, "n_splits": 2}])


if __name__ == "__main__":
    unittest.main()
