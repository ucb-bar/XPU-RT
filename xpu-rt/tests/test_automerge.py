"""Tests for automerge_adjacent (automerge.py)."""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from automerge import automerge_adjacent, automerge_savings  # noqa: E402


def _entry(start, duration, job_name="mlp_control0", hw="CPU_E#0",
           deps=None, time_dep=None, op_id=0):
    e = {
        "id": op_id, "ordinal": 1, "total": 1,
        "dependencies": list(deps or []),
        "hardware_target": hw,
        "start_time": float(start),
        "duration": float(duration),
        "job_name": job_name,
    }
    if time_dep is not None:
        e["time_dependency"] = time_dep
    return e


def _fixture(entries):
    """entries: dict[name -> entry]."""
    fx = {
        "dispatches": entries,
        "metadata": {
            "makespan": max(e["start_time"] + e["duration"] for e in entries.values()),
            "num_operations": len(entries),
            "machines": ["CPU_P#0", "CPU_E#0"],
        },
    }
    return fx


class AutoMergeBasicTest(unittest.TestCase):

    def test_adjacent_pair_merged(self):
        # Two mlp_control0 ops back-to-back on CPU_E#0; gap=5µs.
        fx = _fixture({
            "mlp_control0_op0": _entry(0.0, 100.0, op_id=0),
            "mlp_control0_op1": _entry(105.0, 80.0,
                                       deps=["mlp_control0_op0"], op_id=1),
        })
        out = automerge_adjacent(fx, max_gap_us=50.0, saved_handshake_us=2.0)
        self.assertEqual(len(out["dispatches"]), 1)
        merged = next(iter(out["dispatches"].values()))
        # End time preserved (105 + 80 = 185) minus saved handshake (2).
        self.assertAlmostEqual(merged["duration"], 183.0)
        self.assertEqual(merged["merged_with"], ["mlp_control0_op1"])
        self.assertTrue(merged["is_fused"])

    def test_different_networks_not_merged(self):
        fx = _fixture({
            "mlp_control0_op0": _entry(0.0, 100.0, job_name="mlp_control0", op_id=0),
            "dronet0_op0": _entry(101.0, 80.0, job_name="dronet0", op_id=1),
        })
        out = automerge_adjacent(fx, max_gap_us=50.0)
        self.assertEqual(len(out["dispatches"]), 2)

    def test_different_hw_not_merged(self):
        fx = _fixture({
            "mlp_control0_op0": _entry(0.0, 100.0, hw="CPU_E#0", op_id=0),
            "mlp_control0_op1": _entry(101.0, 80.0, hw="CPU_P#0",
                                       deps=["mlp_control0_op0"], op_id=1),
        })
        out = automerge_adjacent(fx)
        self.assertEqual(len(out["dispatches"]), 2)

    def test_gap_too_large_not_merged(self):
        fx = _fixture({
            "mlp_control0_op0": _entry(0.0, 100.0, op_id=0),
            "mlp_control0_op1": _entry(300.0, 80.0,
                                       deps=["mlp_control0_op0"], op_id=1),
        })
        out = automerge_adjacent(fx, max_gap_us=50.0)
        self.assertEqual(len(out["dispatches"]), 2)

    def test_external_reader_blocks_merge(self):
        # dronet0_op0 also reads mlp_control0_op0 — merging would orphan
        # that dependency.
        fx = _fixture({
            "mlp_control0_op0": _entry(0.0, 100.0, op_id=0),
            "mlp_control0_op1": _entry(105.0, 80.0,
                                       deps=["mlp_control0_op0"], op_id=1),
            "dronet0_op0": _entry(200.0, 50.0, job_name="dronet0",
                                  hw="CPU_P#0",
                                  deps=["mlp_control0_op0"], op_id=2),
        })
        out = automerge_adjacent(fx)
        self.assertEqual(len(out["dispatches"]), 3)

    def test_downstream_dep_rewritten_to_external(self):
        # op2 has an external dep (dronet0), so it can't be merged into
        # the op0+op1 pair. The rewiring should kick in instead.
        fx = _fixture({
            "mlp_control0_op0": _entry(0.0, 100.0, op_id=0),
            "mlp_control0_op1": _entry(105.0, 80.0,
                                       deps=["mlp_control0_op0"], op_id=1),
            "dronet0_op0": _entry(50.0, 30.0, job_name="dronet0",
                                  hw="CPU_P#0", op_id=2),
            "mlp_control0_op2": _entry(190.0, 30.0,
                                       deps=["mlp_control0_op1", "dronet0_op0"],
                                       op_id=3),
        })
        out = automerge_adjacent(fx, max_gap_us=50.0)
        # op0 + op1 merge → op2 stays separate (multi-dep), but its dep
        # list should be rewired from op1 → op0.
        self.assertEqual(len(out["dispatches"]), 3)
        op2 = out["dispatches"]["mlp_control0_op2"]
        self.assertIn("mlp_control0_op0", op2["dependencies"])
        self.assertNotIn("mlp_control0_op1", op2["dependencies"])

    def test_full_chain_collapsed(self):
        # Three back-to-back same-network same-hw ops with no external
        # readers — all three merge into a single fused dispatch.
        fx = _fixture({
            "mlp_control0_op0": _entry(0.0, 100.0, op_id=0),
            "mlp_control0_op1": _entry(105.0, 80.0,
                                       deps=["mlp_control0_op0"], op_id=1),
            "mlp_control0_op2": _entry(190.0, 30.0,
                                       deps=["mlp_control0_op1"], op_id=2),
        })
        out = automerge_adjacent(fx, max_gap_us=50.0, saved_handshake_us=2.0)
        self.assertEqual(len(out["dispatches"]), 1)
        merged = next(iter(out["dispatches"].values()))
        # End of chain is 220; duration = end - start - saved = 218.
        # (saved_handshake_us applied once per pair-collapse step;
        # the chain converges to "end minus start minus latest saving",
        # so we save 2µs total here, not 2×2µs — that's intentional, a
        # conservative-side simplification.)
        self.assertAlmostEqual(merged["duration"], 218.0)
        self.assertEqual(len(merged["merged_with"]), 2)

    def test_savings_summary(self):
        fx = _fixture({
            "mlp_control0_op0": _entry(0.0, 100.0, op_id=0),
            "mlp_control0_op1": _entry(105.0, 80.0,
                                       deps=["mlp_control0_op0"], op_id=1),
        })
        out = automerge_adjacent(fx, max_gap_us=50.0, saved_handshake_us=2.0)
        s = automerge_savings(fx, out)
        self.assertEqual(s["dispatches_before"], 2)
        self.assertEqual(s["dispatches_after"], 1)
        self.assertEqual(s["pairs_merged"], 1)
        self.assertAlmostEqual(s["saved_us"], 2.0, places=2)


if __name__ == "__main__":
    unittest.main()
