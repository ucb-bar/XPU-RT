"""The trace's `dispatch_id` is a record SLOT, not the IR's dispatch id.

`generate_skeleton` sizes the harness's profile record array by the ops that
emit a kernel call; `view` and the `chunk2_c1` family emit none. The harness
stamps each record with its slot in that array and the column is called
`dispatch_id` -- so it drifts from the IR numbering that the schedule, the
profile CSV and the advice all use, by the number of zero-cost ops seen so far.

The failure is silent and reads as a PREDICTION error, which is what makes it
convincing. Measured on the 3-model 4 Hz run:

    joining on the raw id     yolov8_nano0_dispatch_81  pred 17.465  "meas" 0.577  -96.8%
    joining on the IR id      yolov8_nano0_dispatch_81  pred 17.465   meas 18.661   +6.8%

and across the run, median relative error -3.6% -> -2.4%, mean +204.4% ->
-2.6%, worst |error| 16.9 ms -> 2.2 ms. Both sets of numbers are real; only one
set compares an op against itself.
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import k1_trace  # noqa: E402


def _ir(kinds):
    return {"name": "m",
            "ops": [{"dispatch_id": i, "op": k, "name": f"op{i}"}
                    for i, k in enumerate(kinds)]}


def _row(slot, network="m", instance="0", op="conv2d_s8", start=0, end=24000):
    return {"network": network, "instance": instance, "dispatch_id": str(slot),
            "op": op, "name": f"n{slot}",
            "actual_start_cycles": str(start), "actual_end_cycles": str(end),
            "predicted_start_ms": "0", "predicted_duration_ms": "1.0",
            "core_kind": "rvv", "hart": "1"}


class SlotMapSkipsExactlyTheZeroCostOps(unittest.TestCase):

    def test_a_graph_with_no_zero_cost_ops_maps_identically(self):
        """dronet and mlp_control are this case, which is why every earlier
        per-dispatch validation on this path was clean and the drift went
        unnoticed."""
        m = k1_trace.ir_slot_map(_ir(["conv2d_s8"] * 5))
        self.assertEqual(m, {i: i for i in range(5)})

    def test_each_zero_cost_op_shifts_every_later_slot_by_one(self):
        m = k1_trace.ir_slot_map(
            _ir(["conv2d_s8", "conv2d_s8", "view", "conv2d_s8",
                 "chunk2_c1", "conv2d_s8"]))
        #  slot: 0->0, 1->1, then `view` at IR 2 is skipped, so 2->3,
        #  `chunk2_c1` at IR 4 is skipped, so 3->5.
        self.assertEqual(m, {0: 0, 1: 1, 2: 3, 3: 5})

    def test_the_map_covers_every_kernel_emitting_op_and_nothing_else(self):
        kinds = ["conv2d_s8", "view", "chunk2_c1_s8", "relu_s8"]
        m = k1_trace.ir_slot_map(_ir(kinds))
        self.assertEqual(len(m), 2)
        self.assertEqual(sorted(m.values()), [0, 3])


class NormaliseTranslatesOnlyWhenItCan(unittest.TestCase):

    def test_with_a_slot_map_the_id_becomes_the_ir_id(self):
        rows = k1_trace.normalise(
            [_row(2)], {"m": k1_trace.ir_slot_map(
                _ir(["conv2d_s8", "view", "conv2d_s8", "conv2d_s8"]))})
        self.assertEqual(rows[0]["dispatch_id"], 3)
        self.assertEqual(rows[0]["trace_slot"], 2)
        self.assertTrue(rows[0]["dispatch_id_is_ir"])
        self.assertEqual(rows[0]["dispatch_key"], "m0_dispatch_3")

    def test_without_one_the_slot_passes_through_and_says_so(self):
        """Silently passing a slot off as an IR id is the failure; passing it
        through while recording that no translation happened is not."""
        rows = k1_trace.normalise([_row(2)])
        self.assertEqual(rows[0]["dispatch_id"], "2")
        self.assertEqual(rows[0]["trace_slot"], 2)
        self.assertFalse(rows[0]["dispatch_id_is_ir"])

    def test_a_merlin_trace_is_passed_through_untouched(self):
        merlin = [{"dispatch_key": "dronet0_dispatch_0", "start_us": "1",
                   "run_us": "2", "job_name": "dronet0"}]
        self.assertEqual(k1_trace.normalise(merlin), merlin)


class TimeConversionUsesTheBoardsClock(unittest.TestCase):

    def test_ticks_are_24_mhz_not_1_mhz(self):
        self.assertEqual(k1_trace.K1_RDTIME_HZ, 24_000_000.0)
        rows = k1_trace.normalise([_row(0, start=0, end=24_000)])
        self.assertAlmostEqual(rows[0]["run_us"], 1000.0)

    def test_the_axis_starts_at_the_runs_own_t0(self):
        """rdtime is free-running; its absolute value is boot time."""
        rows = k1_trace.normalise([_row(0, start=1_000_000, end=1_024_000),
                                   _row(1, start=1_024_000, end=1_048_000)])
        self.assertAlmostEqual(rows[0]["start_us"], 0.0)
        self.assertAlmostEqual(rows[1]["start_us"], 1000.0)


if __name__ == "__main__":
    unittest.main()
