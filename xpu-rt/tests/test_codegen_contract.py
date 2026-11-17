#!/usr/bin/env python3
"""ModelBlaster's codegen contract, applied to the scheduler.

A convolution's weights are packed for its core width at generation time, so
the solver must give every instance of one convolution the same width, and a
width that divides its output channels. Linears are sliced at run time and may
vary freely.

    python3 xpu-rt/tests/test_codegen_contract.py
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import codegen_contract as cc  # noqa: E402
from workload_factory import build_machine_combinations  # noqa: E402

CONTRACT = {"rules": {
    "uniform_width_across_instances": {"applies_to_ops": ["conv2d_s8", "conv2d_batchnorm2d_s8"]},
    "width_divides_output_channels": {"applies_to_ops": ["conv2d_s8", "conv2d_batchnorm2d_s8"]},
    "runtime_sliceable_ops": {"ops": ["linear_s8"]},
}}
CONV = "dronet$dispatch_3_rvv_k3_x100_conv2d_batchnorm2d_s8_N1xIC32xIH27xIW27xOC6xOH14xOW14xKH3xKW3"
LIN = "mlp_control$dispatch_0_rvv_k3_x100_linear_s8_M1xK16xN256"


class _Op:
    def __init__(self, name, did, times):
        self.operation_name, self.operation_id, self.processing_times = name, did, list(times)

    def get_duration_for_combination(self, k, combos, machines):
        return self.processing_times[k]


class _Workload:
    def __init__(self, ops, combos, machines):
        self.operations, self._c, self.machines = ops, combos, machines

    def get_machine_combinations(self):
        return self._c


class Parsing(unittest.TestCase):
    def test_op_is_the_longest_delimited_match(self):
        self.assertEqual(cc.op_of(CONV, cc.uniform_width_ops(CONTRACT)), "conv2d_batchnorm2d_s8")
        self.assertEqual(cc.op_of(LIN, cc.uniform_width_ops(CONTRACT)), "")

    def test_oc(self):
        self.assertEqual(cc.oc_of(CONV), 6)
        self.assertIsNone(cc.oc_of(LIN))


class ApplyToWorkload(unittest.TestCase):
    def setUp(self):
        self.machines, self.combos = build_machine_combinations({"CPU_P": 4}, mode="shard")
        # widths: 1,1,1,1,2,2,4 ; conv instance 0 is fastest on 2 cores, instance 1 on 1
        t = {1: 10.0, 2: 6.0, 4: 5.0}
        a = [t[len(c)] for c in self.combos]
        b = [t[len(c)] - (5.5 if len(c) == 1 else 0) for c in self.combos]
        self.ops = [_Op("dronet0_dispatch_3", 3, a), _Op("dronet1_dispatch_3", 3, b),
                    _Op("mlp_control0_dispatch_0", 0, a), _Op("mlp_control1_dispatch_0", 0, b)]
        self.wl = _Workload(self.ops, self.combos, self.machines)

        def mod(op):
            return (CONV, "dronet") if op.operation_name.startswith("dronet") else (LIN, "mlp_control")
        self.stats = cc.apply_to_workload(self.wl, mod, CONTRACT)

    def _open_widths(self, op):
        return {len(c) for c, t in zip(self.combos, op.processing_times) if t < cc.PRICED_OUT_MS}

    def test_width_not_dividing_oc_is_excluded(self):
        # OC=6: width 4 never divides it
        self.assertNotIn(4, self._open_widths(self.ops[0]))
        self.assertEqual(self.stats["oc_excluded_options"], 2)

    def test_conv_instances_share_one_width_chosen_by_cost(self):
        # summed cost: width 1 -> 10 + 4.5 = 14.5, width 2 -> 6 + 6 = 12
        self.assertEqual(self._open_widths(self.ops[0]), {2})
        self.assertEqual(self._open_widths(self.ops[1]), {2})
        self.assertEqual(self.stats["pinned_dispatches"], 1)

    def test_linears_are_left_alone(self):
        self.assertEqual(self._open_widths(self.ops[2]), {1, 2, 4})


class Violations(unittest.TestCase):
    def test_reports_mixed_widths_and_oc(self):
        d = {"dronet0_dispatch_3": {"id": 3, "module_name": CONV, "hardware_target": "CPU_P#0+CPU_P#1"},
             "dronet1_dispatch_3": {"id": 3, "module_name": CONV,
                                    "hardware_target": "CPU_P#0+CPU_P#1+CPU_P#2+CPU_P#3"},
             "mlp_control0_dispatch_0": {"id": 0, "module_name": LIN, "hardware_target": "CPU_P#0"},
             "mlp_control1_dispatch_0": {"id": 0, "module_name": LIN,
                                         "hardware_target": "CPU_P#0+CPU_P#1"}}
        v = cc.violations(d, CONTRACT)
        self.assertEqual(len(v), 2, v)
        self.assertTrue(any("widths [2, 4]" in x for x in v))
        self.assertTrue(any("OC=6 on 4 cores" in x for x in v))

    def test_missing_contract_is_an_error(self):
        with self.assertRaises(cc.ContractUnavailable):
            cc.load("/nonexistent/codegen_contract.json")


if __name__ == "__main__":
    unittest.main()
