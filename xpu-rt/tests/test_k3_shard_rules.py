"""K3 shard rules in the codegen contract (k3-work TASK-016).

* shareable_ops (hard): an op without a ModelBlaster pool path gets no
  multi-core option, whatever its profile says.
* conv_split_axis: a K3 conv2d_s8 with enough output rows is split by rows,
  so a width that does not divide OC stays open.
* width margin (soft): a wider width is kept only if >= margin faster than the
  next narrower width kept; a conv's one width is the widest kept.
* The repo's own contract (zephyr-chipyard-sw/modelblaster/cores/...) carries
  the same rules, and schedules checked by `violations` flag both breaches.
"""
from __future__ import annotations

import json
import os
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

import codegen_contract as cc  # noqa: E402
from workload_factory import build_machine_combinations  # noqa: E402

CONTRACT = {"rules": {
    "uniform_width_across_instances": {"applies_to_ops": ["conv2d_s8"]},
    "width_divides_output_channels": {"applies_to_ops": ["conv2d_s8"]},
    "conv_split_axis": {"applies_to_ops": ["conv2d_s8"], "backend_prefixes": ["rvv_k3", "ime_k3"],
                        "min_pixels": 36},
    "shareable_ops": {"ops": ["conv2d_s8", "linear_s8"]},
}}
ROWCONV = "yolov8_nano$dispatch_4_ime_k3_x100_conv2d_s8_N1xIC32xIH20xIW20xOC30xOH20xOW20xKH3xKW3"
SMALL = "yolov8_nano$dispatch_9_ime_k3_a100_conv2d_s8_N1xIC64xIH5xIW5xOC30xOH5xOW5xKH3xKW3"
K1CONV = "yolov8_nano$dispatch_4_rvv_x60_conv2d_s8_N1xIC32xIH20xIW20xOC30xOH20xOW20xKH3xKW3"
SILU = "yolov8_nano$dispatch_5_ime_k3_x100_silu_s8_n12800"
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


class Rules(unittest.TestCase):
    def test_margin_widths(self):
        self.assertEqual(cc.margin_widths({1: 100, 2: 95, 4: 70, 8: 72}, 0.10), [1, 4])
        self.assertEqual(cc.margin_widths({1: 100, 2: 89, 4: 81, 8: 50}, 0.10), [1, 2, 8])
        self.assertEqual(cc.margin_widths({1: 100, 2: 99}, 0.0), [1, 2])

    def test_row_split(self):
        self.assertTrue(cc.row_split(ROWCONV, 4, CONTRACT))       # 20 rows, 400 px
        self.assertFalse(cc.row_split(ROWCONV, 32, CONTRACT))     # fewer rows than cores
        self.assertFalse(cc.row_split(SMALL, 4, CONTRACT))        # 25 px < 36
        self.assertFalse(cc.row_split(K1CONV, 4, CONTRACT))       # not a K3 backend
        self.assertFalse(cc.row_split(SILU, 4, CONTRACT))


class Apply(unittest.TestCase):
    def setUp(self):
        self.machines, self.combos = build_machine_combinations({"CPU_P": 8}, mode="shard")

    def times(self, by_width):
        return [by_width[len(c)] for c in self.combos]

    def open_widths(self, op):
        return {len(c) for c, t in zip(self.combos, op.processing_times) if t < cc.PRICED_OUT_MS}

    def run_rules(self, specs, margin):
        ops = [_Op(f"{mod.split('$')[0]}{i}_dispatch_{did}", did, self.times(t))
               for (mod, did, t) in specs for i in range(2)]
        mods = {f"{mod.split('$')[0]}{i}_dispatch_{did}": (mod, mod.split("$")[0])
                for (mod, did, t) in specs for i in range(2)}
        st = cc.apply_to_workload(_Workload(ops, self.combos, self.machines),
                                  lambda op: mods[op.operation_name], CONTRACT, margin=margin)
        return ops, st

    def test_unshareable_op_gets_one_core_even_if_its_profile_says_otherwise(self):
        ops, st = self.run_rules([(SILU, 5, {1: 10.0, 2: 8.0, 4: 6.0, 8: 5.0})], margin=0.0)
        self.assertEqual(self.open_widths(ops[0]), {1})
        self.assertGreater(st["unshareable_options"], 0)

    def test_row_split_conv_keeps_widths_that_do_not_divide_oc(self):
        ops, _ = self.run_rules([(ROWCONV, 4, {1: 10.0, 2: 6.0, 4: 3.0, 8: 2.0})], margin=0.0)
        self.assertEqual(self.open_widths(ops[0]), {8})            # OC=30, rows: every width buildable
        ops, _ = self.run_rules([(SMALL, 9, {1: 10.0, 2: 6.0, 4: 3.0, 8: 2.0})], margin=0.0)
        self.assertEqual(self.open_widths(ops[0]), {2})            # OC=30: only 2 divides; no rows

    def test_margin_pins_a_conv_to_the_narrower_width_on_a_near_tie(self):
        t = {1: 10.0, 2: 6.0, 4: 5.6, 8: 5.5}      # 4 is 6.7% faster than 2: not enough
        ops, st = self.run_rules([(ROWCONV, 4, t)], margin=0.0)
        self.assertEqual(self.open_widths(ops[0]), {8})
        ops, st = self.run_rules([(ROWCONV, 4, t)], margin=0.10)
        self.assertEqual(self.open_widths(ops[0]), {2})
        self.assertEqual(st["margin_narrowed_dispatches"], 1)

    def test_margin_limits_runtime_sliceable_widths(self):
        ops, st = self.run_rules([(LIN, 0, {1: 10.0, 2: 9.5, 4: 7.0, 8: 6.9})], margin=0.10)
        self.assertEqual(self.open_widths(ops[0]), {1, 4})
        self.assertGreater(st["margin_excluded_options"], 0)


class RepoContract(unittest.TestCase):
    PATH = os.path.join(os.path.dirname(os.path.dirname(_HERE)), "zephyr-chipyard-sw",
                        "modelblaster", "cores", "codegen_contract.json")

    @unittest.skipUnless(os.path.exists(PATH), "ModelBlaster submodule not checked out")
    def test_repo_contract_has_the_rules(self):
        c = json.load(open(self.PATH))
        self.assertIn("conv2d_s8", cc.shareable_ops(c))
        self.assertNotIn("silu_s8", cc.shareable_ops(c))
        self.assertTrue(cc.row_split(ROWCONV, 8, c))
        self.assertFalse(cc.row_split(K1CONV, 8, c))

    def test_violations(self):
        d = {"a": {"module_name": SILU, "hardware_target": "CPU_P#0+CPU_P#1", "id": 5},
             "b": {"module_name": ROWCONV, "hardware_target": "+".join(f"CPU_P#{i}" for i in range(4)), "id": 4},
             "c": {"module_name": SMALL, "hardware_target": "+".join(f"CPU_E#{i}" for i in range(4)), "id": 9}}
        v = cc.violations(d, CONTRACT)
        self.assertTrue(any(x.startswith("a:") and "no pool path" in x for x in v))
        self.assertFalse(any(x.startswith("b:") for x in v))       # rows: OC=30 is fine
        self.assertTrue(any(x.startswith("c:") and "width must divide OC" in x for x in v))


if __name__ == "__main__":
    unittest.main()
