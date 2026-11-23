"""K3 shard rules in the codegen contract (k3-work TASK-016).

* shareable_ops (hard): an op without a ModelBlaster pool path gets no
  multi-core option, whatever its profile says.
* conv_split_axis: a K3 conv2d_s8 with enough output rows is split by rows,
  so a width that does not divide OC stays open.
* width margin (soft): a wider width is kept only if >= margin faster than the
  next narrower width kept; a conv's one width is the widest kept.
* The repo's own contract (zephyr-chipyard-sw/modelblaster/cores/...) carries
  the same rules, and schedules checked by `violations` flag both breaches.
* TASK-017: the simple layers (silu, add, maxpool, ...) are shareable on the K3
  backends only (shareable_ops.backend_prefixes), one width per dispatch
  (uniform_width_across_instances), chosen by the same margin.
* K1 as on main: with shareable_ops.applies_to_backend_prefixes and
  uniform_width_across_instances.backend_prefixes, a K1 dispatch gets exactly
  the options main's rules (the convolution rules only) give it, and a K3
  dispatch the same options as without the scope.
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
K1SILU = "yolov8_nano$dispatch_5_rvv_x60_silu_s8_n12800"
ELU = "mlp_control$dispatch_1_ime_k3_a100_elu_s8_n256"
# TASK-017 rules: simple layers shareable on K3 backends, one width per dispatch
CONTRACT17 = {"rules": {**CONTRACT["rules"],
    "uniform_width_across_instances": {"applies_to_ops": ["conv2d_s8", "silu_s8", "elu_s8"]},
    "shareable_ops": {"ops": ["conv2d_s8", "elu_s8", "linear_s8", "silu_s8"],
                      "backend_prefixes": {"elu_s8": ["rvv_k3", "ime_k3"], "silu_s8": ["rvv_k3", "ime_k3"]}},
}}
K3 = ["rvv_k3", "ime_k3"]
# Main's rules as XPU-RT reads them: the convolution rules only
MAIN_CONVS = ["conv2d_s8", "conv2d_batchnorm2d_s8", "conv2d_batchnorm2d_silu_s8", "conv2d_silu_s8"]
MAIN = {"rules": {"uniform_width_across_instances": {"applies_to_ops": MAIN_CONVS},
                  "width_divides_output_channels": {"applies_to_ops": MAIN_CONVS}}}
# The K3 rules on top of main's, without and with the K3 scope (ModelBlaster's contract now)
UNSCOPED = {"rules": {**CONTRACT17["rules"],
    "uniform_width_across_instances": {"applies_to_ops": MAIN_CONVS + ["silu_s8", "elu_s8"]},
    "width_divides_output_channels": {"applies_to_ops": MAIN_CONVS},
}}
SCOPED = {"rules": {**UNSCOPED["rules"],
    "uniform_width_across_instances": {"applies_to_ops": MAIN_CONVS + ["silu_s8", "elu_s8"],
                                       "backend_prefixes": {"elu_s8": K3, "silu_s8": K3}},
    "shareable_ops": {**CONTRACT17["rules"]["shareable_ops"], "applies_to_backend_prefixes": K3},
}}
K1AVG = "dronet$dispatch_7_rvv_x60_avgpool2d_s8_n64"              # in no list
K1LIN = "mlp_control$dispatch_0_ime_x60_linear_s8_M1xK16xN256"
K1FUSED = "dronet$dispatch_2_rvv_x60_conv2d_batchnorm2d_s8_N1xIC8xIH16xIW16xOC6xOH8xOW8xKH3xKW3"
K3SOFT = "mlp_control$dispatch_3_rvv_k3_a100_softmax_s8_n16"       # K3, in no list
K1_SPECS = [(K1SILU, 5, {1: 10.0, 2: 5.0, 4: 3.0, 8: 2.0}), (K1CONV, 4, {1: 10.0, 2: 6.0, 4: 3.0, 8: 2.0}),
            (K1AVG, 7, {1: 4.0, 2: 2.5, 4: 2.0, 8: 1.9}), (K1LIN, 0, {1: 10.0, 2: 9.5, 4: 7.0, 8: 6.9}),
            (K1FUSED, 2, {1: 8.0, 2: 4.0, 4: 3.0, 8: 2.0})]
K3_SPECS = [(SILU, 5, {1: 100.0, 2: 55.0, 4: 30.0, 8: 28.0}), (ELU, 1, {1: 0.005, 2: 0.020, 4: 0.021, 8: 0.025}),
            (ROWCONV, 4, {1: 10.0, 2: 6.0, 4: 5.6, 8: 5.5}), (SMALL, 9, {1: 10.0, 2: 6.0, 4: 3.0, 8: 2.0}),
            (LIN, 0, {1: 10.0, 2: 9.5, 4: 7.0, 8: 6.9}), (K3SOFT, 3, {1: 3.0, 2: 1.0, 4: 0.8, 8: 0.7})]


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

    def run_rules(self, specs, margin, contract=CONTRACT):
        ops = [_Op(f"{mod.split('$')[0]}{i}_dispatch_{did}", did, self.times(t))
               for (mod, did, t) in specs for i in range(2)]
        mods = {f"{mod.split('$')[0]}{i}_dispatch_{did}": (mod, mod.split("$")[0])
                for (mod, did, t) in specs for i in range(2)}
        st = cc.apply_to_workload(_Workload(ops, self.combos, self.machines),
                                  lambda op: mods[op.operation_name], contract, margin=margin)
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


    def test_k3_simple_layer_gets_one_width_by_the_margin(self):
        t = {1: 100.0, 2: 55.0, 4: 30.0, 8: 28.0}     # 8 is 6.7% faster than 4: not enough
        ops, st = self.run_rules([(SILU, 5, t)], margin=0.10, contract=CONTRACT17)
        self.assertEqual(st["unshareable_options"], 0)
        for op in ops:                                  # both instances pinned to 4
            self.assertEqual(self.open_widths(op), {4})

    def test_k3_simple_layer_stays_on_one_core_when_wider_is_not_10pct_faster(self):
        ops, _ = self.run_rules([(ELU, 1, {1: 0.005, 2: 0.020, 4: 0.021, 8: 0.025})], margin=0.10,
                                contract=CONTRACT17)
        self.assertEqual(self.open_widths(ops[0]), {1})

    def test_simple_layer_on_k1_backend_gets_one_core(self):
        ops, st = self.run_rules([(K1SILU, 5, {1: 10.0, 2: 5.0, 4: 3.0, 8: 2.0})], margin=0.10,
                                 contract=CONTRACT17)
        self.assertEqual(self.open_widths(ops[0]), {1})
        self.assertGreater(st["unshareable_options"], 0)

    def test_k1_dispatches_get_what_mains_rules_give(self):
        for margin in (0.0,):          # main has no width margin
            main_ops, main_st = self.run_rules(K1_SPECS, margin, contract=MAIN)
            ops, st = self.run_rules(K1_SPECS, margin, contract=SCOPED)
            self.assertEqual([o.processing_times for o in ops], [o.processing_times for o in main_ops])
            self.assertEqual(st["unshareable_options"], 0)
            for k in ("oc_excluded_options", "pinned_dispatches", "unpinnable"):
                self.assertEqual(st[k], main_st[k], k)
        # before the scope, the K1 silu and avgpool lost their multi-core options
        ops, _ = self.run_rules(K1_SPECS, 0.0, contract=UNSCOPED)
        self.assertEqual(self.open_widths(ops[0]), {1})
        self.assertEqual(self.open_widths(ops[4]), {1})

    def test_k3_dispatches_unchanged_by_the_scope(self):
        for margin in (0.0, 0.10):
            before, st0 = self.run_rules(K3_SPECS, margin, contract=UNSCOPED)
            after, st1 = self.run_rules(K3_SPECS, margin, contract=SCOPED)
            self.assertEqual([o.processing_times for o in after], [o.processing_times for o in before])
            self.assertEqual(st1, st0)
            self.assertGreater(st1["unshareable_options"], 0)        # the K3 softmax still on one core

    @unittest.skipUnless(os.path.exists(os.path.join(os.path.dirname(os.path.dirname(_HERE)),
                         "zephyr-chipyard-sw", "modelblaster", "cores", "codegen_contract.json")),
                         "ModelBlaster submodule not checked out")
    def test_repo_contract_gives_k1_mains_options(self):
        c = json.load(open(RepoContract.PATH))
        main_ops, _ = self.run_rules(K1_SPECS, 0.0, contract=MAIN)
        ops, st = self.run_rules(K1_SPECS, 0.0, contract=c)
        self.assertEqual([o.processing_times for o in ops], [o.processing_times for o in main_ops])
        self.assertEqual(st["unshareable_options"], 0)

    def test_width_pinned(self):
        self.assertTrue(cc.width_pinned(SILU, "silu_s8", SCOPED))
        self.assertFalse(cc.width_pinned(K1SILU, "silu_s8", SCOPED))
        self.assertTrue(cc.width_pinned(K1CONV, "conv2d_s8", SCOPED))     # no prefix list: everywhere
        self.assertTrue(cc.width_pinned(K1SILU, "silu_s8", UNSCOPED))     # unscoped contract
        self.assertFalse(cc.width_pinned(LIN, "linear_s8", SCOPED))

    def test_op_shareable_scoped(self):
        self.assertTrue(cc.op_shareable(K1SILU, "silu_s8", SCOPED))
        self.assertTrue(cc.op_shareable(K1AVG, "", SCOPED))
        self.assertTrue(cc.op_shareable(SILU, "silu_s8", SCOPED))
        self.assertFalse(cc.op_shareable(K3SOFT, "", SCOPED))

    def test_op_shareable(self):
        self.assertTrue(cc.op_shareable(SILU, "silu_s8", CONTRACT17))
        self.assertFalse(cc.op_shareable(K1SILU, "silu_s8", CONTRACT17))
        self.assertTrue(cc.op_shareable(K1CONV, "conv2d_s8", CONTRACT17))   # no prefix list: everywhere
        self.assertFalse(cc.op_shareable(SILU, "silu_s8", CONTRACT))        # TASK-016 list
        self.assertTrue(cc.op_shareable(SILU, "silu_s8", {"rules": {}}))     # no list at all


class RepoContract(unittest.TestCase):
    PATH = os.path.join(os.path.dirname(os.path.dirname(_HERE)), "zephyr-chipyard-sw",
                        "modelblaster", "cores", "codegen_contract.json")

    @unittest.skipUnless(os.path.exists(PATH), "ModelBlaster submodule not checked out")
    def test_repo_contract_has_the_rules(self):
        c = json.load(open(self.PATH))
        self.assertIn("conv2d_s8", cc.shareable_ops(c))
        self.assertNotIn("gelu_s8", cc.shareable_ops(c))
        # TASK-017: simple layers shareable on K3 backends only, one width per dispatch
        self.assertTrue(cc.op_shareable(SILU, "silu_s8", c))
        self.assertIn("silu_s8", cc.uniform_width_ops(c))
        self.assertTrue(cc.width_pinned(SILU, "silu_s8", c))
        # the K1 keeps main's options: every op shareable, only convolutions pinned
        self.assertTrue(cc.op_shareable(K1SILU, "silu_s8", c))
        self.assertTrue(cc.op_shareable(K1AVG, "", c))
        self.assertFalse(cc.width_pinned(K1SILU, "silu_s8", c))
        self.assertTrue(cc.width_pinned(K1CONV, "conv2d_s8", c))
        self.assertTrue(cc.row_split(ROWCONV, 8, c))
        self.assertFalse(cc.row_split(K1CONV, 8, c))

    def test_violations(self):
        d = {"a": {"module_name": SILU, "hardware_target": "CPU_P#0+CPU_P#1", "id": 5},
             "b": {"module_name": ROWCONV, "hardware_target": "+".join(f"CPU_P#{i}" for i in range(4)), "id": 4},
             "c": {"module_name": SMALL, "hardware_target": "+".join(f"CPU_E#{i}" for i in range(4)), "id": 9}}
        v = cc.violations(d, CONTRACT)
        self.assertTrue(any(x.startswith("a:") and "no pool path" in x for x in v))
        v17 = cc.violations(dict(d, k={"module_name": K1SILU, "hardware_target": "CPU_P#0+CPU_P#1", "id": 5}),
                            CONTRACT17)
        self.assertFalse(any(x.startswith("a:") for x in v17))   # K3 silu on 2 cores: fine now
        self.assertTrue(any(x.startswith("k:") and "no pool path" in x for x in v17))
        vs = cc.violations(dict(d, k={"module_name": K1SILU, "hardware_target": "CPU_P#0+CPU_P#1", "id": 5},
                                k2={"module_name": K1SILU, "hardware_target": "CPU_P#2", "id": 5},
                                a2={"module_name": SILU, "hardware_target": "CPU_P#4", "id": 5}), SCOPED)
        self.assertFalse(any(x.startswith("k") for x in vs))       # K1 silu: any width, per instance
        self.assertTrue(any("yolov8_nano dispatch 5" in x and "[1, 2]" in x for x in vs))  # K3 silu pinned
        self.assertEqual(cc.violations({"k": {"module_name": K1SILU, "hardware_target": "CPU_P#0+CPU_P#1",
                                              "id": 5}}, SCOPED),
                         cc.violations({"k": {"module_name": K1SILU, "hardware_target": "CPU_P#0+CPU_P#1",
                                              "id": 5}}, MAIN))
        self.assertFalse(any(x.startswith("b:") for x in v))       # rows: OC=30 is fine
        self.assertTrue(any(x.startswith("c:") and "width must divide OC" in x for x in v))


if __name__ == "__main__":
    unittest.main()
