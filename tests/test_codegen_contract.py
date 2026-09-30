"""The compiler's constraints, as something the scheduler is held to.

The contract exists because the scheduler's option space was wider than the compiler's
and nothing connected them: `shard` mode lets every periodic instance of a dispatch pick
its own core width, which for a packed convolution cannot be generated, and the board
build discovered it at stage 1 of 5 with an error raised from inside a shell script.

These pin the properties that make the contract worth having:

1. it agrees with what ModelBlaster actually enforces -- two lists that drift are worse
   than one list, because the scheduler would believe an op is unconstrained;
2. a missing or empty contract is an ERROR, not an empty ruleset that passes everything;
3. the op is recovered from `module_name`, since a schedule entry has no op field, and
   the backend tag contains underscores so no positional split works;
4. the real schedules that broke the build are still reported, and the ones that built
   are still accepted.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_REPO, "xpu-rt"))

import codegen_contract as cc  # noqa: E402


def _entry(net, inst, did, op, width, oc=None, impl="rvv", kind="CPU_P"):
    shape = f"N1xIC32xIH27xIW27xOC{oc}xOH14xOW14" if oc else "N1xK256xN1024"
    return {
        "id": did, "job_name": f"{net}{inst}",
        "hardware_target": "+".join(f"{kind}#{i}" for i in range(width)),
        "module_name": f"{net}$dispatch_{did}_rvv_x60_{op}_{shape}",
        "impl": impl, "duration": 1.0, "start_time": 0.0,
    }


def _sched(entries):
    return {"dispatches": {f"{e['job_name']}_dispatch_{e['id']}": e for e in entries}}


class TheContractIsLoadable(unittest.TestCase):

    def test_it_ships_with_the_submodule(self):
        c = cc.load()
        self.assertEqual(c["schema"], "modelblaster_codegen_contract/v1")
        self.assertIn("uniform_width_across_instances", c["rules"])

    def test_a_missing_contract_is_an_error_not_an_empty_ruleset(self):
        with self.assertRaises(cc.ContractUnavailable):
            cc.load("/nonexistent/codegen_contract.json")

    def test_an_empty_ruleset_is_an_error(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump({"schema": "x", "rules": {}}, f)
            p = f.name
        try:
            with self.assertRaises(cc.ContractUnavailable):
                cc.load(p)
        finally:
            os.unlink(p)

    def test_it_matches_what_modelblaster_enforces(self):
        """The whole point: one definition, two readers."""
        sys.path.insert(0, os.path.join(_REPO, "ModelBlaster"))
        from pipeline.schedule_shards import _PACKED_WEIGHT_SHARD_OPS
        self.assertEqual(set(cc.packed_weight_ops()), set(_PACKED_WEIGHT_SHARD_OPS))


class TheOpComesFromTheModuleName(unittest.TestCase):
    """A schedule entry has no `op` field, and the backend tag has underscores."""

    def test_it_does_not_confuse_nested_op_names(self):
        ops = cc.packed_weight_ops()
        self.assertEqual(
            cc.op_of("dronet$dispatch_3_rvv_x60_conv2d_batchnorm2d_s8_N1xIC32", ops),
            "conv2d_batchnorm2d_s8")
        self.assertEqual(
            cc.op_of("dronet$dispatch_0_rvv_x60_conv2d_s8_N1xIC3", ops), "conv2d_s8")

    def test_an_unknown_op_is_empty_not_a_guess(self):
        self.assertEqual(cc.op_of("x$dispatch_0_rvv_x60_gelu_s8_N1x256",
                                  cc.packed_weight_ops()), "")


class UniformWidthIsRequiredOnlyWhereWeightsArePacked(unittest.TestCase):

    def test_a_conv_varying_width_across_instances_is_refused(self):
        vs = cc.violations(_sched([
            _entry("dronet", 0, 0, "conv2d_s8", 2, oc=32),
            _entry("dronet", 1, 0, "conv2d_s8", 4, oc=32),
        ]))
        self.assertEqual([v["rule"] for v in vs], ["uniform_width_across_instances"])
        self.assertEqual(vs[0]["severity"], "refuse")
        self.assertEqual((vs[0]["network"], vs[0]["dispatch_id"]), ("dronet", 0))

    def test_a_linear_varying_width_across_instances_is_fine(self):
        """Row-major weights are sliced at runtime from the entry's own pool width."""
        vs = cc.violations(_sched([
            _entry("ffn_block", 0, 1, "linear_s8", 1),
            _entry("ffn_block", 1, 1, "linear_s8", 4),
        ]))
        self.assertEqual(vs, [])

    def test_a_conv_at_one_width_everywhere_is_fine(self):
        vs = cc.violations(_sched([
            _entry("dronet", 0, 0, "conv2d_s8", 4, oc=32),
            _entry("dronet", 1, 0, "conv2d_s8", 4, oc=32),
        ]))
        self.assertEqual(vs, [])


class OtherRulesTheScheduleCanShow(unittest.TestCase):

    def test_a_width_that_does_not_divide_output_channels_is_refused(self):
        vs = cc.violations(_sched([_entry("dronet", 0, 0, "conv2d_s8", 4, oc=30)]))
        self.assertEqual([v["rule"] for v in vs], ["width_divides_output_channels"])
        self.assertIn("OC=30", vs[0]["detail"])

    def test_a_combination_mixing_core_kinds_is_refused(self):
        e = _entry("dronet", 0, 0, "conv2d_s8", 2, oc=32)
        e["hardware_target"] = "CPU_P#0+CPU_E#0"
        vs = cc.violations(_sched([e]))
        self.assertIn("machine_combinations", [v["rule"] for v in vs])

    def test_an_ime_dispatch_on_cluster_one_is_refused(self):
        e = _entry("ffn_block", 0, 1, "linear_s8", 1, impl="ime", kind="CPU_E")
        vs = cc.violations(_sched([e]))
        self.assertIn("ime", [v["rule"] for v in vs])
        self.assertIn("TRAPS", [v["detail"] for v in vs if v["rule"] == "ime"][0])

    def test_refusals_are_reported_before_warnings(self):
        e = _entry("dronet", 0, 0, "conv2d_s8", 4, oc=30)
        vs = cc.violations(_sched([e]))
        sev = [v["severity"] for v in vs]
        self.assertEqual(sev, sorted(sev, key=lambda s: {"refuse": 0}.get(s, 1)))


class TheRealSchedulesThatBrokeTheBuild(unittest.TestCase):
    """Regression against the artifacts, not just synthetic entries.

    Skipped when the schedules are not in the checkout rather than silently passing.
    """

    def _check(self, stem):
        p = os.path.join(_REPO, "schedules", f"scheduled_{stem}_greedy_profiled.json")
        if not os.path.exists(p):
            self.skipTest(f"{stem} is not in this checkout")
        return cc.violations(p)

    def test_the_w5_shard_schedule_is_still_refused(self):
        """The schedule that broke the board build, from a FROZEN excerpt.

        It does not read `schedules/scheduled_w5_..._greedy_profiled.json`, which is a
        build OUTPUT: since `codegen_contract.pin_uniform_widths` makes the greedy shard
        schedule uniform-width, a rerun overwrites that path with a clean schedule, and
        a test reading it would fail without the refusal having regressed. A regression
        test cannot depend on a file the pipeline rewrites. The excerpt keeps only the
        four dronet dispatches that violate the rule, so it is 10 KB and immune to reruns.
        """
        p = os.path.join(_REPO, "tests", "fixtures",
                         "w5_shard_uniform_width_violation.json")
        vs = cc.violations(json.load(open(p)))
        refuse = [v for v in vs if v["severity"] == "refuse"]
        self.assertTrue(refuse, "this schedule is what the board build died on")
        self.assertTrue(all(v["network"] == "dronet" for v in refuse))
        self.assertEqual({v["rule"] for v in refuse},
                         {"uniform_width_across_instances"})

    def test_pinning_makes_that_same_shard_schedule_buildable(self):
        """The other half of the pair: pinning, asserted on the real w5 shard spec.

        Without pinning, greedy gives dronet's conv dispatches different widths across
        instances and the candidate is thrown away -- which makes w4 and w5 read as
        workloads where no lever helps. With it, the same solve is
        contract-clean. Skipped rather than silently passing when the live schedule is
        absent, since it is a build output.
        """
        vs = self._check("w5_ffn_dronet_yolo_r2_shard")
        if vs is None:
            self.skipTest("no live w5 shard schedule in this checkout")
        # A schedule produced with the pin on must carry no width refusal at all.
        width = [v for v in vs if v["rule"] == "uniform_width_across_instances"]
        self.assertEqual(
            width, [],
            "the live w5 shard schedule still mixes widths across instances; "
            "was it solved without XPURT_UNIFORM_PACKED_WIDTH=1?")

    def test_the_w5_ime_schedule_is_accepted(self):
        self.assertEqual(self._check("w5_ffn_dronet_yolo_r1_ime"), [])


if __name__ == "__main__":
    unittest.main()
