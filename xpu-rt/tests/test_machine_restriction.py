"""A network may be pinned to a set of harts, and the pin must survive shard mode.

WHY THIS IS NOT `infeasible_machines`. `parse_infeasible_combinations` mapped a
dispatch's excluded machines into indices of the **machines** list. Under
`machine_combination_mode: "shard"` a combination is an aligned BLOCK of harts,
not a hart, so the two index spaces disagree and the one existing exclusion
mechanism could not express a placement restriction on a sharded workload
(recorded in docs/Feature/board_and_model_gaps.md). These tests pin the combination-space
reading, and the per-network `allowed_machines` restriction built on it.

WHY THE COST IS WRITTEN TOO. The MILP and CP-SAT paths read
`Operation.infeasible_combinations`; the greedy list scheduler does not read it
at all -- it takes the combination with the earliest completion. A set-only
exclusion would therefore be honoured by two solvers out of three, silently.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from workload_factory import (  # noqa: E402
    PINNED_OUT_COST_MS,
    build_machine_combinations,
    create_workload_from_network_hierarchy,
    impl_restriction_to_infeasible_combinations,
    machine_restriction_to_infeasible_combinations,
    parse_infeasible_combinations,
)

MACHINES, SHARD_COMBOS = build_machine_combinations(
    {"CPU_P": 4, "CPU_E": 4}, mode="shard")


def _kept(excluded):
    return [SHARD_COMBOS[k] for k in range(len(SHARD_COMBOS)) if k not in excluded]


class RestrictionIsInCombinationSpace(unittest.TestCase):
    def test_kind_selector_keeps_every_block_of_that_kind(self):
        exc = machine_restriction_to_infeasible_combinations(
            ["CPU_P"], MACHINES, SHARD_COMBOS)
        kept = _kept(exc)
        # the four P singletons, the two aligned pairs, and the quad
        self.assertEqual(len(kept), 7)
        self.assertTrue(all(m.startswith("CPU_P") for c in kept for m in c))
        self.assertIn(["CPU_P#0", "CPU_P#1", "CPU_P#2", "CPU_P#3"], kept)

    def test_block_straddling_the_boundary_is_excluded_whole(self):
        """The property machine-index space cannot express: a block is excluded
        when ANY of its harts is outside the allowed set, because the block
        occupies all of them for its whole duration."""
        exc = machine_restriction_to_infeasible_combinations(
            ["CPU_P#0", "CPU_P#1"], MACHINES, SHARD_COMBOS)
        kept = _kept(exc)
        self.assertEqual(kept, [["CPU_P#0"], ["CPU_P#1"], ["CPU_P#0", "CPU_P#1"]])

    def test_single_core_selector(self):
        exc = machine_restriction_to_infeasible_combinations(
            ["CPU_E#0"], MACHINES, SHARD_COMBOS)
        self.assertEqual(_kept(exc), [["CPU_E#0"]])

    def test_case_insensitive(self):
        self.assertEqual(
            machine_restriction_to_infeasible_combinations(
                ["cpu_p"], MACHINES, SHARD_COMBOS),
            machine_restriction_to_infeasible_combinations(
                ["CPU_P"], MACHINES, SHARD_COMBOS))

    def test_empty_restriction_is_no_restriction(self):
        for empty in (None, [], ()):
            self.assertEqual(
                machine_restriction_to_infeasible_combinations(
                    empty, MACHINES, SHARD_COMBOS), set())

    def test_unknown_selector_raises(self):
        with self.assertRaises(ValueError) as cm:
            machine_restriction_to_infeasible_combinations(
                ["CPU_X"], MACHINES, SHARD_COMBOS, context="network 'n'")
        self.assertIn("names no machine", str(cm.exception))
        self.assertIn("network 'n'", str(cm.exception))

    def test_restriction_that_leaves_nothing_raises(self):
        combos = [["CPU_P#0", "CPU_P#1"]]
        with self.assertRaises(ValueError) as cm:
            machine_restriction_to_infeasible_combinations(
                ["CPU_P#0"], ["CPU_P#0", "CPU_P#1"], combos)
        self.assertIn("leaves no runnable machine combination", str(cm.exception))


class WidthRestriction(unittest.TestCase):
    """How many harts a dispatch may occupy, per network. Orthogonal to WHICH
    harts: pinning yolo to cluster 0 says nothing about whether a frame runs on
    one hart or spreads over four."""

    def test_width_alone(self):
        exc = machine_restriction_to_infeasible_combinations(
            None, MACHINES, SHARD_COMBOS, widths=4)
        self.assertEqual(_kept(exc), [["CPU_P#0", "CPU_P#1", "CPU_P#2", "CPU_P#3"],
                                      ["CPU_E#0", "CPU_E#1", "CPU_E#2", "CPU_E#3"]])

    def test_width_and_machines_compose(self):
        exc = machine_restriction_to_infeasible_combinations(
            ["CPU_P"], MACHINES, SHARD_COMBOS, widths=4)
        self.assertEqual(_kept(exc), [["CPU_P#0", "CPU_P#1", "CPU_P#2", "CPU_P#3"]])

    def test_width_list(self):
        exc = machine_restriction_to_infeasible_combinations(
            ["CPU_P"], MACHINES, SHARD_COMBOS, widths=[1, 2])
        self.assertEqual(sorted(len(c) for c in _kept(exc)), [1, 1, 1, 1, 2, 2])

    def test_width_one_is_the_per_network_form_of_no_sharding(self):
        exc = machine_restriction_to_infeasible_combinations(
            None, MACHINES, SHARD_COMBOS, widths=1)
        self.assertTrue(all(len(c) == 1 for c in _kept(exc)))
        self.assertEqual(len(_kept(exc)), len(MACHINES))

    def test_unavailable_width_raises(self):
        with self.assertRaises(ValueError) as cm:
            machine_restriction_to_infeasible_combinations(
                ["CPU_P"], MACHINES, SHARD_COMBOS, widths=3)
        self.assertIn("matches no machine combination", str(cm.exception))

    def test_width_that_excludes_the_allowed_machines_raises(self):
        with self.assertRaises(ValueError) as cm:
            machine_restriction_to_infeasible_combinations(
                ["CPU_E#0"], MACHINES, SHARD_COMBOS, widths=4)
        self.assertIn("leaves no runnable machine combination", str(cm.exception))

    def test_bad_width_value_raises(self):
        for bad in (0, -1, "4", [], [1, "2"], True):
            with self.assertRaises(ValueError):
                machine_restriction_to_infeasible_combinations(
                    ["CPU_P"], MACHINES, SHARD_COMBOS, widths=bad)


IMPL_MACHINES, IMPL_COMBOS = ["CPU_P#0", "CPU_P#1", "CPU_E#0", "CPU_E#1"], [
    ["CPU_P#0"], ["CPU_P#1"], ["CPU_P#0", "CPU_P#1"],          # rvv, cluster 0
    ["CPU_P#0"], ["CPU_P#1"], ["CPU_P#0", "CPU_P#1"],          # ime, cluster 0 only
    ["CPU_E#0"], ["CPU_E#1"], ["CPU_E#0", "CPU_E#1"],          # rvv, cluster 1
]
IMPL_OF = ["rvv"] * 3 + ["ime"] * 3 + ["rvv"] * 3


class ImplRestriction(unittest.TestCase):
    """Which implementation a network is placed on. The third question of the
    same shape as which harts and how many -- and the one a per-dispatch cost
    model cannot answer, because a solver handed both cells always takes the
    cheaper one."""

    def test_keeps_only_the_named_impl(self):
        exc = impl_restriction_to_infeasible_combinations(["ime"], IMPL_OF)
        self.assertEqual(exc, {0, 1, 2, 6, 7, 8})

    def test_case_insensitive_and_multiple(self):
        self.assertEqual(
            impl_restriction_to_infeasible_combinations(["IME", "rvv"], IMPL_OF), set())

    def test_absent_is_no_restriction(self):
        for empty in (None, [], ()):
            self.assertEqual(
                impl_restriction_to_infeasible_combinations(empty, IMPL_OF), set())
        # and with no impl axis at all, an absent restriction is still fine
        self.assertEqual(impl_restriction_to_infeasible_combinations(None, None), set())

    def test_without_the_impl_axis_it_raises(self):
        with self.assertRaises(ValueError) as cm:
            impl_restriction_to_infeasible_combinations(["ime"], None, context="network 'n'")
        self.assertIn("enable_impls", str(cm.exception))

    def test_unknown_impl_raises(self):
        with self.assertRaises(ValueError) as cm:
            impl_restriction_to_infeasible_combinations(["npu"], IMPL_OF)
        self.assertIn("matches no combination", str(cm.exception))


class InfeasibleMachinesInCombinationSpace(unittest.TestCase):
    def test_excludes_every_block_touching_the_named_hart(self):
        exc = parse_infeasible_combinations(
            {"infeasible_machines": ["CPU_E#0"]}, MACHINES, SHARD_COMBOS)
        self.assertEqual(
            sorted(SHARD_COMBOS[k] for k in exc),
            [["CPU_E#0"], ["CPU_E#0", "CPU_E#1"],
             ["CPU_E#0", "CPU_E#1", "CPU_E#2", "CPU_E#3"]])
        # machine-index space would have answered {4}, which under shard mode is
        # the P-core pair ['CPU_P#0','CPU_P#1'] -- a different placement entirely.
        self.assertNotIn(4, exc)

    def test_singleton_combinations_are_the_legacy_reading(self):
        machines = ["CPU_P", "CPU_E"]
        self.assertEqual(
            parse_infeasible_combinations({"infeasible_machines": ["CPU_E"]}, machines),
            {1})

    def test_absent_field_is_empty(self):
        self.assertEqual(parse_infeasible_combinations({}, MACHINES, SHARD_COMBOS), set())


def _write_graph(tmpdir, name, dispatches):
    d = os.path.join(tmpdir, name, f"{name}.int8")
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, f"{name}.int8_dispatch_graph.json")
    json.dump({"dispatches": dispatches}, open(p, "w"))
    return p


class AllowedMachinesReachesTheOperations(unittest.TestCase):
    """The spec field has to arrive on every Operation of every INSTANCE, and as
    both an exclusion and a cost."""

    def _build(self, tmp, allowed, width=None):
        g = _write_graph(tmp, "pinned", {"d0": {"id": 0, "dependencies": []}})
        g2 = _write_graph(tmp, "free", {"d0": {"id": 0, "dependencies": []}})
        nd = {"networks": {
            "pinned": {"id": 0, "identifier": "pinned", "dispatch_deps_path": g,
                       "period": 10.0, "window_duration": 10.0, "num_instances": 3,
                       "allowed_machines": allowed,
                       **({"machine_width": width} if width is not None else {})},
            "free": {"id": 1, "identifier": "free", "dispatch_deps_path": g2,
                     "period": 10.0, "window_duration": 10.0, "num_instances": 3},
        }, "edges": [], "horizon_ms": 30.0}
        return create_workload_from_network_hierarchy(
            nd, repo_base_path=tmp, machines=MACHINES,
            transfer_times=np.zeros((len(MACHINES), len(MACHINES))),
            processing_times=None, random_seed=0,
            machine_combinations=SHARD_COMBOS)

    def test_every_instance_is_pinned_and_priced_out(self):
        with tempfile.TemporaryDirectory() as tmp:
            wl = self._build(tmp, ["CPU_P"])
            pinned = [op for op in wl.operations if op.operation_name.startswith("pinned")]
            free = [op for op in wl.operations if op.operation_name.startswith("free")]
            self.assertEqual(len(pinned), 3)
            self.assertEqual(len(free), 3)
            for op in pinned:
                for k, combo in enumerate(SHARD_COMBOS):
                    on_p = all(m.startswith("CPU_P") for m in combo)
                    self.assertEqual(k not in op.infeasible_combinations, on_p,
                                     f"{op.operation_name} combo {combo}")
                    # the greedy list scheduler reads the COST, not the set
                    self.assertEqual(op.processing_times[k] >= PINNED_OUT_COST_MS,
                                     not on_p, f"{op.operation_name} combo {combo}")
            for op in free:
                self.assertFalse(op.infeasible_combinations)
                self.assertTrue(all(t < PINNED_OUT_COST_MS for t in op.processing_times))

    def test_pricing_one_network_out_does_not_touch_another(self):
        """Periodic instances share a cached processing-times list; mutating it
        in place would leak across instances and across networks."""
        with tempfile.TemporaryDirectory() as tmp:
            wl = self._build(tmp, ["CPU_E#3"])
            free = [op for op in wl.operations if op.operation_name.startswith("free")]
            for op in free:
                self.assertTrue(all(t < PINNED_OUT_COST_MS for t in op.processing_times))

    def test_machine_width_reaches_the_operations(self):
        with tempfile.TemporaryDirectory() as tmp:
            wl = self._build(tmp, ["CPU_P"], width=4)
            pinned = [op for op in wl.operations if op.operation_name.startswith("pinned")]
            quad = SHARD_COMBOS.index(["CPU_P#0", "CPU_P#1", "CPU_P#2", "CPU_P#3"])
            for op in pinned:
                self.assertEqual(
                    {k for k in range(len(SHARD_COMBOS))} - op.infeasible_combinations,
                    {quad})
                self.assertTrue(op.processing_times[quad] < PINNED_OUT_COST_MS)
                self.assertTrue(op.processing_times[0] >= PINNED_OUT_COST_MS)

    def test_prefer_impls_pins_what_it_can_and_falls_back_per_dispatch(self):
        """`impl` is per dispatch: an op with no kernel for the preferred
        implementation has no cell there at any cost (profile_loader files those
        at 1e8), and excluding its other combinations too would make it
        unplaceable. It keeps them; the ones that DO have a cell are pinned."""
        with tempfile.TemporaryDirectory() as tmp:
            g = _write_graph(tmp, "net", {"has_ime": {"id": 0, "dependencies": []},
                                          "no_ime": {"id": 1, "dependencies": ["has_ime"]}})
            # combination 0/1/2 rvv, 3/4/5 ime (same machines, as enable_impls emits)
            times_ok = [5.0, 5.0, 3.0, 2.0, 2.0, 1.0]
            times_no = [5.0, 5.0, 3.0, 1e8, 1e8, 1e8]     # no ime kernel for this op
            # non-periodic, so the dispatch keeps its own name and its own costs
            nd = {"networks": {"net": {
                "id": 0, "identifier": "net", "dispatch_deps_path": g,
                "prefer_impls": ["ime"]}}, "edges": []}
            wl = create_workload_from_network_hierarchy(
                nd, repo_base_path=tmp, machines=["CPU_P#0", "CPU_P#1"],
                transfer_times=np.zeros((2, 2)),
                processing_times={"net_has_ime": times_ok, "net_no_ime": times_no},
                random_seed=0,
                machine_combinations=[["CPU_P#0"], ["CPU_P#1"], ["CPU_P#0", "CPU_P#1"]] * 2,
                combo_impls=["rvv", "rvv", "rvv", "ime", "ime", "ime"])
            ops = {op.operation_name: op for op in wl.operations}
            # the op with an ime cell is held there
            self.assertEqual(ops["net_has_ime"].infeasible_combinations, {0, 1, 2})
            self.assertTrue(all(ops["net_has_ime"].processing_times[k] >= PINNED_OUT_COST_MS
                                for k in (0, 1, 2)))
            # the op without one keeps every combination it had
            self.assertEqual(ops["net_no_ime"].infeasible_combinations, set())
            self.assertEqual(ops["net_no_ime"].processing_times[:3], [5.0, 5.0, 3.0])

    def test_prefer_impls_without_the_impl_axis_stops_the_build(self):
        with tempfile.TemporaryDirectory() as tmp:
            g = _write_graph(tmp, "net", {"d0": {"id": 0, "dependencies": []}})
            nd = {"networks": {"net": {
                "id": 0, "identifier": "net", "dispatch_deps_path": g,
                "period": 50.0, "window_duration": 50.0, "num_instances": 1,
                "prefer_impls": ["ime"]}}, "edges": [], "horizon_ms": 50.0}
            with self.assertRaises(ValueError):
                create_workload_from_network_hierarchy(
                    nd, repo_base_path=tmp, machines=MACHINES,
                    transfer_times=np.zeros((len(MACHINES), len(MACHINES))),
                    processing_times=None, random_seed=0,
                    machine_combinations=SHARD_COMBOS)

    def test_bad_selector_stops_the_build(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                self._build(tmp, ["CPU_G"])


class ShardOnlyNetworksNone(unittest.TestCase):
    """"hold every network at one core" has to be sayable. An empty list means
    "no restriction", so before the `none` sentinel the only way to express it
    was to name a network that could not widen anyway."""

    def setUp(self):
        import codegen_contract
        self.cc = codegen_contract
        self._saved = os.environ.get("XPURT_SHARD_ONLY_NETS")

    def tearDown(self):
        if self._saved is None:
            os.environ.pop("XPURT_SHARD_ONLY_NETS", None)
        else:
            os.environ["XPURT_SHARD_ONLY_NETS"] = self._saved

    def test_unset_is_none(self):
        os.environ.pop("XPURT_SHARD_ONLY_NETS", None)
        self.assertIsNone(self.cc.shard_only_networks_from_env())

    def test_literal_none_is_the_empty_set(self):
        for spelling in ("none", "NONE", " None "):
            os.environ["XPURT_SHARD_ONLY_NETS"] = spelling
            got = self.cc.shard_only_networks_from_env()
            self.assertEqual(got, set(), spelling)
            self.assertIsNotNone(got, spelling)

    def test_names_still_parse(self):
        os.environ["XPURT_SHARD_ONLY_NETS"] = "a, b"
        self.assertEqual(self.cc.shard_only_networks_from_env(), {"a", "b"})

    def test_empty_set_holds_every_network_at_one_core(self):
        from workload import Operation
        ops = [Operation([1.0] * len(SHARD_COMBOS), operation_name="net_dispatch_0")]
        ops[0].op_network = "net"
        n = self.cc.restrict_shard_to_networks(ops, SHARD_COMBOS, MACHINES, set())
        self.assertEqual(n, 1)
        wide = {k for k, c in enumerate(SHARD_COMBOS) if len(c) > 1}
        self.assertEqual(ops[0].infeasible_combinations, wide)


if __name__ == "__main__":
    unittest.main()
