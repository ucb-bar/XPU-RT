"""K3: two core kinds profiled as two targets, in one schedule (k3-work TASK-016).

The SpacemiT K3 has X100 cores (cpus 0-7) and A100 cores (cpus 8-15).
ModelBlaster's K3 runner files each kind's profiles under its own folder
(rvv_k3_x100, ime_k3_x100, rvv_k3_a100, ime_k3_a100) and under the target
named by PROFILE_CPU (spacemit_x100 / spacemit_a100), with topo tags counted
from 0 within the kind (MB_TOPO_RELATIVE). With a single `profile.target` the
scheduler would read one kind's target for both. Per-kind
`hardware.profile.targets` turn each combination's hw into a `hw@target`
label, so every cost cell reads its own kind's measurement -- including each
implementation of a kind under `scheduler.enable_impls`.

Absent from every K1 config, which is unchanged.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import unittest

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_XPURT = os.path.dirname(_HERE)
_REPO = os.path.dirname(_XPURT)
sys.path.insert(0, _REPO)
sys.path.insert(0, _XPURT)

from profile_loader import load_profiled_processing_times, split_hw_label  # noqa: E402
from workload_factory import (  # noqa: E402
    build_machine_combinations,
    expand_combinations_with_impls,
    machine_type_prefix,
)

SCRIPT = os.path.join(_REPO, "scripts", "run_xpurt_schedule.py")
GRAPH = os.path.join(_HERE, "data", "mlp_control.int8", "mlp_control.int8_dispatch_graph.json")   # 7 dispatches
TARGETS = {"CPU_P": "spacemit_x100", "CPU_E": "spacemit_a100"}
IMPLS = {"cpu_p": {"rvv": "rvv_k3_x100", "ime": "ime_k3_x100"},
         "cpu_e": {"rvv": "rvv_k3_a100", "ime": "ime_k3_a100"}}

# Synthetic per-dispatch times (ms) by (hw folder, target, topo).
TREE = {
    ("rvv_k3_x100", "spacemit_x100", "topo_0"): 1.0,
    ("ime_k3_x100", "spacemit_x100", "topo_0"): 2.0,
    ("rvv_k3_a100", "spacemit_a100", "topo_0"): 3.0,
    ("ime_k3_a100", "spacemit_a100", "topo_0"): 4.0,
    ("rvv_k3_x100", "spacemit_x100", "topo_0_1"): 0.6,
    ("ime_k3_a100", "spacemit_a100", "topo_0_1"): 2.5,
    ("rvv_k3_a100", "spacemit_a100", "topo_0_1"): 1.7,
    ("ime_k3_x100", "spacemit_x100", "topo_0_1"): 1.1,
    # Decoys where the single-target path (profile.target = spacemit_x100)
    # would read the A100 folders, and the other kind's target.
    ("rvv_k3_a100", "spacemit_x100", "topo_0"): 0.01,
    ("ime_k3_a100", "spacemit_x100", "topo_0"): 0.01,
    ("ime_k3_x100", "spacemit_a100", "topo_0"): 0.01,
}
HDR = ("dispatch_id,module_name,vmfb_path,mlir_path,mean_time,mean_unit,mean_time_ns,"
       "returncode,log_path,source,op,shape,cycles,cycles_n,cycles_cv_pct,implementation")


def write_tree(root: str) -> None:
    for (hw, target, topo), t in TREE.items():
        d = os.path.join(root, "gen", "profile", hw, target, "mlp_control", "mlp_control.int8",
                         f"mlp_control_{target}_{hw}_mlp_control.int8", topo)
        os.makedirs(d, exist_ok=True)
        rows = [HDR] + [f"{i},mlp_control$dispatch_{i}_{hw}_linear_s8,,,{t},ms,{t * 1e6},0,,k3,"
                        f"linear_s8,,1,49,0.1,curated/x" for i in range(7)]
        with open(os.path.join(d, "results.csv"), "w") as f:
            f.write("\n".join(rows) + "\n")


def load_script_module():
    spec = importlib.util.spec_from_file_location("run_xpurt_schedule_k3test", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def labelled(combos, hws):
    return [f"{hw}@{TARGETS[machine_type_prefix(c[0])]}" for hw, c in zip(hws, combos)]


class Labels(unittest.TestCase):
    def test_hw_labels_split_into_folder_and_target(self):
        self.assertEqual(split_hw_label("ime_k3_a100@spacemit_a100", "spacemit_x100"),
                         ("ime_k3_a100", "spacemit_a100"))
        self.assertEqual(split_hw_label("rvv_x60", "spacemit_x60"), ("rvv_x60", "spacemit_x60"))


class ProfileTargetsConfig(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = load_script_module()

    def parse(self, profile, machines=None):
        return self.mod._parse_profile_targets(profile, machines or {"CPU_P": 8, "CPU_E": 8})

    def full(self):
        return {"target": "spacemit_k3",
                "targets": {"cpu_p": "spacemit_x100", "cpu_e": "spacemit_a100"}}

    def test_a_k1_profile_block_yields_nothing(self):
        self.assertIsNone(self.parse({"target": "spacemit_x60", "topo_tag": "topo_0"}))

    def test_the_k3_block_is_read_with_upper_case_kinds(self):
        self.assertEqual(self.parse(self.full()), TARGETS)

    def test_every_machine_kind_needs_an_entry(self):
        cfg = self.full()
        del cfg["targets"]["cpu_e"]
        with self.assertRaises(ValueError):
            self.parse(cfg)

    def test_malformed_values_are_rejected(self):
        for name, value in (("list", ["spacemit_x100"]),
                            ("None target", {"cpu_p": "spacemit_x100", "cpu_e": None}),
                            ("@ in target", {"cpu_p": "spacemit_x100", "cpu_e": "a@b"}),
                            ("/ in target", {"cpu_p": "spacemit_x100", "cpu_e": "a/b"}),
                            ("unknown kind", {"cpu_p": "x", "cpu_e": "y", "gpu": "z"})):
            with self.subTest(name), self.assertRaises(ValueError):
                self.parse({"targets": value})

    def test_a_zero_count_kind_may_stay_listed(self):
        self.assertEqual(self.parse(self.full(), machines={"CPU_P": 8}), {"CPU_P": "spacemit_x100"})


class CostCellsComeFromTheirOwnKind(unittest.TestCase):
    """Loader level: every cost cell reads its kind's target, its own folder."""

    def load(self, combos, labels):
        net = {"mlp_control": {"id": 0, "identifier": "mlp_control", "dispatch_deps_path": GRAPH}}
        with tempfile.TemporaryDirectory() as tmp:
            write_tree(tmp)
            times, _, _, _ = load_profiled_processing_times(
                networks=net, repo_base_path=tmp, machine_combinations=combos,
                combo_hw=labels, profile_target="spacemit_x100",
                cpu_p_profile_hw=labels[0], cpu_e_profile_hw=labels[-1],
                rng=np.random.default_rng(0), p_core_speedup=1.0)
        return times

    def test_one_folder_per_kind(self):
        _, combos = build_machine_combinations({"CPU_P": 1, "CPU_E": 1})
        times = self.load(combos, labelled(combos, ["rvv_k3_x100", "ime_k3_a100"]))
        for i in range(7):
            self.assertEqual(times[f"mlp_control_dispatch_{i}"], [1.0, 4.0])   # no decoy

    def test_every_implementation_reads_its_kinds_target(self):
        """enable_impls: four (kind, impl) folders, each at its own target and width."""
        _, combos = build_machine_combinations({"CPU_P": 2, "CPU_E": 2}, mode="shard")
        hws = [IMPLS[machine_type_prefix(c[0]).lower()]["rvv"] for c in combos]
        combos, hws, impls = expand_combinations_with_impls(combos, hws, IMPLS)
        times = self.load(combos, labelled(combos, hws))
        want = {("rvv_k3_x100", 1): 1.0, ("ime_k3_x100", 1): 2.0,
                ("rvv_k3_a100", 1): 3.0, ("ime_k3_a100", 1): 4.0,
                ("rvv_k3_x100", 2): 0.6, ("ime_k3_x100", 2): 1.1,
                ("rvv_k3_a100", 2): 1.7, ("ime_k3_a100", 2): 2.5}
        expect = [want[(hw, len(c))] for hw, c in zip(hws, combos)]
        self.assertEqual(sorted(set(impls)), ["ime", "rvv"])
        for i in range(7):
            self.assertEqual(times[f"mlp_control_dispatch_{i}"], expect)

    def test_a_missing_kind_profile_names_its_target(self):
        _, combos = build_machine_combinations({"CPU_P": 1, "CPU_E": 1})
        with self.assertRaises(FileNotFoundError) as cm:
            self.load(combos, labelled(combos, ["rvv_k3_x100", "nosuch_a100"]))
        self.assertIn("gen/profile/nosuch_a100/spacemit_a100/", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
