"""SpaceMiT K3 port: two 8-core machine types, width shards, `--board k3`.

K3 = 16 CPUs on shared memory. X100 = CPUs 0-7 maps to `cpu_p`, A100 = CPUs
8-15 maps to `cpu_e`, so the K1 two-slot machine model is reused with 8 cores
per slot instead of 4. These tests pin the three places that assumption lives:

* shard-mode combinations for {CPU_P: 8, CPU_E: 8}. A dispatch must never mix
  types (X100 and A100 have different microarchitectures and separate
  profiles, so a mixed block has no cost), and the topo tag must be
  WIDTH-RELATIVE (`topo_0_1` for any 2-wide block, including CPU_E#4+CPU_E#5),
  because the profile trees are keyed by width, not by absolute hart id.
* profile lookup with two distinct hw names. Built on a tmp fixture tree, never
  the live `gen/mb/profile`, which is still being populated.
* the feasibility checker's `--board` switch: k3 accepts CPU_P#0-7/CPU_E#0-7,
  and the k1 default still rejects CPU_P#4 (byte-identical K1 behaviour).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_XPURT = os.path.dirname(_HERE)
_REPO = os.path.dirname(_XPURT)
sys.path.insert(0, _XPURT)

from profile_loader import find_profile_csv  # noqa: E402
from workload_factory import (  # noqa: E402
    build_machine_combinations, topo_tag_for_combination)

CHECK = os.path.join(_REPO, "scripts", "check_schedule_feasibility.py")
MACHINES, COMBOS = build_machine_combinations({"CPU_P": 8, "CPU_E": 8},
                                              mode="shard")


def _kind(core):
    return core.split("#")[0]


class K3ShardCombinations(unittest.TestCase):
    def test_fifteen_combinations_per_kind(self):
        """8 singles + 4 pairs + 2 quads + 1 octet = 15, for each type."""
        for kind in ("CPU_P", "CPU_E"):
            n = sum(1 for c in COMBOS if _kind(c[0]) == kind)
            self.assertEqual(n, 15, kind)
        self.assertEqual(len(COMBOS), 30)

    def test_widths_are_1_2_4_8_for_each_kind(self):
        for kind in ("CPU_P", "CPU_E"):
            widths = sorted({len(c) for c in COMBOS if _kind(c[0]) == kind})
            self.assertEqual(widths, [1, 2, 4, 8], kind)

    def test_a_combination_never_mixes_types(self):
        for c in COMBOS:
            self.assertEqual(len({_kind(x) for x in c}), 1, c)

    def test_topo_tags_are_width_relative_and_identical_across_types(self):
        """A100's 2-wide block at cores 4-5 is costed from `topo_0_1`, the same
        directory name the X100 uses -- the hw name (profile_hw) is what
        distinguishes the two trees."""
        want = {1: "topo_0", 2: "topo_0_1", 4: "topo_0_1_2_3",
                8: "topo_0_1_2_3_4_5_6_7"}
        for c in COMBOS:
            self.assertEqual(topo_tag_for_combination(c), want[len(c)], c)
        quad_hi = ["CPU_E#4", "CPU_E#5", "CPU_E#6", "CPU_E#7"]
        self.assertIn(quad_hi, COMBOS)
        self.assertEqual(topo_tag_for_combination(quad_hi), "topo_0_1_2_3")


class K3TwoHwProfileLookup(unittest.TestCase):
    """`profile_hw` maps cpu_p/cpu_e to distinct trees; each must resolve on its own."""

    def _touch(self, root, hw, topo):
        d = os.path.join(root, "gen", "profile", hw, "spacemit_k3", "dronet",
                         "dronet.int8", f"dronet_spacemit_k3_{hw}_dronet.int8",
                         topo)
        os.makedirs(d)
        p = os.path.join(d, "results.csv")
        open(p, "w").write("dispatch_id\n")
        return p

    def test_x100_and_a100_resolve_to_their_own_files(self):
        with tempfile.TemporaryDirectory() as root:
            x = self._touch(root, "rvv_k3_x100", "topo_0_1")
            a = self._touch(root, "rvv_k3_a100", "topo_0_1")
            kw = dict(model="dronet", target="spacemit_k3",
                      basename="dronet.int8", topo_tag="topo_0_1")
            self.assertEqual(find_profile_csv(root, hw="rvv_k3_x100", **kw), x)
            self.assertEqual(find_profile_csv(root, hw="rvv_k3_a100", **kw), a)

    def test_a_missing_a100_tree_is_not_answered_with_the_x100_one(self):
        """The two clusters have different timings; borrowing one for the other
        would give a full, plausible, wrong cost table."""
        with tempfile.TemporaryDirectory() as root:
            self._touch(root, "rvv_k3_x100", "topo_0")
            self.assertIsNone(find_profile_csv(
                root, hw="rvv_k3_a100", model="dronet",
                target="spacemit_k3", basename="dronet.int8",
                topo_tag="topo_0"))


class K3FeasibilityBoardFlag(unittest.TestCase):
    def _run(self, *extra, targets):
        sched = {"dispatches": {
            f"d{i}": {"start_time": 0.0, "duration": 1.0,
                      "hardware_target": t, "dependencies": []}
            for i, t in enumerate(targets)}}
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "s.json")
            json.dump(sched, open(p, "w"))
            return subprocess.run(
                [sys.executable, CHECK, "--schedule", p, *extra],
                capture_output=True, text=True)

    ALL = [f"CPU_P#{i}" for i in range(8)] + [f"CPU_E#{i}" for i in range(8)]

    def test_k3_accepts_all_sixteen_cores(self):
        r = self._run("--board", "k3", targets=self.ALL)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("FEASIBLE", r.stdout)

    def test_k3_refuses_core_eight_of_a_type(self):
        r = self._run("--board", "k3", targets=["CPU_P#8"])
        self.assertEqual(r.returncode, 1, r.stdout + r.stderr)

    def test_k1_default_still_rejects_cpu_p_4(self):
        r = self._run(targets=["CPU_P#4"])
        self.assertEqual(r.returncode, 1, r.stdout + r.stderr)
        r = self._run("--board", "k1", targets=["CPU_P#4"])
        self.assertEqual(r.returncode, 1, r.stdout + r.stderr)


if __name__ == "__main__":
    unittest.main()


class K3CapabilitiesTests(unittest.TestCase):
    """IME runs on every K3 CPU (measured), so an ime_* profile on either
    kind must pass the config-time legality check that rejects it on the K1."""

    def test_k3_accepts_ime_on_both_kinds(self):
        from capabilities import check_profile_hw_map, capabilities_for_target
        check_profile_hw_map({"cpu_p": "rvv_k3_x100", "cpu_e": "ime_k3_a100"},
                             capabilities=capabilities_for_target("spacemit_k3"))

    def test_k1_still_rejects_ime_on_cluster1(self):
        from capabilities import (check_profile_hw_map, capabilities_for_target,
                                  IllegalPlacement)
        with self.assertRaises(IllegalPlacement):
            check_profile_hw_map({"cpu_p": "rvv_x60", "cpu_e": "ime_x60"},
                                 capabilities=capabilities_for_target("spacemit_x60"))
