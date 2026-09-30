"""Costing a build that was not table-guided.

`scripts/make_ime_profile.py` derives an `ime_x60` profile from the measured RVV
one and a matched pair of board runs. Its default gate is the measured table's
answer to "does the engine deserve this op" (`ime_cost.ime_useful`), and it
keeps only the dispatches the IME run ran FASTER.

Neither is right for a build made with `MB_IME_FORCE=1`, which takes the IME
kernel for every op that HAS one -- including ops the table never measured and
ops it measures as losses. Costing that build from the table's answer would
leave its extra ops priced as RVV, and a schedule solved against those costs is
a schedule for a build nobody made. `--ime-ops-from-picks` asks the build's own
`kernel_picks.json` instead, and then records the measured ratio whichever way
it points. These tests pin both halves.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOL = os.path.join(REPO, "scripts", "make_ime_profile.py")

_spec = importlib.util.spec_from_file_location("_make_ime_profile", TOOL)
MIP = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(MIP)


class ImeOpsFromPicks(unittest.TestCase):
    def _picks(self, tmp, body):
        p = os.path.join(tmp, "kernel_picks.json")
        json.dump(body, open(p, "w"))
        return p

    def test_forced_ops_are_taken_verbatim(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = self._picks(tmp, {"target": "ime_x60", "table_guided": False,
                                  "ime_force": True,
                                  "ime_forced_ops": ["conv2d_s8", "conv2d_batchnorm2d_silu_s8"]})
            self.assertEqual(MIP._ime_ops_from_picks(p),
                             {"conv2d_s8", "conv2d_batchnorm2d_silu_s8"})

    def test_falls_back_to_the_picks_themselves(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = self._picks(tmp, {"target": "ime_x60", "picks": {
                "conv2d_s8": "curated[ime]/ime_vmadot_4x4x8",
                "add_s8": "curated[rvv]/direct"}})
            self.assertEqual(MIP._ime_ops_from_picks(p), {"conv2d_s8"})

    def test_a_build_with_nothing_on_the_engine_is_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = self._picks(tmp, {"target": "rvv_x60",
                                  "picks": {"add_s8": "curated[rvv]/direct"}})
            with self.assertRaises(SystemExit) as cm:
                MIP._ime_ops_from_picks(p)
            self.assertIn("names no op on the matrix engine", str(cm.exception))


class WinnersOnly(unittest.TestCase):
    """The only-if-better rule, and the one case that suspends it."""

    def _pair(self, tmp, rvv_cycles, ime_cycles):
        def w(name, cyc):
            p = os.path.join(tmp, name)
            rows = "\n".join(f"{i},d{i},conv2d_s8,{c}" for i, c in enumerate(cyc))
            open(p, "w").write(
                "=== MODELBLASTER_PROFILE_BEGIN ===\n"
                "dispatch_id,name,op,cycles\n" + rows + "\n=== END ===\n")
            return p
        return f"{w('rvv.txt', rvv_cycles)}:{w('ime.txt', ime_cycles)}"

    def test_default_keeps_winners_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            spec = self._pair(tmp, [100, 100], [50, 200])
            ratios, _, _, n = MIP._ratios_from_runs(spec)
            self.assertEqual(n, 2)
            self.assertEqual(sorted(ratios), [0])          # dispatch 1 lost, so no cell
            self.assertAlmostEqual(ratios[0], 2.0)

    def test_a_chosen_build_records_the_losses_too(self):
        with tempfile.TemporaryDirectory() as tmp:
            spec = self._pair(tmp, [100, 100], [50, 200])
            ratios, _, _, _ = MIP._ratios_from_runs(spec, winners_only=False)
            self.assertEqual(sorted(ratios), [0, 1])
            self.assertAlmostEqual(ratios[0], 2.0)
            self.assertAlmostEqual(ratios[1], 0.5)          # costs MORE on the engine
            self.assertLess(ratios[1], 1.0)


class TheDeployedForcedProfile(unittest.TestCase):
    """The artifact this repo ships for the blanket arm, as it stands on disk."""

    P = ("gen/mb_force/profile/ime_x60/spacemit_x60/yolov8_nano_64x96/"
         "yolov8_nano_64x96.int8/yolov8_nano_64x96_spacemit_x60_ime_x60_"
         "yolov8_nano_64x96.int8/topo_0_1_2_3")

    def setUp(self):
        self.d = os.path.join(REPO, self.P)
        if not os.path.exists(self.d):
            self.skipTest("gen/mb_force profile not present")

    def test_every_conv_dispatch_carries_an_ime_cell(self):
        rows = list(csv.DictReader(open(os.path.join(self.d, "results.csv"))))
        conv = [r for r in rows if (r.get("op") or "").startswith("conv2d")]
        self.assertEqual(len(conv), 63)
        self.assertTrue(all("ime" in (r.get("implementation") or "").lower() for r in conv))
        other = [r for r in rows if not (r.get("op") or "").startswith("conv2d")]
        self.assertTrue(all("ime" not in (r.get("implementation") or "").lower()
                            for r in other),
                        "an op with no IME kernel must not carry an ime cell")

    def test_provenance_says_it_is_not_only_if_better(self):
        prov = json.load(open(os.path.join(self.d, "PROVENANCE.json")))
        self.assertIs(prov["only_if_better"], False)
        self.assertIn("kernel_picks.json", prov["ime_ops_source"])
        self.assertGreater(prov["n_rows_derived_slower_than_rvv"], 0)


if __name__ == "__main__":
    unittest.main()
