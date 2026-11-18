"""scripts/check_k3_placement.py against hand-written placement rows (K3)."""
from __future__ import annotations

import os
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(_REPO, "scripts"))

import check_k3_placement as chk  # noqa: E402

REGISTRY = os.path.join(_REPO, "zephyr-chipyard-sw", "modelblaster", "cores", "spacemit_k3.json")
PAIRS = chk.parse_pairs("x100=ime_k3_x100,a100=ime_k3_a100",
                        "x100:rvv=rvv_k3,x100:ime=ime_k3_x100,a100:rvv=rvv_k3_a100,a100:ime=ime_k3_a100")


def row(eid, kind, impl, hart, before, after=None, harts=None):
    return {"entry_id": str(eid), "network": "yolov8_nano", "instance": "0",
            "dispatch_id": str(eid), "core_kind": kind, "impl": impl,
            "hart": str(hart), "n_harts": str(len((harts or str(hart)).split())),
            "harts": harts or str(hart),
            "cpu_before": str(before), "cpu_after": str(before if after is None else after)}


@unittest.skipUnless(os.path.exists(REGISTRY), "ModelBlaster submodule not checked out")
class CheckK3Placement(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.by_cpu = chk.load_registry(REGISTRY)

    def test_a_correct_mixed_run_passes(self):
        rows = [row(0, "a100", "ime", 8, 8), row(1, "x100", "rvv", 0, 0),
                row(2, "a100", "rvv", 9, 9), row(3, "x100", "ime", 3, 3),
                row(4, "a100", "ime", 10, 10, harts="10 11"),
                row(5, "x100", "x100", 1, 1)]          # impl unset: the kind's own pairing
        self.assertEqual(chk.check_rows(rows, self.by_cpu, PAIRS), [])

    def test_each_violation_is_reported(self):
        cases = {
            "never ran": row(0, "a100", "ime", 8, -1, -1),
            "ran on 0": row(0, "a100", "ime", 8, 0),                  # unregistered A100 worker
            "ran on 1 -> 2": row(0, "x100", "rvv", 1, 1, 2),          # migrated mid-kernel
            "span kinds": row(0, "x100", "rvv", 7, 7, harts="7 8"),
            "no backend": row(0, "a100", "int4", 8, 8),
        }
        for want, r in cases.items():
            with self.subTest(want):
                bad = chk.check_rows([r], self.by_cpu, PAIRS)
                self.assertTrue(any(want in b for b in bad), bad)

    def test_a_pairing_slip_is_not_legal(self):
        slip = chk.parse_pairs("x100=ime_k3_x100,a100=ime_k3_x100", "")   # X100 tile on an A100
        bad = chk.check_rows([row(0, "a100", "a100", 8, 8)], self.by_cpu, slip)
        self.assertTrue(any("not legal on a100" in b for b in bad), bad)
        slip = chk.parse_pairs("x100=rvv_k3_a100,a100=ime_k3_a100", "")   # zvl1024b on an X100
        bad = chk.check_rows([row(0, "x100", "x100", 0, 0)], self.by_cpu, slip)
        self.assertTrue(any("not legal on x100" in b for b in bad), bad)

    def test_view_ops_without_a_kernel_are_skipped(self):
        r = row(0, "x100", "x100", 0, -1, -1)
        r["dispatch_id"] = "-1"
        self.assertEqual(chk.check_rows([r], self.by_cpu, PAIRS), [])


if __name__ == "__main__":
    unittest.main()
