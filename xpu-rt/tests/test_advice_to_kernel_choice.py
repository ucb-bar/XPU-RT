"""Unit tests for scripts/advice_to_kernel_choice.py.

`choose_implementation` was emitted and ignored: its only consumer rewrote
`.vmfb` paths, an IREE mechanism this path does not have. These pin the
behaviour of the replacement, whose whole difficulty is that the advice is per
DISPATCH and the mechanism it drives is per OP KIND.
"""

from __future__ import annotations

import importlib.util
import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


bridge = _load(os.path.join(_REPO, "scripts", "advice_to_kernel_choice.py"),
               "_kernel_choice")

SIG = "N1xC128xIH5xIW5xOH5xOW5xKH5xKW5xSH1xSW1xPH2xPW2xDH1xDW1"
SHAPE = {"N": 1, "C": 128, "IH": 5, "IW": 5, "OH": 5, "OW": 5,
         "KH": 5, "KW": 5, "SH": 1, "SW": 1, "PH": 2, "PW": 2,
         "DH": 1, "DW": 1}


def _op(did, kind="maxpool2d_s8"):
    return {"name": f"p{did}", "op": kind, "dispatch_id": did,
            "inputs": ["x"], "outputs": [f"y{did}"], "shape": dict(SHAPE),
            "depends_on": []}


def _chose(did, proposed="scalar", gain=0.215, kind="maxpool2d_s8"):
    return {"model": "m", "dispatch_id": did,
            "recommendation": "choose_implementation", "priority": 2,
            "confidence": "medium", "constraints": {},
            "evidence": {"baseline_impl": "rvv_x60", "proposed_impl": proposed,
                         "gain_fraction": gain,
                         "baseline_kernel": "curated[rvv]/direct",
                         "op": f"m$dispatch_{did}_rvv_x60_{kind}_{SIG}"}}


def _unchanged(did, best_alternative="scalar", kind="maxpool2d_s8"):
    ev = {"baseline_impl": "rvv_x60",
          "op": f"m$dispatch_{did}_rvv_x60_{kind}_{SIG}"}
    if best_alternative:
        ev["best_alternative"] = best_alternative
        ev["gain_fraction"] = 0.104
    return {"model": "m", "dispatch_id": did, "recommendation": "unchanged",
            "priority": 5, "confidence": "medium", "constraints": {},
            "evidence": ev}


def _run(items, ops):
    with tempfile.TemporaryDirectory() as td:
        adv, ir = os.path.join(td, "a.json"), os.path.join(td, "g.json")
        out = os.path.join(td, "o.json")
        json.dump({"schedule_id": "s", "advice": items}, open(adv, "w"))
        json.dump({"name": "m", "ops": ops}, open(ir, "w"))
        argv = sys.argv
        sys.argv = ["advice_to_kernel_choice", "--advice", adv, "--ir", ir,
                    "--model", "m", "--out", out]
        err = io.StringIO()
        try:
            with redirect_stderr(err), redirect_stdout(io.StringIO()):
                code = bridge.main()
        finally:
            sys.argv = argv
        res = json.load(open(out)) if os.path.exists(out) else None
        return code, res, err.getvalue()


class SilenceIsNotObjection(unittest.TestCase):
    """The measured case this file exists for.

    yolov8_nano's three `maxpool2d_s8` dispatches are the same kind AND the
    same shape, and all three measure faster on scalar: 21.5%, 21.2%, 10.4%.
    Only two cleared `implementation_advice`'s 0.05 ms absolute floor; the
    third missed it by 0.0033 ms. Reading its silence as "the curated kernel
    won here" refuses a pin all three agree with.
    """

    def test_a_sibling_under_the_floor_does_not_block_the_pin(self):
        code, res, _ = _run(
            [_chose(40), _chose(41), _unchanged(42, "scalar")],
            [_op(40), _op(41), _op(42)])
        self.assertEqual(code, 0)
        self.assertEqual(res["keep_reference_ops"], ["maxpool2d_s8"])
        self.assertEqual(res["_provenance"]["pins"][0]
                         ["concurring_below_threshold"], [42])

    def test_a_sibling_that_genuinely_prefers_the_curated_kernel_blocks_it(self):
        """No `best_alternative` means nothing beat the baseline there, so the
        pin would make that dispatch slower. There is no op-kind pin that
        satisfies both, and the resolutions are decisions."""
        code, res, err = _run(
            [_chose(40), _chose(41), _unchanged(42, best_alternative=None)],
            [_op(40), _op(41), _op(42)])
        self.assertEqual(code, 1)
        self.assertIsNone(res)
        self.assertIn("No pin satisfies both", err)
        self.assertIn("curated kernel wins", err)

    def test_a_sibling_with_no_advice_at_all_blocks_it(self):
        """Unknown direction is not permission."""
        code, res, err = _run([_chose(40), _chose(41)],
                              [_op(40), _op(41), _op(42)])
        self.assertEqual(code, 1)
        self.assertIn("direction is unknown", err)


class OnlyAdviceAPinCanExpress(unittest.TestCase):

    def test_proposing_a_vector_impl_is_skipped_not_pinned(self):
        """'prefer the vector build here' is what the curated swap already
        does by default; no pin expresses it, so emitting one would be a
        change in the wrong direction."""
        code, res, err = _run([_chose(40, proposed="IME")], [_op(40)])
        self.assertEqual(code, 1)
        self.assertIn("curated swap already", err)
        self.assertIsNone(res)

    def test_the_emitted_flag_is_the_codegen_flag(self):
        code, res, _ = _run([_chose(40), _unchanged(41, "scalar")],
                            [_op(40), _op(41)])
        self.assertEqual(code, 0)
        self.assertEqual(res["generate_kernels_flag"],
                         "--keep-reference-ops maxpool2d_s8")

    def test_the_advised_gain_is_recorded_as_not_transferable(self):
        """The proposed time was measured in the proposed impl's OWN build:
        the same reference C compiled with `-march=rv64gc`. Kept inside an
        rvv build it is compiled with vector flags. The mechanism transfers,
        the number does not."""
        _, res, _ = _run([_chose(40), _unchanged(41, "scalar")],
                         [_op(40), _op(41)])
        pin = res["_provenance"]["pins"][0]
        self.assertTrue(pin["advised_gain_is_from_another_build"])
        self.assertIn("re-profiled", res["_provenance"]["caveat"])


class GraphIdentityGuardsThisBridgeToo(unittest.TestCase):
    def test_a_disagreeing_signature_refuses_the_run(self):
        op = _op(40)
        op["shape"]["C"] = 64          # the IR is a different graph
        code, res, err = _run([_chose(40)], [op])
        self.assertEqual(code, 2)
        self.assertIn("did not come from", err)
        self.assertIsNone(res)


if __name__ == "__main__":
    unittest.main()
