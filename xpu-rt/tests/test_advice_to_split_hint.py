"""Unit tests for scripts/advice_to_split_hint.py.

The bridge exists so that split advice reaches ModelBlaster at all, and its
whole job is to fail HERE rather than downstream: a hint the rewriter refuses
is worse than no hint, because the reason arrives without the advice that
caused it.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
_SCRIPTS = os.path.join(_REPO, "scripts")


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


bridge = _load(os.path.join(_SCRIPTS, "advice_to_split_hint.py"), "_split_bridge")


def _conv_op(did, oc, ic=64, ih=20, iw=20, kind="conv2d_batchnorm2d_silu_s8"):
    shape = {"N": 1, "IC": ic, "IH": ih, "IW": iw, "OC": oc,
             "OH": ih, "OW": iw, "KH": 3, "KW": 3,
             "SH": 1, "SW": 1, "PH": 1, "PW": 1}
    if kind == "conv2d_s8":
        return {"name": f"c{did}", "op": kind, "dispatch_id": did,
                "inputs": ["x"], "outputs": [f"y{did}"], "shape": shape,
                "depends_on": []}
    return {"name": f"c{did}", "op": kind, "dispatch_id": did,
            "inputs": ["x"], "outputs": [f"y{did}"], "depends_on": [],
            "sub_ops": [{"op": "conv2d_s8", "shape": shape,
                         "inputs": ["x"], "outputs": [f"y{did}_conv"]}]}


def _advice_item(did, svc_us, slot_us, module_name, rec="split"):
    return {"model": "m", "dispatch_id": did, "recommendation": rec,
            "priority": 1, "confidence": "high",
            "evidence": {"service_time_us": svc_us,
                         "periodic_free_slot_us": slot_us,
                         "op": module_name},
            "constraints": {"max_target_piece_us": slot_us}}


def _run(advice_items, ops, extra_argv=()):
    """Run the bridge CLI; return (exit_code, hint_or_None, stderr)."""
    with tempfile.TemporaryDirectory() as td:
        adv = os.path.join(td, "advice.json")
        ir = os.path.join(td, "graph.json")
        out = os.path.join(td, "hint.json")
        with open(adv, "w") as f:
            json.dump({"schema_version": 1, "schedule_id": "s",
                       "advice": advice_items}, f)
        with open(ir, "w") as f:
            json.dump({"name": "m", "ops": ops}, f)
        argv = sys.argv
        sys.argv = ["advice_to_split_hint", "--advice", adv, "--ir", ir,
                    "--model", "m", "--out", out, *extra_argv]
        import io
        from contextlib import redirect_stderr, redirect_stdout
        err = io.StringIO()
        try:
            with redirect_stderr(err), redirect_stdout(io.StringIO()):
                code = bridge.main()
        finally:
            sys.argv = argv
        hint = json.load(open(out)) if os.path.exists(out) else None
        return code, hint, err.getvalue()


#: A dispatch that carries a real shape signature, so graph identity is
#: checkable. Every scenario below needs one -- which is the point: without it
#: the bridge refuses rather than joining on ids alone.
def _anchor(did=99):
    op = {"name": "a", "op": "add_s8", "dispatch_id": did, "inputs": ["p"],
          "outputs": ["q"], "shape": {"n": 25600}, "depends_on": []}
    item = {"model": "m", "dispatch_id": did, "recommendation": "unchanged",
            "priority": 3, "confidence": "high",
            "evidence": {"op": f"m$dispatch_{did}_rvv_x60_add_s8_n25600"},
            "constraints": {}}
    return op, item


class NSplitsIsDerivedFromTheMeasurement(unittest.TestCase):
    """`decision_loop.py` hard-codes n_splits=2. The advice carries a measured
    service time and the slot it has to fit into, so the factor is a
    consequence, not a choice."""

    def test_factor_covers_the_overrun(self):
        anchor_op, anchor_item = _anchor()
        code, hint, _ = _run(
            [_advice_item(0, 17465, 6416, "m$dispatch_0_rvv_x60_"
                                          "conv2d_batchnorm2d_silu_s8_noshape"),
             anchor_item],
            [_conv_op(0, oc=80), anchor_op])
        self.assertEqual(code, 0)
        # 17465/6416 = 2.72 -> needs 3 -> rounded up to a divisor of 80.
        self.assertEqual(hint["networks"][0]["split_ops"],
                         [{"op": 0, "n_splits": 4}])

    def test_factor_is_rounded_up_to_a_divisor_of_the_axis(self):
        """`apply_split_hint` requires OC % n_splits == 0 and refuses
        otherwise, so a factor that does not divide is a hint that gets
        rejected downstream with less context than here."""
        anchor_op, anchor_item = _anchor()
        code, hint, _ = _run(
            [_advice_item(0, 9000, 4000, "m$dispatch_0_rvv_x60_conv2d_s8_"
                                         "N1xIC64xIH20xIW20xOC64xOH20xOW20x"
                                         "KH3xKW3xSH1xSW1xPH1xPW1"),
             anchor_item],
            [_conv_op(0, oc=64, kind="conv2d_s8"), anchor_op])
        self.assertEqual(code, 0)
        n = hint["networks"][0]["split_ops"][0]["n_splits"]
        self.assertEqual(64 % n, 0)
        self.assertGreaterEqual(n, 3)

    def test_a_factor_above_the_cap_is_skipped_not_silently_shrunk(self):
        """4-way OC sharding was measured at +76% total work, so the cap is
        real. A split that cannot reach the target is reported as such rather
        than emitted as if it had."""
        anchor_op, anchor_item = _anchor()
        code, hint, err = _run(
            [_advice_item(0, 100000, 1000, "m$dispatch_0_rvv_x60_"
                                           "conv2d_batchnorm2d_silu_s8_noshape"),
             anchor_item],
            [_conv_op(0, oc=64), anchor_op])
        self.assertEqual(code, 1)
        self.assertIn("SKIPPED", err)
        self.assertIsNone(hint)

    def test_below_target_is_emitted_when_asked_and_says_so(self):
        anchor_op, anchor_item = _anchor()
        code, hint, _ = _run(
            [_advice_item(0, 100000, 1000, "m$dispatch_0_rvv_x60_"
                                           "conv2d_batchnorm2d_silu_s8_noshape"),
             anchor_item],
            [_conv_op(0, oc=64), anchor_op],
            extra_argv=("--allow-below-target",))
        self.assertEqual(code, 0)
        deriv = hint["_provenance"]["derivation"][0]
        self.assertFalse(deriv["reaches_target"])
        self.assertEqual(deriv["n_splits"], 4)


class ItRefusesRatherThanEmittingAHintTheRewriterRejects(unittest.TestCase):

    def test_an_unsplittable_op_kind_is_refused_with_the_reason(self):
        anchor_op, anchor_item = _anchor()
        code, hint, err = _run(
            [_advice_item(0, 9000, 4000,
                          "m$dispatch_0_rvv_x60_maxpool2d_s8_N1xC128xIH5xIW5"),
             anchor_item],
            [{"name": "p", "op": "maxpool2d_s8", "dispatch_id": 0,
              "inputs": ["x"], "outputs": ["y"],
              "shape": {"N": 1, "C": 128, "IH": 5, "IW": 5}, "depends_on": []},
             anchor_op])
        self.assertEqual(code, 1)
        self.assertIn("not\nsplit-capable".replace("\n", " "), err)
        self.assertIsNone(hint)

    def test_an_unknown_dispatch_id_is_refused(self):
        anchor_op, anchor_item = _anchor()
        code, _, err = _run(
            [_advice_item(7, 9000, 4000, "m$dispatch_7_rvv_x60_"
                                         "conv2d_batchnorm2d_silu_s8_noshape"),
             anchor_item],
            [_conv_op(0, oc=64), anchor_op])
        self.assertEqual(code, 1)
        self.assertIn("not in", err)

    def test_advice_with_no_measurement_cannot_derive_a_factor(self):
        anchor_op, anchor_item = _anchor()
        item = _advice_item(0, 0, 0, "m$dispatch_0_rvv_x60_"
                                     "conv2d_batchnorm2d_silu_s8_noshape")
        code, _, err = _run([item, anchor_item], [_conv_op(0, oc=64), anchor_op])
        self.assertEqual(code, 1)
        self.assertIn("service_time_us", err)


class GraphIdentityIsEstablishedBeforeAnythingIsDerived(unittest.TestCase):
    """The failure this exists for, reproduced from the real artifacts: a
    320x320 yolov8_nano profile (226.86 ms) joined against the deployed 64x96
    IR (46.4 ms) passes every per-op check -- same ids, same op kinds, and
    every fused conv profiled as `noshape` -- while deriving split factors
    from service times 25x too large."""

    def test_a_disagreeing_signature_anywhere_refuses_the_whole_run(self):
        anchor_op, anchor_item = _anchor()
        anchor_op["shape"] = {"n": 6144}          # the IR is a different size
        code, hint, err = _run(
            [_advice_item(0, 17465, 6416, "m$dispatch_0_rvv_x60_"
                                          "conv2d_batchnorm2d_silu_s8_noshape"),
             anchor_item],
            [_conv_op(0, oc=80), anchor_op])
        self.assertEqual(code, 2)
        self.assertIn("did not come from", err)
        self.assertIsNone(hint)

    def test_op_kind_alone_does_not_establish_identity(self):
        """A model's topology is identical at every input size, so matching
        kinds prove nothing. With no signature anywhere, the bridge refuses."""
        code, hint, err = _run(
            [_advice_item(0, 17465, 6416, "m$dispatch_0_rvv_x60_"
                                          "conv2d_batchnorm2d_silu_s8_noshape")],
            [_conv_op(0, oc=80)])
        self.assertEqual(code, 2)
        self.assertIn("no dispatch in the advice carries a shape signature",
                      err)
        self.assertIsNone(hint)

    def test_the_escape_hatch_exists_and_is_explicit(self):
        code, hint, _ = _run(
            [_advice_item(0, 17465, 6416, "m$dispatch_0_rvv_x60_"
                                          "conv2d_batchnorm2d_silu_s8_noshape")],
            [_conv_op(0, oc=80)],
            extra_argv=("--skip-identity-check",))
        self.assertEqual(code, 0)
        self.assertIsNotNone(hint)

    def test_list_valued_shape_entries_are_not_false_mismatches(self):
        """A concat's `C_inputs` is a list. Rendering it as Python does rather
        than `|`-joined made 16 of 33 checkable dispatches look wrong."""
        want = bridge.shape_tokens({"N": 1, "H": 20, "W": 20,
                                    "C_inputs": [128, 64], "C_total": 192})
        self.assertIn("C_inputs128|64", want)
        self.assertNotIn("C_inputs[128, 64]", want)


class TheBridgeAndTheRewriterAgreeOnWhatIsSplittable(unittest.TestCase):
    """`SPLITTABLE_AXIS` is mirrored, not imported, so the script runs without
    ModelBlaster on the path. This is what makes the mirror safe: drift fails a
    test instead of producing hints the rewriter refuses."""

    def test_supported_kinds_match_apply_split_hint(self):
        rewriter_path = os.path.join(_REPO, "ModelBlaster", "pipeline",
                                     "apply_split_hint.py")
        if not os.path.exists(rewriter_path):
            self.skipTest("ModelBlaster not present in this tree")
        rewriter = _load(rewriter_path, "_apply_split_hint")
        self.assertEqual(set(bridge.SPLITTABLE_AXIS),
                         set(rewriter._SPLITTABLE),
                         "advice_to_split_hint.SPLITTABLE_AXIS has drifted "
                         "from apply_split_hint._SPLITTABLE; a hint for a kind "
                         "only one of them knows is a hint that gets refused")


if __name__ == "__main__":
    unittest.main()
