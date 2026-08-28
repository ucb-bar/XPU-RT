"""Unit tests for scripts/advice_to_unfuse_hint.py.

`unfuse_advice` and `apply_unfuse_hint` both existed; nothing connected them,
so the verb could be produced and applied but never by the loop itself.

This bridge's job is mostly to REFUSE. Unfusing a working fused kernel is a
large loss -- `conv2d_batchnorm2d_silu_s8` is 97% of yolov8n's runtime and
applies BN and SiLU inside the conv's register tile -- so the only justification
is the one the advice measures: the fused op matched no curated kernel and ran
the scalar reference.
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


bridge = _load(os.path.join(_REPO, "scripts", "advice_to_unfuse_hint.py"),
               "_unfuse_bridge")


def _fused(did, n_subs=3):
    subs = [{"op": k, "shape": {"N": 1, "OC": 16}}
            for k in ("conv2d_s8", "batchnorm2d_s8", "silu_s8")[:n_subs]]
    return {"name": f"l{did}", "op": "conv2d_batchnorm2d_silu_s8",
            "dispatch_id": did, "inputs": ["x"], "outputs": [f"y{did}"],
            "sub_ops": subs, "depends_on": []}


def _item(did, fused_impl="reference/scalar", rec="unfuse"):
    return {"model": "m", "dispatch_id": did, "recommendation": rec,
            "priority": 1, "confidence": "high",
            "constraints": {"requires_constituent_kernels": True},
            "evidence": {"fused_impl": fused_impl, "n_constituents": 3,
                         "service_time_us": 17500.0,
                         "constituent_impls": {"conv2d_s8": "rvv_conv2d_s8.c"},
                         "op": f"m$dispatch_{did}_rvv_x60_"
                               f"conv2d_batchnorm2d_silu_s8_noshape"}}


def _anchor(did=99):
    op = {"name": "a", "op": "add_s8", "dispatch_id": did, "inputs": ["p"],
          "outputs": ["q"], "shape": {"n": 25600}, "depends_on": []}
    item = {"model": "m", "dispatch_id": did, "recommendation": "unchanged",
            "priority": 5, "confidence": "high", "constraints": {},
            "evidence": {"op": f"m$dispatch_{did}_rvv_x60_add_s8_n25600"}}
    return op, item


def _run(items, ops):
    with tempfile.TemporaryDirectory() as td:
        adv, ir = os.path.join(td, "a.json"), os.path.join(td, "g.json")
        out = os.path.join(td, "o.json")
        json.dump({"schedule_id": "s", "advice": items}, open(adv, "w"))
        json.dump({"name": "m", "ops": ops}, open(ir, "w"))
        argv = sys.argv
        sys.argv = ["advice_to_unfuse_hint", "--advice", adv, "--ir", ir,
                    "--model", "m", "--out", out]
        err = io.StringIO()
        try:
            with redirect_stderr(err), redirect_stdout(io.StringIO()):
                code = bridge.main()
        finally:
            sys.argv = argv
        res = json.load(open(out)) if os.path.exists(out) else None
        return code, res, err.getvalue()


class ItEmitsTheContractTheRewriterReads(unittest.TestCase):

    def test_the_contract_matches_apply_unfuse_hint(self):
        rewriter_path = os.path.join(_REPO, "ModelBlaster", "pipeline",
                                     "apply_unfuse_hint.py")
        if not os.path.exists(rewriter_path):
            self.skipTest("ModelBlaster not present in this tree")
        rewriter = _load(rewriter_path, "_apply_unfuse_hint")
        self.assertEqual(bridge.CONTRACT, rewriter.HINT_CONTRACT)

    def test_an_advised_fused_op_becomes_an_unfuse_op(self):
        anchor_op, anchor_item = _anchor()
        code, res, _ = _run([_item(0), anchor_item], [_fused(0), anchor_op])
        self.assertEqual(code, 0)
        self.assertEqual(res["networks"][0]["unfuse_ops"], [{"op": 0}])
        self.assertEqual(res["networks"][0]["network"], "m")

    def test_the_provenance_records_the_standing_invariant(self):
        """A rewrite may not change modelled work unless kernels exist for the
        result. Restoring three ops that each need one is exactly that case."""
        anchor_op, anchor_item = _anchor()
        _, res, _ = _run([_item(0), anchor_item], [_fused(0), anchor_op])
        self.assertIn("check_kernel_coverage", res["_provenance"]["invariant"])


class ItRefusesToWidenTheGate(unittest.TestCase):
    """The gate is narrow on purpose. Before the curated fused kernel existed,
    57 of yolov8_nano's 90 dispatches ran the reference inside a build labelled
    `rvv_x60` -- 99.8% of 4974.8 ms, 0.81x against pure scalar. Unfusing on any
    basis other than that measured fallback reproduces it."""

    def test_a_working_fused_kernel_is_refused(self):
        anchor_op, anchor_item = _anchor()
        code, res, err = _run(
            [_item(0, fused_impl="curated[rvv]/rvv_oc_blocked_bn_silu_epilogue"),
             anchor_item], [_fused(0), anchor_op])
        self.assertEqual(code, 1)
        self.assertIsNone(res)
        self.assertIn("not a reference fallback", err)
        self.assertIn("epilogue fusion", err)

    def test_an_op_the_ir_does_not_record_as_fused_is_refused(self):
        anchor_op, anchor_item = _anchor()
        plain = {"name": "c", "op": "conv2d_s8", "dispatch_id": 0,
                 "inputs": ["x"], "outputs": ["y"], "shape": {"OC": 16},
                 "depends_on": []}
        code, _, err = _run([_item(0), anchor_item], [plain, anchor_op])
        self.assertEqual(code, 1)
        self.assertIn("no fusion to undo", err)

    def test_a_disagreeing_graph_refuses_the_run(self):
        anchor_op, anchor_item = _anchor()
        anchor_op["shape"] = {"n": 6144}
        code, res, err = _run([_item(0), anchor_item], [_fused(0), anchor_op])
        self.assertEqual(code, 2)
        self.assertIn("did not come from", err)
        self.assertIsNone(res)


if __name__ == "__main__":
    unittest.main()
