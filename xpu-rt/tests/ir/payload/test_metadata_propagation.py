"""Tests for B.2 — FX meta propagation into xDSL op attributes.

Exercises ``_forward_fx_meta`` directly (import-level unit) and
through the full ``FXImporter`` flow on a tiny FX module.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from compgen.capture.torch_export import capture_model
from compgen.ir.payload.decompositions import DecompResult
from compgen.ir.payload.import_fx import FXImporter, _forward_fx_meta
from xdsl.dialects.builtin import Float32Type, StringAttr, TensorType
from xdsl.dialects.tensor import EmptyOp


def _make_op():
    t = TensorType(Float32Type(), [2, 4])
    return EmptyOp([], t)


def test_forward_fx_meta_pattern_hint_only():
    op = _make_op()
    _forward_fx_meta(op, {"_compgen_pattern": "softmax"})
    assert op.attributes["compgen._pattern_hint"].data == "softmax"


def test_forward_fx_meta_decomp_hint_wins_when_fx_absent():
    op = _make_op()
    _forward_fx_meta(op, {}, decomp_hint="rms_norm")
    assert op.attributes["compgen._pattern_hint"].data == "rms_norm"


def test_forward_fx_meta_fx_hint_wins_over_decomp():
    op = _make_op()
    _forward_fx_meta(op, {"_compgen_pattern": "layer_norm"}, decomp_hint="other")
    assert op.attributes["compgen._pattern_hint"].data == "layer_norm"


def test_forward_fx_meta_is_idempotent():
    op = _make_op()
    op.attributes["compgen._pattern_hint"] = StringAttr("pre_existing")
    _forward_fx_meta(op, {"_compgen_pattern": "other"})
    # Idempotent: never overwrites
    assert op.attributes["compgen._pattern_hint"].data == "pre_existing"


def test_forward_fx_meta_transpose_absorbed_and_fuse_dequant():
    op = _make_op()
    _forward_fx_meta(
        op,
        {
            "_compgen_pattern": "matmul",
            "_compgen_transpose_absorbed": True,
            "_compgen_fuse_dequant": True,
        },
    )
    assert op.attributes["compgen.transpose_absorbed"].data == "true"
    assert op.attributes["compgen.fuse_dequant"].data == "true"


def test_decomp_result_accepts_pattern_hint():
    r = DecompResult(ops=[], pattern_hint="foo")
    assert r.pattern_hint == "foo"


def test_decomp_result_default_pattern_hint_is_none():
    r = DecompResult(ops=[])
    assert r.pattern_hint is None


# --- Integration: end-to-end with a tiny model ---


class _TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(8, 8)

    def forward(self, x):
        return torch.relu(self.fc(x))


def test_importer_does_not_crash_on_extended_table():
    """Regression: the expanded DECOMPOSITION_TABLE must still handle
    a bare MLP without raising. This is the smallest end-to-end probe
    that the metadata-forwarding path compiles cleanly."""
    m = _TinyMLP()
    x = torch.randn(2, 8)
    ep = capture_model(m, (x,))
    importer = FXImporter()
    module = importer.import_graph(ep)
    # No errors; some ops decomposed
    errors = [d for d in importer.diagnostics if d.level == "error"]
    assert errors == []
    assert importer.decomposed_count >= 1
