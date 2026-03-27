"""Tests for canonicalization pass."""

from __future__ import annotations

import sys
from pathlib import Path

from compgen.capture.torch_export import capture_model
from compgen.ir.payload.canonicalize import CanonicalizationReport, CanonicalizePass, canonicalize
from compgen.ir.payload.import_fx import fx_to_xdsl

EXAMPLES_DIR = Path(__file__).parent.parent.parent.parent / "examples" / "models"


def _get_simple_mlp():
    sys.path.insert(0, str(EXAMPLES_DIR))
    from simple_mlp import SimpleMLP, get_sample_inputs
    return SimpleMLP(), get_sample_inputs()


def test_canonicalization_report_fields() -> None:
    report = CanonicalizationReport(ops_before=10, ops_after=7, transforms_applied=["fold_constants", "dce"])
    assert report.ops_before == 10
    assert report.ops_after == 7
    assert len(report.transforms_applied) == 2


def test_canonicalization_report_defaults() -> None:
    report = CanonicalizationReport(ops_before=5, ops_after=5)
    assert report.transforms_applied == []
    assert report.warnings == []


def test_canonicalize_pass_run() -> None:
    """CanonicalizePass.run should return module and report with op counts."""
    model, inputs = _get_simple_mlp()
    ep = capture_model(model, inputs)
    module, _ = fx_to_xdsl(ep)

    canon_pass = CanonicalizePass()
    result_module, report = canon_pass.run(module)
    assert result_module is not None
    assert report.ops_before > 0
    assert report.ops_after == report.ops_before  # MVP: no transforms


def test_canonicalize_convenience() -> None:
    """canonicalize() should work as one-call wrapper."""
    model, inputs = _get_simple_mlp()
    ep = capture_model(model, inputs)
    module, _ = fx_to_xdsl(ep)

    result_module, report = canonicalize(module)
    assert result_module is not None
    assert isinstance(report, CanonicalizationReport)
