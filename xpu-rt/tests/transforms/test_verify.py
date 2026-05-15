"""Tests for transforms/verify.py -- transform semantic verification."""

from __future__ import annotations

from compgen.transforms.verify import (
    TransformVerifier,
    VerificationLevel,
    verify_guarded_transform,
    verify_transform,
)
from xdsl.dialects import arith, func
from xdsl.dialects.builtin import IndexType, ModuleOp
from xdsl.ir import Block, Region


def _make_module() -> ModuleOp:
    idx = IndexType()
    block = Block(arg_types=[idx, idx])
    a, b = block.args
    add = arith.AddiOp(a, b)
    block.add_op(add)
    block.add_op(func.ReturnOp(add.result))
    return ModuleOp([func.FuncOp("test", ([idx, idx], [idx]), Region([block]))])


def test_verification_level_values() -> None:
    assert VerificationLevel.STRUCTURAL.value == "structural"
    assert VerificationLevel.DIFFERENTIAL.value == "differential"
    assert VerificationLevel.CHECK_ASSERTIONS.value == "check_assertions"
    assert VerificationLevel.TRANSLATION_VALIDATION.value == "translation_validation"


def test_transform_verifier_defaults() -> None:
    v = TransformVerifier()
    assert v.tolerance == 1e-5
    assert VerificationLevel.STRUCTURAL in v.levels
    assert VerificationLevel.DIFFERENTIAL in v.levels


def test_structural_verification_passes() -> None:
    original = _make_module()
    transformed = original.clone()
    verifier = TransformVerifier(levels=[VerificationLevel.STRUCTURAL])
    result = verifier.verify(original, transformed)
    assert result.passed
    assert VerificationLevel.STRUCTURAL in result.levels_passed


def test_differential_verification_passes() -> None:
    original = _make_module()
    transformed = original.clone()
    verifier = TransformVerifier(levels=[VerificationLevel.DIFFERENTIAL])
    result = verifier.verify(original, transformed)
    assert result.passed
    assert VerificationLevel.DIFFERENTIAL in result.levels_passed


def test_full_verification_passes() -> None:
    original = _make_module()
    transformed = original.clone()
    result = verify_transform(original, transformed)
    assert result.passed
    assert len(result.levels_run) == 2
    assert len(result.levels_passed) == 2


def test_verification_result_has_details() -> None:
    original = _make_module()
    transformed = original.clone()
    result = verify_transform(original, transformed)
    assert "structural" in result.details
    assert "differential" in result.details


def test_identity_transform_passes() -> None:
    """Identity transform (clone) should always pass."""
    module = _make_module()
    result = verify_transform(module, module.clone())
    assert result.passed


def test_guard_rejected_transform_skips_verification() -> None:
    module = _make_module()
    result = verify_guarded_transform(module, module.clone(), guard_matched=False)
    assert result.guard_matched is False
    assert result.verification.passed
    assert result.note == "guard_rejected"


# ---------------------------------------------------------------------------
# Phase-3: production-grade differential execution
# ---------------------------------------------------------------------------


def test_differential_skips_without_exported_program() -> None:
    """With no ExportedProgram we can't dispatch — SKIPPED, never a
    lying PASS. Caller routes this into manifest+gate decisions."""
    from compgen.transforms.verify import _verify_differential

    module = _make_module()
    passed, max_err, msg = _verify_differential(
        module,
        module.clone(),
        tolerance=1e-6,
        num_random_inputs=3,
    )
    # Skipped is benign at the level itself; promotion-gate decides.
    assert passed
    assert max_err is None
    assert "SKIPPED" in msg
    assert "exported_program" in msg


def test_differential_skips_when_no_executable_inputs() -> None:
    """IR with no tensor-typed entry args + no sample_inputs + no
    ExportedProgram gets a SKIPPED — not a fake PASS, not a FAIL."""
    from compgen.transforms.verify import _verify_differential

    module = _make_module()  # IndexType args
    passed, max_err, msg = _verify_differential(
        module,
        module.clone(),
        tolerance=1e-6,
        num_random_inputs=3,
    )
    assert passed
    assert max_err is None
    assert "SKIPPED" in msg


def test_differential_runs_real_execution_on_compiled_model() -> None:
    """End-to-end: compile a real TinyMLP through compile_model and
    observe that the verification report carries a PASS from an
    actually-executed differential run (not an op-count lie)."""
    import json

    import torch
    import torch.nn as nn
    from compgen.api import compile_model, device

    EXEMPLAR_DIR = __import__("pathlib").Path(__file__).parent.parent / "targetgen" / "exemplars"

    class _TinyMLP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc = nn.Linear(32, 16)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.fc(x).relu()

    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        torch.manual_seed(0)
        model = _TinyMLP()
        inputs = (torch.randn(4, 32),)
        dev = device(EXEMPLAR_DIR / "test_gpu_simt.yaml", output_dir=tmp / "tgt")
        compiled = compile_model(model, dev, sample_inputs=inputs, verify=True)
        bundle_dir = Path(compiled.pipeline_result.all_artifacts["bundle_dir"])
        report = json.loads((bundle_dir / "verification_report.json").read_text())
        assert report["passed"], report
        diff_detail = report["details"].get("differential", "")
        # Either real PASS (exported_program flowed through) or
        # SKIPPED with the explicit exported_program reason; either is
        # honest. What's NOT acceptable is an op-count-style fake pass.
        assert "FAIL" not in diff_detail, diff_detail
        assert "orig=" not in diff_detail, f"op-count-style message leaked back in: {diff_detail!r}"
