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
