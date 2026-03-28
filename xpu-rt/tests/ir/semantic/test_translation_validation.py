"""Tests for translation validation."""

from __future__ import annotations

from compgen.ir.semantic.translation_validation import TranslationValidationResult
from xdsl.dialects import arith, func
from xdsl.dialects.builtin import IndexType, ModuleOp
from xdsl.ir import Block, Region


def test_result_construction() -> None:
    r = TranslationValidationResult(valid=True, status="valid")
    assert r.valid is True


def test_validate_identity_lowering() -> None:
    """Identity lowering (no change) should validate as correct."""
    from compgen.ir.semantic.translation_validation import validate_translation

    idx = IndexType()
    block = Block(arg_types=[idx, idx])
    a, b = block.args
    add = arith.AddiOp(a, b)
    block.add_op(add)
    block.add_op(func.ReturnOp(add.result))
    module = ModuleOp([func.FuncOp("test", ([idx, idx], [idx]), Region([block]))])

    result = validate_translation(module, module.clone())
    assert result.valid is True
    assert result.status == "valid"
