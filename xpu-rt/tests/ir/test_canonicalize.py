"""Tests for IR canonicalization."""

from __future__ import annotations

from xdsl.dialects.builtin import Float32Type, FunctionType, ModuleOp, TensorType
from xdsl.dialects.func import FuncOp, ReturnOp
from xdsl.dialects.tensor import EmptyOp
from xdsl.ir import Block, Region

from compgen.ir.payload.canonicalize import CanonicalizationReport, CanonicalizePass, canonicalize


def _make_trivial_module() -> ModuleOp:
    """Build a small valid xDSL module for testing."""
    f32 = Float32Type()
    tensor_type = TensorType(f32, [4, 4])
    func_type = FunctionType.from_lists([tensor_type], [tensor_type])

    block = Block(arg_types=[tensor_type])
    block.add_op(ReturnOp(block.args[0]))

    region = Region([block])
    func_op = FuncOp("test_fn", func_type, region)
    return ModuleOp([func_op])


def test_canonicalize_preserves_semantics() -> None:
    """Canonicalization should not change IR semantics."""
    module = _make_trivial_module()

    # Count ops before
    ops_before = sum(1 for _ in module.walk())

    result_module, report = canonicalize(module)

    # Count ops after -- module should be unchanged (MVP is identity)
    ops_after = sum(1 for _ in result_module.walk())
    assert ops_before == ops_after

    # The returned module should still verify
    result_module.verify()


def test_canonicalize_report() -> None:
    """Canonicalization should produce a report with op counts."""
    module = _make_trivial_module()

    canon_pass = CanonicalizePass()
    result_module, report = canon_pass.run(module)

    assert isinstance(report, CanonicalizationReport)
    assert report.ops_before > 0
    assert report.ops_after > 0
    # MVP applies no transforms
    assert report.transforms_applied == []
    assert report.warnings == []
    # ops_before == ops_after because MVP is identity
    assert report.ops_before == report.ops_after
