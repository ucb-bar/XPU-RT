"""Tests for transforms/apply.py -- transform application."""

from __future__ import annotations

from compgen.transforms.apply import TransformApplicator, TransformDiagnostic, TransformedIR, apply_transforms
from compgen.transforms.synthesize import TransformScript
from xdsl.dialects.builtin import Float32Type, FunctionType, ModuleOp, TensorType
from xdsl.dialects.func import FuncOp, ReturnOp
from xdsl.ir import Block, Region


def test_transform_diagnostic_construction() -> None:
    d = TransformDiagnostic(
        transform_name="tile",
        level="warning",
        message="tile size not a power of 2",
    )
    assert d.transform_name == "tile"
    assert d.level == "warning"
    assert d.message == "tile size not a power of 2"
    assert d.op_name == ""


def test_transformed_ir_defaults() -> None:
    t = TransformedIR(module=None)
    assert t.module is None
    assert t.scripts_applied == []
    assert t.diagnostics == []


def _make_test_module() -> ModuleOp:
    """Build a minimal valid xDSL module."""
    f32 = Float32Type()
    tensor_type = TensorType(f32, [4, 4])
    func_type = FunctionType.from_lists([tensor_type], [tensor_type])

    block = Block(arg_types=[tensor_type])
    block.add_op(ReturnOp(block.args[0]))

    region = Region([block])
    func_op = FuncOp("forward", func_type, region)
    return ModuleOp([func_op])


def test_transform_applicator_apply() -> None:
    """TransformApplicator.apply should return a TransformedIR."""
    module = _make_test_module()
    applicator = TransformApplicator()

    # A valid no-op RewritePattern script
    noop_script = TransformScript(
        name="noop_pattern",
        content=(
            "from xdsl.pattern_rewriter import RewritePattern, PatternRewriter\n"
            "from xdsl.ir import Operation\n"
            "\n"
            "class NoopPattern(RewritePattern):\n"
            "    def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter) -> None:\n"
            "        return\n"
        ),
    )

    result = applicator.apply(module, [noop_script])

    assert isinstance(result, TransformedIR)
    assert result.module is not None
    # The noop pattern should have been found and applied successfully
    assert len(result.scripts_applied) == 1
    assert result.scripts_applied[0].name == "noop_pattern"
    # Should have an info-level diagnostic about successful application
    info_diags = [d for d in result.diagnostics if d.level == "info"]
    assert len(info_diags) > 0


def test_apply_transforms_convenience() -> None:
    """apply_transforms should work with default settings."""
    module = _make_test_module()

    # A script with a syntax error should produce an error diagnostic, not crash
    bad_script = TransformScript(
        name="bad_syntax",
        content="def this is not valid python {{{{",
    )

    result = apply_transforms(module, [bad_script])

    assert isinstance(result, TransformedIR)
    assert result.module is not None
    # The bad script should NOT be in scripts_applied
    assert len(result.scripts_applied) == 0
    # Should have an error diagnostic
    error_diags = [d for d in result.diagnostics if d.level == "error"]
    assert len(error_diags) == 1
    assert "Syntax error" in error_diags[0].message
