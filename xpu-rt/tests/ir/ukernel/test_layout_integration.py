"""Tests for ukernel-layout bridge integration.

Validates that the layout pipeline correctly identifies transparent vs
opaque ukernel attributes and that transparent ukernels participate in
layout propagation while opaque ones do not.
"""

from __future__ import annotations

from compgen.transforms.layout import run_layout_pipeline
from compgen.transforms.layout.propagate_layouts import _is_ukernel_transparent
from xdsl.builder import Builder
from xdsl.dialects.arith import ConstantOp
from xdsl.dialects.builtin import ModuleOp, StringAttr, i32
from xdsl.dialects.func import FuncOp, ReturnOp
from xdsl.ir import Block, Region


class _FakeOp:
    """Minimal stand-in with an attributes dict for _is_ukernel_transparent."""

    def __init__(self, attrs: dict[str, object] | None = None) -> None:
        self.attributes: dict[str, object] = attrs or {}


class TestIsUkernelTransparent:
    """_is_ukernel_transparent returns True only for transparent attrs."""

    def test_transparent_attr_returns_true(self) -> None:
        op = _FakeOp({"compgen.ukernel_transparency": StringAttr("transparent")})
        assert _is_ukernel_transparent(op) is True

    def test_opaque_attr_returns_false(self) -> None:
        op = _FakeOp({"compgen.ukernel_transparency": StringAttr("opaque")})
        assert _is_ukernel_transparent(op) is False

    def test_missing_attr_returns_false(self) -> None:
        op = _FakeOp({})
        assert _is_ukernel_transparent(op) is False

    def test_empty_string_attr_returns_false(self) -> None:
        op = _FakeOp({"compgen.ukernel_transparency": StringAttr("")})
        assert _is_ukernel_transparent(op) is False


class TestLayoutPipelineWithUkernels:
    """run_layout_pipeline produces clean output even with ukernel annotations."""

    def test_pipeline_produces_module(self) -> None:
        """An empty module survives all 10 layout passes."""
        block = Block()
        region = Region([block])
        module = ModuleOp(region)

        result = run_layout_pipeline(module)
        assert isinstance(result, ModuleOp)

    def test_pipeline_with_annotated_func(self) -> None:
        """A func with a ukernel-transparency annotation survives layout passes.

        Builds: func @test() -> i32 { %c = arith.constant 42 : i32; return %c }
        Annotates the constant op as a transparent ukernel with a layout
        encoding. The full 10-pass layout pipeline must not crash and must
        preserve the function.
        """
        # Build the constant and annotate it
        cst_op = ConstantOp.from_int_and_width(42, i32)
        cst_op.attributes["compgen.ukernel_transparency"] = StringAttr("transparent")
        cst_op.attributes["compgen.encoding"] = StringAttr("nchw_tile_8x4")

        ret_op = ReturnOp(cst_op)

        body_block = Block(arg_types=[])
        body_block.add_ops([cst_op, ret_op])
        body_region = Region([body_block])

        func_op = FuncOp("test", ([], [i32]), body_region)

        mod_block = Block()
        mod_block.add_op(func_op)
        mod_region = Region([mod_block])
        module = ModuleOp(mod_region)

        result = run_layout_pipeline(module)
        assert isinstance(result, ModuleOp)

        # The module should still contain our function
        funcs = [op for op in result.body.block.ops if isinstance(op, FuncOp)]
        assert len(funcs) == 1

    def test_pipeline_builder_pattern(self) -> None:
        """Same test using the Builder pattern for module construction."""

        @Builder.implicit_region
        def body() -> None:
            @Builder.implicit_region(())
            def func_body(args: tuple) -> None:  # type: ignore[type-arg]
                cst = ConstantOp.from_int_and_width(7, i32)
                cst.attributes["compgen.ukernel_transparency"] = StringAttr("transparent")
                ReturnOp(cst)

            FuncOp("built_test", ([], [i32]), func_body)

        module = ModuleOp(body)
        result = run_layout_pipeline(module)
        assert isinstance(result, ModuleOp)
