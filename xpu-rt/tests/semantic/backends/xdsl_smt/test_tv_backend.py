"""Tests for the translation validation backend."""

from __future__ import annotations

import pytest
from xdsl.builder import ImplicitBuilder
from xdsl.context import Context as MLContext
from xdsl.dialects.arith import AddiOp, ConstantOp
from xdsl.dialects.builtin import Builtin, IntegerAttr, IntegerType, ModuleOp
from xdsl.dialects.func import Func, FuncOp, ReturnOp
from xdsl.ir import Block, Region

from compgen.semantic.backends.xdsl_smt.tv_backend import (
    ArithZ3Lowerer,
    TranslationValidationBackend,
)

z3 = pytest.importorskip("z3")


def _make_addi_func(const_val: int, name: str = "test") -> ModuleOp:
    """Create a module with: func @test(%arg0: i32) -> i32 { return %arg0 + const_val }."""
    i32 = IntegerType(32)

    block = Block(arg_types=[i32])
    with ImplicitBuilder(block) as (arg0,):
        c = ConstantOp(IntegerAttr(const_val, i32))
        add = AddiOp(arg0, c.result)
        ReturnOp(add.result)

    region = Region([block])
    func = FuncOp.build(
        properties={"sym_name": name, "function_type": (i32,), "res_attrs": None},
        regions=[region],
    )

    return ModuleOp([func])


class TestTranslationValidationBackend:
    """Test TV backend with simple arith programs."""

    def test_identical_programs_valid(self) -> None:
        """Same program refines itself."""
        mod = _make_addi_func(7)
        backend = TranslationValidationBackend(timeout_ms=5000)
        result = backend.check_refinement(mod, mod.clone())
        assert result.ok
        assert result.status == "valid"

    def test_constant_folding_valid(self) -> None:
        """Adding 3 then 4 is equivalent to adding 7."""
        # Before: %r = addi(%arg0, 3); %r2 = addi(%r, 4); return %r2
        # After:  %r = addi(%arg0, 7); return %r
        i32 = IntegerType(32)

        # Before: arg0 + 3 + 4
        block_before = Block(arg_types=[i32])
        with ImplicitBuilder(block_before) as (arg0,):
            c3 = ConstantOp(IntegerAttr(3, i32))
            add1 = AddiOp(arg0, c3.result)
            c4 = ConstantOp(IntegerAttr(4, i32))
            add2 = AddiOp(add1.result, c4.result)
            ReturnOp(add2.result)

        func_before = FuncOp.build(
            properties={"sym_name": "test", "function_type": (i32,), "res_attrs": None},
            regions=[Region([block_before])],
        )
        mod_before = ModuleOp([func_before])

        # After: arg0 + 7
        mod_after = _make_addi_func(7)

        backend = TranslationValidationBackend(timeout_ms=5000)
        result = backend.check_refinement(mod_before, mod_after)
        assert result.ok
        assert result.status == "valid"

    def test_wrong_constant_invalid(self) -> None:
        """Adding 5 is NOT equivalent to adding 7."""
        mod_before = _make_addi_func(5)
        mod_after = _make_addi_func(7)

        backend = TranslationValidationBackend(timeout_ms=5000)
        result = backend.check_refinement(mod_before, mod_after)
        assert not result.ok
        assert result.status == "invalid"
        assert result.counterexample is not None

    def test_no_func_returns_unknown(self) -> None:
        """Module without func.func returns unknown."""
        mod = ModuleOp([])
        backend = TranslationValidationBackend(timeout_ms=5000)
        result = backend.check_refinement(mod, mod)
        assert not result.ok
        assert result.status == "unknown"


class TestArithZ3Lowerer:
    """Test the arith-to-Z3 lowering."""

    def test_lower_constant(self) -> None:
        """ConstantOp lowers to BitVecVal."""
        mod = _make_addi_func(42)
        func = list(mod.body.block.ops)[0]
        lowerer = ArithZ3Lowerer()
        inputs, outputs = lowerer.lower_func(func, prefix="t_")
        assert len(inputs) == 1
        assert len(outputs) == 1
