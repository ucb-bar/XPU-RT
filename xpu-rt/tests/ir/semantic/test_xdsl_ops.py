"""Tests for Semantic IR xDSL operations."""

from __future__ import annotations

import io

from compgen.ir.semantic.ops import (
    RefinementOp,
    Semantic,
    SemanticInvariantOp,
    SemanticPredicateOp,
)
from xdsl.context import Context
from xdsl.dialects import builtin as builtin_dialect
from xdsl.dialects.builtin import (
    ArrayAttr,
    IntegerAttr,
    IntegerType,
    ModuleOp,
    StringAttr,
    SymbolRefAttr,
)
from xdsl.ir import Block, Region
from xdsl.parser import Parser
from xdsl.printer import Printer


def test_semantic_dialect_name() -> None:
    assert Semantic.name == "semantic"


def test_semantic_dialect_op_count() -> None:
    ops = list(Semantic.operations)
    assert len(ops) == 3


def test_predicate_build() -> None:
    op = SemanticPredicateOp.build(properties={
        "pred_name": StringAttr("eq"),
        "operand_names": ArrayAttr([StringAttr("x"), StringAttr("y")]),
        "semantic_type": StringAttr("bitvector"),
        "bit_width": IntegerAttr(32, IntegerType(64)),
    })
    assert op.pred_name.data == "eq"
    assert op.bit_width.value.data == 32


def test_predicate_without_bit_width() -> None:
    op = SemanticPredicateOp.build(properties={
        "pred_name": StringAttr("no_overflow"),
        "operand_names": ArrayAttr([StringAttr("a")]),
        "semantic_type": StringAttr("integer"),
    })
    assert op.bit_width is None


def test_refinement_build() -> None:
    op = RefinementOp.build(properties={
        "source_ref": SymbolRefAttr("source_matmul"),
        "target_ref": SymbolRefAttr("target_matmul"),
        "conditions": ArrayAttr([StringAttr("no_nan"), StringAttr("no_inf")]),
    })
    assert op.source_ref.root_reference.data == "source_matmul"
    assert len(op.conditions.data) == 2


def test_refinement_without_conditions() -> None:
    op = RefinementOp.build(properties={
        "source_ref": SymbolRefAttr("src"),
        "target_ref": SymbolRefAttr("tgt"),
    })
    assert op.conditions is None


def test_invariant_build() -> None:
    op = SemanticInvariantOp.build(properties={
        "region_ref": SymbolRefAttr("loop_0"),
        "predicate_name": StringAttr("monotonic_increase"),
        "operand_names": ArrayAttr([StringAttr("iter_var")]),
    })
    assert op.predicate_name.data == "monotonic_increase"


def test_semantic_round_trip() -> None:
    """Print → parse → print round-trip."""
    pred = SemanticPredicateOp.build(properties={
        "pred_name": StringAttr("slt"),
        "operand_names": ArrayAttr([StringAttr("a"), StringAttr("b")]),
        "semantic_type": StringAttr("integer"),
    })
    ref = RefinementOp.build(properties={
        "source_ref": SymbolRefAttr("src"),
        "target_ref": SymbolRefAttr("tgt"),
    })

    block = Block()
    block.add_op(pred)
    block.add_op(ref)
    module = ModuleOp(Region(block))

    buf = io.StringIO()
    Printer(stream=buf).print_op(module)
    text = buf.getvalue()

    ctx = Context()
    ctx.register_dialect("semantic", lambda: Semantic)
    ctx.register_dialect("builtin", lambda: builtin_dialect.Builtin)
    parsed = Parser(ctx, text).parse_module()

    buf2 = io.StringIO()
    Printer(stream=buf2).print_op(parsed)
    assert buf2.getvalue() == text
