"""Tests for Recipe IR Family F: Provenance/Feedback operations.

Covers FromAgentOp, FromEqsatOp, FromTemplateOp, FeedbackOp,
RejectOp, PromoteOp, LineageOp.
"""

from __future__ import annotations

import io

from compgen.ir.recipe.attrs import CostAttr
from compgen.ir.recipe.ops_provenance import (
    FeedbackOp,
    FromAgentOp,
    FromEqsatOp,
    FromTemplateOp,
    LineageOp,
    PromoteOp,
    RejectOp,
)
from xdsl.dialects.builtin import (
    ArrayAttr,
    IntegerAttr,
    IntegerType,
    StringAttr,
    SymbolRefAttr,
)
from xdsl.printer import Printer


def _i64(val: int) -> IntegerAttr:
    return IntegerAttr(val, IntegerType(64))


def _print_op(op) -> str:
    buf = io.StringIO()
    Printer(stream=buf).print_op(op)
    return buf.getvalue()


# -- FromAgentOp --------------------------------------------------------------


def test_from_agent_minimal() -> None:
    op = FromAgentOp.build(properties={
        "agent_id": StringAttr("gemini-2.5-pro"),
        "iteration": _i64(3),
    })
    assert op.agent_id.data == "gemini-2.5-pro"
    assert op.iteration.value.data == 3
    assert op.reasoning is None


def test_from_agent_with_reasoning() -> None:
    op = FromAgentOp.build(properties={
        "agent_id": StringAttr("agent-1"),
        "iteration": _i64(0),
        "reasoning": StringAttr("matmul is memory bound, try tiling"),
    })
    assert op.reasoning.data == "matmul is memory bound, try tiling"


def test_from_agent_name() -> None:
    assert FromAgentOp.name == "recipe.from_agent"


def test_from_agent_verify_ok() -> None:
    op = FromAgentOp.build(properties={
        "agent_id": StringAttr("a"),
        "iteration": _i64(0),
    })
    op.verify()


# -- FromEqsatOp --------------------------------------------------------------


def test_from_eqsat_minimal() -> None:
    op = FromEqsatOp.build(properties={
        "rule_name": StringAttr("arith_simplify"),
    })
    assert op.rule_name.data == "arith_simplify"
    assert op.eclass_count is None


def test_from_eqsat_with_eclass() -> None:
    op = FromEqsatOp.build(properties={
        "rule_name": StringAttr("distributivity"),
        "eclass_count": _i64(42),
    })
    assert op.eclass_count.value.data == 42


def test_from_eqsat_name() -> None:
    assert FromEqsatOp.name == "recipe.from_eqsat"


# -- FromTemplateOp -----------------------------------------------------------


def test_from_template_minimal() -> None:
    op = FromTemplateOp.build(properties={
        "template_name": StringAttr("matmul_basic"),
    })
    assert op.template_name.data == "matmul_basic"
    assert op.template_version is None


def test_from_template_with_version() -> None:
    op = FromTemplateOp.build(properties={
        "template_name": StringAttr("conv2d_nhwc"),
        "template_version": _i64(2),
    })
    assert op.template_version.value.data == 2


def test_from_template_name() -> None:
    assert FromTemplateOp.name == "recipe.from_template"


# -- FeedbackOp ---------------------------------------------------------------


def test_feedback_minimal() -> None:
    op = FeedbackOp.build(properties={
        "candidate_ref": SymbolRefAttr("c0"),
        "outcome": StringAttr("passed"),
    })
    assert op.outcome.data == "passed"
    assert op.measured_cost is None
    assert op.message is None


def test_feedback_full() -> None:
    cost = CostAttr(120, "measured")
    op = FeedbackOp.build(properties={
        "candidate_ref": SymbolRefAttr("c0"),
        "outcome": StringAttr("failed"),
        "measured_cost": cost,
        "message": StringAttr("numerical divergence at output 0"),
    })
    assert op.measured_cost.value_us.value.data == 120
    assert op.message.data == "numerical divergence at output 0"


def test_feedback_name() -> None:
    assert FeedbackOp.name == "recipe.feedback"


# -- RejectOp -----------------------------------------------------------------


def test_reject_minimal() -> None:
    op = RejectOp.build(properties={
        "candidate_ref": SymbolRefAttr("c0"),
        "reason": StringAttr("verification failed"),
    })
    assert op.reason.data == "verification failed"
    assert op.feedback_ref is None


def test_reject_with_feedback_ref() -> None:
    op = RejectOp.build(properties={
        "candidate_ref": SymbolRefAttr("c0"),
        "reason": StringAttr("timeout"),
        "feedback_ref": SymbolRefAttr("fb0"),
    })
    assert op.feedback_ref is not None


def test_reject_name() -> None:
    assert RejectOp.name == "recipe.reject"


# -- PromoteOp ----------------------------------------------------------------


def test_promote_build() -> None:
    op = PromoteOp.build(properties={
        "candidate_ref": SymbolRefAttr("c0"),
        "recipe_key": StringAttr("matmul_f32_gpu0"),
        "version": _i64(1),
    })
    assert op.recipe_key.data == "matmul_f32_gpu0"
    assert op.version.value.data == 1


def test_promote_name() -> None:
    assert PromoteOp.name == "recipe.promote"


# -- LineageOp ----------------------------------------------------------------


def test_lineage_build() -> None:
    op = LineageOp.build(properties={
        "candidate_ref": SymbolRefAttr("c2"),
        "parent_refs": ArrayAttr([SymbolRefAttr("c0"), SymbolRefAttr("c1")]),
        "generation": _i64(3),
    })
    assert len(op.parent_refs.data) == 2
    assert op.generation.value.data == 3


def test_lineage_name() -> None:
    assert LineageOp.name == "recipe.lineage"


def test_lineage_printable() -> None:
    op = LineageOp.build(properties={
        "candidate_ref": SymbolRefAttr("c0"),
        "parent_refs": ArrayAttr([]),
        "generation": _i64(0),
    })
    text = _print_op(op)
    assert "recipe.lineage" in text
