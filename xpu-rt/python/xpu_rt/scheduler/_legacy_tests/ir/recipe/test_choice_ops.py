"""Tests for Recipe IR Family D: Choice/Search operations.

Covers AlternativesOp (with region), RankOp, SearchBudgetOp, RequireEqsatOp,
RequireSolverOp, DeferChoiceOp, PromoteCandidateOp.
"""

from __future__ import annotations

import io

import pytest
from compgen.ir.recipe.ops_candidate import TileOp
from compgen.ir.recipe.ops_choice import (
    AlternativesOp,
    DeferChoiceOp,
    PromoteCandidateOp,
    RankOp,
    RequireEqsatOp,
    RequireSolverOp,
    SearchBudgetOp,
)
from xdsl.dialects.builtin import (
    ArrayAttr,
    IntegerAttr,
    IntegerType,
    StringAttr,
    SymbolRefAttr,
)
from xdsl.ir import Block, Region
from xdsl.printer import Printer
from xdsl.utils.exceptions import VerifyException


def _i64(val: int) -> IntegerAttr:
    return IntegerAttr(val, IntegerType(64))


def _print_op(op) -> str:
    buf = io.StringIO()
    Printer(stream=buf).print_op(op)
    return buf.getvalue()


# -- AlternativesOp -----------------------------------------------------------


def test_alternatives_with_empty_region() -> None:
    """AlternativesOp can hold an empty region."""
    alt = AlternativesOp.build(
        properties={"region_ref": SymbolRefAttr("seg0")},
        regions=[Region(Block())],
    )
    assert len(alt.body.blocks) == 1
    assert len(list(alt.body.block.ops)) == 0


def test_alternatives_with_candidates_in_region() -> None:
    """AlternativesOp region can hold candidate ops."""
    tile1 = TileOp.build(
        properties={
            "region_ref": SymbolRefAttr("seg0"),
            "tile_sizes": ArrayAttr([_i64(64)]),
        }
    )
    tile2 = TileOp.build(
        properties={
            "region_ref": SymbolRefAttr("seg0"),
            "tile_sizes": ArrayAttr([_i64(128)]),
        }
    )
    alt = AlternativesOp.build(
        properties={"region_ref": SymbolRefAttr("seg0")},
        regions=[Region(Block([tile1, tile2]))],
    )
    ops = list(alt.body.block.ops)
    assert len(ops) == 2
    assert all(isinstance(op, TileOp) for op in ops)


def test_alternatives_name() -> None:
    assert AlternativesOp.name == "recipe.alternatives"


def test_alternatives_printable() -> None:
    alt = AlternativesOp.build(
        properties={"region_ref": SymbolRefAttr("seg0")},
        regions=[Region(Block())],
    )
    text = _print_op(alt)
    assert "recipe.alternatives" in text


# -- RankOp -------------------------------------------------------------------


def test_rank_build_minimal() -> None:
    op = RankOp.build(
        properties={
            "candidate_ref": SymbolRefAttr("c0"),
            "priority": _i64(1),
        }
    )
    assert op.priority.value.data == 1
    assert op.score is None


def test_rank_build_with_score() -> None:
    op = RankOp.build(
        properties={
            "candidate_ref": SymbolRefAttr("c0"),
            "priority": _i64(1),
            "score": _i64(950),
        }
    )
    assert op.score.value.data == 950


def test_rank_name() -> None:
    assert RankOp.name == "recipe.rank"


# -- SearchBudgetOp -----------------------------------------------------------


def test_search_budget_minimal() -> None:
    op = SearchBudgetOp.build(
        properties={
            "max_iterations": _i64(100),
        }
    )
    assert op.max_iterations.value.data == 100
    assert op.timeout_ms is None


def test_search_budget_with_timeout() -> None:
    op = SearchBudgetOp.build(
        properties={
            "max_iterations": _i64(50),
            "timeout_ms": _i64(30000),
        }
    )
    assert op.timeout_ms.value.data == 30000


def test_search_budget_verify_ok() -> None:
    op = SearchBudgetOp.build(
        properties={
            "max_iterations": _i64(10),
        }
    )
    op.verify()


def test_search_budget_verify_zero_iterations_fails() -> None:
    op = SearchBudgetOp.build(
        properties={
            "max_iterations": _i64(0),
        }
    )
    with pytest.raises(VerifyException, match="max_iterations must be positive"):
        op.verify()


def test_search_budget_verify_negative_timeout_fails() -> None:
    op = SearchBudgetOp.build(
        properties={
            "max_iterations": _i64(10),
            "timeout_ms": _i64(-1),
        }
    )
    with pytest.raises(VerifyException, match="timeout_ms must be positive"):
        op.verify()


# -- RequireEqsatOp -----------------------------------------------------------


def test_require_eqsat_minimal() -> None:
    op = RequireEqsatOp.build(
        properties={
            "region_ref": SymbolRefAttr("r0"),
        }
    )
    assert op.rule_categories is None
    assert op.max_iterations is None


def test_require_eqsat_with_categories() -> None:
    op = RequireEqsatOp.build(
        properties={
            "region_ref": SymbolRefAttr("r0"),
            "rule_categories": ArrayAttr([StringAttr("arith"), StringAttr("linalg")]),
            "max_iterations": _i64(20),
        }
    )
    assert len(op.rule_categories.data) == 2


# -- RequireSolverOp ----------------------------------------------------------


def test_require_solver_valid_types() -> None:
    for solve_type in ("placement", "schedule", "memory"):
        op = RequireSolverOp.build(
            properties={
                "solve_type": StringAttr(solve_type),
            }
        )
        op.verify()


def test_require_solver_invalid_type_fails() -> None:
    op = RequireSolverOp.build(
        properties={
            "solve_type": StringAttr("bogus"),
        }
    )
    with pytest.raises(VerifyException, match="Invalid solve_type"):
        op.verify()


def test_require_solver_name() -> None:
    assert RequireSolverOp.name == "recipe.require_solver"


# -- DeferChoiceOp ------------------------------------------------------------


def test_defer_choice_minimal() -> None:
    op = DeferChoiceOp.build(
        properties={
            "region_ref": SymbolRefAttr("r0"),
        }
    )
    assert op.reason is None


def test_defer_choice_with_reason() -> None:
    op = DeferChoiceOp.build(
        properties={
            "region_ref": SymbolRefAttr("r0"),
            "reason": StringAttr("needs profiling data"),
        }
    )
    assert op.reason.data == "needs profiling data"


# -- PromoteCandidateOp -------------------------------------------------------


def test_promote_candidate_build() -> None:
    result = PromoteCandidateOp.build(
        properties={
            "candidate_ref": SymbolRefAttr("c0"),
            "from_alternatives": SymbolRefAttr("alt0"),
        }
    )
    assert result.name == "recipe.promote_candidate"
