"""Recipe IR Family D: Choice/Search operations.

These express alternatives and search structure. AlternativesOp is the
only op with a region body -- it contains candidate sub-recipes.
"""

from __future__ import annotations

from xdsl.dialects.builtin import ArrayAttr, IntegerAttr, StringAttr, SymbolRefAttr
from xdsl.irdl import (
    IRDLOperation,
    irdl_op_definition,
    opt_prop_def,
    prop_def,
    region_def,
    traits_def,
)
from xdsl.traits import NoTerminator, Pure
from xdsl.utils.exceptions import VerifyException


@irdl_op_definition
class AlternativesOp(IRDLOperation):
    """Multiple legal alternatives for a region.

    The body region contains candidate ops (actions, facts, ranks)
    representing different legal optimization paths.
    """

    name = "recipe.alternatives"

    region_ref = prop_def(SymbolRefAttr)
    body = region_def()

    traits = traits_def(NoTerminator())


@irdl_op_definition
class RankOp(IRDLOperation):
    """Priority/score ranking for a candidate."""

    name = "recipe.rank"

    candidate_ref = prop_def(SymbolRefAttr)
    priority = prop_def(IntegerAttr)
    score = opt_prop_def(IntegerAttr)  # e.g., cost model score in milliunits

    traits = traits_def(Pure())


@irdl_op_definition
class SearchBudgetOp(IRDLOperation):
    """Search budget constraints for optimization."""

    name = "recipe.search_budget"

    max_iterations = prop_def(IntegerAttr)
    timeout_ms = opt_prop_def(IntegerAttr)

    traits = traits_def(Pure())

    def verify_(self) -> None:
        if self.max_iterations.value.data <= 0:
            raise VerifyException(f"max_iterations must be positive, got {self.max_iterations.value.data}")
        if self.timeout_ms is not None and self.timeout_ms.value.data <= 0:
            raise VerifyException(f"timeout_ms must be positive, got {self.timeout_ms.value.data}")


@irdl_op_definition
class RequireEqsatOp(IRDLOperation):
    """Request equality saturation for a region.

    When lowered, triggers ``eqsat/pipeline.py:run_eqsat_pass()``
    with the specified rule categories.
    """

    name = "recipe.require_eqsat"

    region_ref = prop_def(SymbolRefAttr)
    rule_categories = opt_prop_def(ArrayAttr)  # ArrayAttr of StringAttr
    max_iterations = opt_prop_def(IntegerAttr)

    traits = traits_def(Pure())


@irdl_op_definition
class RequireSolverOp(IRDLOperation):
    """Request a solver-backed optimization.

    Solve types: "placement", "schedule", "memory".
    When lowered, dispatches to the appropriate solve backend.
    """

    name = "recipe.require_solver"

    solve_type = prop_def(StringAttr)
    timeout_ms = opt_prop_def(IntegerAttr)

    traits = traits_def(Pure())

    def verify_(self) -> None:
        valid = {"placement", "schedule", "memory"}
        if self.solve_type.data not in valid:
            raise VerifyException(f"Invalid solve_type '{self.solve_type.data}', expected one of {valid}")


@irdl_op_definition
class DeferChoiceOp(IRDLOperation):
    """Defer a decision for a region to a later stage or agent."""

    name = "recipe.defer_choice"

    region_ref = prop_def(SymbolRefAttr)
    reason = opt_prop_def(StringAttr)

    traits = traits_def(Pure())


@irdl_op_definition
class PromoteCandidateOp(IRDLOperation):
    """Promote a specific candidate from an alternatives set."""

    name = "recipe.promote_candidate"

    candidate_ref = prop_def(SymbolRefAttr)
    from_alternatives = prop_def(SymbolRefAttr)

    traits = traits_def(Pure())


__all__ = [
    "AlternativesOp",
    "DeferChoiceOp",
    "PromoteCandidateOp",
    "RankOp",
    "RequireEqsatOp",
    "RequireSolverOp",
    "SearchBudgetOp",
]
