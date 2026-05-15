"""Recipe IR Family E: Verification Obligation operations.

These encode what must be checked before a candidate is accepted.
Proof obligations are first-class, not comments.
"""

from __future__ import annotations

from xdsl.dialects.builtin import IntegerAttr, StringAttr, SymbolRefAttr
from xdsl.irdl import (
    IRDLOperation,
    irdl_op_definition,
    opt_prop_def,
    prop_def,
    traits_def,
)
from xdsl.traits import Pure

from compgen.ir.recipe.attrs import DeviceRefAttr


@irdl_op_definition
class RequireDiffTestOp(IRDLOperation):
    """Require differential testing for a region.

    Compares outputs before/after optimization within tolerance.
    """

    name = "recipe.require_diff_test"

    region_ref = prop_def(SymbolRefAttr)
    tolerance = opt_prop_def(IntegerAttr)  # tolerance in ULPs

    traits = traits_def(Pure())


@irdl_op_definition
class RequireTranslationValidationOp(IRDLOperation):
    """Require translation validation (SMT-backed) for a rewrite.

    Proves that the target program refines the source.
    """

    name = "recipe.require_translation_validation"

    region_ref = prop_def(SymbolRefAttr)
    source_op = opt_prop_def(StringAttr)
    target_op = opt_prop_def(StringAttr)

    traits = traits_def(Pure())


@irdl_op_definition
class RequireLayoutInvariantOp(IRDLOperation):
    """Require that a layout invariant holds after optimization."""

    name = "recipe.require_layout_invariant"

    region_ref = prop_def(SymbolRefAttr)
    expected_layout = prop_def(StringAttr)

    traits = traits_def(Pure())


@irdl_op_definition
class RequireMemoryBoundOp(IRDLOperation):
    """Require that memory usage stays within a bound."""

    name = "recipe.require_memory_bound"

    region_ref = prop_def(SymbolRefAttr)
    max_bytes = prop_def(IntegerAttr)
    device = opt_prop_def(DeviceRefAttr)

    traits = traits_def(Pure())


@irdl_op_definition
class RequireCheckFileOp(IRDLOperation):
    """Require FileCheck-style assertions from a CHECK file.

    Uses ``ir/checks.py:IRChecker`` for structural verification.
    """

    name = "recipe.require_check_file"

    check_file_path = prop_def(StringAttr)

    traits = traits_def(Pure())


@irdl_op_definition
class RequireProfileBudgetOp(IRDLOperation):
    """Require that runtime performance stays within a budget."""

    name = "recipe.require_profile_budget"

    region_ref = prop_def(SymbolRefAttr)
    max_latency_us = prop_def(IntegerAttr)
    device = opt_prop_def(DeviceRefAttr)

    traits = traits_def(Pure())


__all__ = [
    "RequireCheckFileOp",
    "RequireDiffTestOp",
    "RequireLayoutInvariantOp",
    "RequireMemoryBoundOp",
    "RequireProfileBudgetOp",
    "RequireTranslationValidationOp",
]
