"""Semantic IR xDSL operations.

Proper xDSL operations for the semantic layer (Layer 3). These encode
precise mathematical semantics for verification tooling:
    - Predicates (boolean-valued assertions)
    - Refinement relations (translation validation)
    - Invariants (loop/region invariants)

These ops are used by the verification pipeline when Recipe IR
``RequireTranslationValidationOp`` obligations are lowered.
"""

from __future__ import annotations

from xdsl.dialects.builtin import ArrayAttr, IntegerAttr, StringAttr, SymbolRefAttr
from xdsl.ir import Dialect
from xdsl.irdl import (
    IRDLOperation,
    irdl_op_definition,
    opt_prop_def,
    prop_def,
    traits_def,
)
from xdsl.traits import Pure


@irdl_op_definition
class SemanticPredicateOp(IRDLOperation):
    """A semantic predicate (boolean-valued assertion).

    Predicates: "eq", "ne", "slt", "sle", "sgt", "sge",
                "ult", "ule", "ugt", "uge", "no_overflow".
    """

    name = "semantic.predicate"

    pred_name = prop_def(StringAttr)  # predicate kind
    operand_names = prop_def(ArrayAttr)  # ArrayAttr of StringAttr
    semantic_type = prop_def(StringAttr)  # "bitvector", "integer", "boolean", "real"
    bit_width = opt_prop_def(IntegerAttr)

    traits = traits_def(Pure())


@irdl_op_definition
class RefinementOp(IRDLOperation):
    """Translation validation: target refines source.

    Every behavior of the target is a legal behavior of the source.
    Conditions constrain when refinement holds.
    """

    name = "semantic.refinement"

    source_ref = prop_def(SymbolRefAttr)
    target_ref = prop_def(SymbolRefAttr)
    conditions = opt_prop_def(ArrayAttr)  # ArrayAttr of StringAttr

    traits = traits_def(Pure())


@irdl_op_definition
class SemanticInvariantOp(IRDLOperation):
    """Loop or region invariant for verification."""

    name = "semantic.invariant"

    region_ref = prop_def(SymbolRefAttr)
    predicate_name = prop_def(StringAttr)
    operand_names = opt_prop_def(ArrayAttr)

    traits = traits_def(Pure())


Semantic = Dialect(
    "semantic",
    [SemanticPredicateOp, RefinementOp, SemanticInvariantOp],
    [],
)
"""The Semantic IR dialect for verification."""


__all__ = [
    "RefinementOp",
    "Semantic",
    "SemanticInvariantOp",
    "SemanticPredicateOp",
]
