"""Recipe IR Family A: Scope/Anchor operations.

These identify stable payload regions that the rest of the recipe
references. Every region, segment, and anchor gets a stable symbol
so the LLM and downstream tools never need fragile text-matching.
"""

from __future__ import annotations

from xdsl.dialects.builtin import ArrayAttr, IntegerAttr, StringAttr, SymbolRefAttr
from xdsl.irdl import (
    IRDLOperation,
    irdl_op_definition,
    opt_prop_def,
    prop_def,
    traits_def,
)
from xdsl.traits import Pure, SymbolOpInterface

from compgen.ir.recipe.attrs import EffectClassAttr, ShapeSummaryAttr


@irdl_op_definition
class RecipeRegionOp(IRDLOperation):
    """Stable handle into a Payload IR region.

    Every significant payload region (matmul, fused block, etc.) gets a
    RecipeRegionOp with a unique symbol. Candidate actions, facts, and
    verification obligations reference this symbol.
    """

    name = "recipe.region"

    sym_name = prop_def(StringAttr)
    payload_region_id = prop_def(StringAttr)
    shape_summary = opt_prop_def(ShapeSummaryAttr)
    effect_class = opt_prop_def(EffectClassAttr)
    op_count = opt_prop_def(IntegerAttr)

    traits = traits_def(SymbolOpInterface())


@irdl_op_definition
class SegmentOp(IRDLOperation):
    """Group multiple regions into a scheduling/optimization unit.

    Segments are contiguous sets of regions that should be optimized
    together (following the Constable segmentation pattern).
    """

    name = "recipe.segment"

    sym_name = prop_def(StringAttr)
    region_refs = prop_def(ArrayAttr)  # ArrayAttr of SymbolRefAttr

    traits = traits_def(SymbolOpInterface())


@irdl_op_definition
class AnchorOp(IRDLOperation):
    """Stable symbol for referencing a specific payload op across recipe versions.

    Unlike RecipeRegionOp (which refers to a region/subgraph), AnchorOp
    refers to a single specific operation in the payload IR.
    """

    name = "recipe.anchor"

    sym_name = prop_def(StringAttr)
    payload_op_name = prop_def(StringAttr)

    traits = traits_def(SymbolOpInterface(), Pure())


@irdl_op_definition
class RecipeGuardOp(IRDLOperation):
    """Stable handle for a promoted synthesized guard artifact."""

    name = "recipe.guard"

    sym_name = prop_def(StringAttr)
    guard_key = prop_def(StringAttr)
    transform_family = prop_def(StringAttr)
    guard_kind = opt_prop_def(StringAttr)
    target_class = opt_prop_def(StringAttr)

    traits = traits_def(SymbolOpInterface(), Pure())


@irdl_op_definition
class BindPayloadOp(IRDLOperation):
    """Bind a recipe scope to a specific payload module.

    Establishes the connection between this recipe and the payload IR
    module it describes decisions for.
    """

    name = "recipe.bind_payload"

    region_ref = prop_def(SymbolRefAttr)
    payload_module_id = prop_def(StringAttr)

    traits = traits_def(Pure())


__all__ = [
    "AnchorOp",
    "BindPayloadOp",
    "RecipeGuardOp",
    "RecipeRegionOp",
    "SegmentOp",
]
