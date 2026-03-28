"""Bridge between agent Actions and Recipe IR ops.

Bidirectional conversion so the agent's existing Action-based flow
can accumulate a Recipe IR module alongside its normal operation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from xdsl.dialects.builtin import (
    ArrayAttr,
    IntegerAttr,
    IntegerType,
    StringAttr,
    SymbolRefAttr,
)
from xdsl.ir import Operation

from compgen.ir.recipe.attrs import DeviceRefAttr, ProvenanceAttr
from compgen.ir.recipe.ops_candidate import (
    FuseOp,
    InsertCopyBoundaryOp,
    PlaceOnDeviceOp,
    TileOp,
)
from compgen.ir.recipe.ops_choice import RequireEqsatOp

if TYPE_CHECKING:
    from compgen.agent.env import (
        Action,
    )


def _int(v: int) -> IntegerAttr:
    return IntegerAttr(v, IntegerType(64))


def _prov(iteration: int) -> ProvenanceAttr:
    return ProvenanceAttr("agent", iteration)


def _candidate_sym(prefix: str, region_id: str, iteration: int) -> StringAttr:
    return StringAttr(f"{prefix}_{region_id}_{iteration}")


def action_to_recipe_op(action: Action, iteration: int = 0) -> Operation | None:
    """Convert an agent Action to a Recipe IR operation.

    Returns None for actions that don't map to recipe ops
    (e.g., NoopAction, InspectAction, AnalyzeAction, BenchmarkAction).

    Args:
        action: Agent action to convert.
        iteration: Current step/iteration number for provenance.

    Returns:
        Recipe IR operation, or None.
    """
    from compgen.agent.env import (
        AssignDeviceAction,
        EqSatAction,
        FuseAction,
        InsertCopyAction,
        TileAction,
    )

    if isinstance(action, TileAction) and action.region_id:
        return TileOp.build(properties={
            "sym_name": _candidate_sym("cand_tile", action.region_id, iteration),
            "region_ref": SymbolRefAttr(action.region_id),
            "tile_sizes": ArrayAttr([_int(s) for s in action.tile_sizes]),
            "provenance": _prov(iteration),
        })

    if isinstance(action, FuseAction) and action.region_id:
        refs = [SymbolRefAttr(action.region_id)]
        if action.target_region_id:
            refs.append(SymbolRefAttr(action.target_region_id))
        return FuseOp.build(properties={
            "sym_name": _candidate_sym("cand_fuse", action.region_id, iteration),
            "fuse_regions": ArrayAttr(refs),
            "provenance": _prov(iteration),
        })

    if isinstance(action, AssignDeviceAction) and action.region_id:
        return PlaceOnDeviceOp.build(properties={
            "sym_name": _candidate_sym("cand_place", action.region_id, iteration),
            "region_ref": SymbolRefAttr(action.region_id),
            "device": DeviceRefAttr(action.device_index, "device"),
            "provenance": _prov(iteration),
        })

    if isinstance(action, InsertCopyAction) and action.region_id:
        return InsertCopyBoundaryOp.build(properties={
            "sym_name": _candidate_sym("cand_copy", action.region_id, iteration),
            "src_region": SymbolRefAttr(action.region_id),
            "dst_region": SymbolRefAttr(action.target_region_id or action.region_id),
            "tensor_name": StringAttr("data"),
            "is_async": _int(1 if action.async_ else 0),
            "provenance": _prov(iteration),
        })

    if isinstance(action, EqSatAction) and action.region_id:
        props: dict = {
            "region_ref": SymbolRefAttr(action.region_id),
        }
        if action.rule_categories:
            props["rule_categories"] = ArrayAttr(
                [StringAttr(c) for c in action.rule_categories],
            )
        return RequireEqsatOp.build(properties=props)

    # Verification actions → Recipe IR verification ops
    from compgen.agent.env import RequestVerificationAction
    from compgen.ir.recipe.ops_verify import (
        RequireDiffTestOp,
        RequireTranslationValidationOp,
    )

    if isinstance(action, RequestVerificationAction) and action.region_id:
        if action.level in ("translation_validation", "both"):
            return RequireTranslationValidationOp.build(properties={
                "region_ref": SymbolRefAttr(action.region_id),
            })
        if action.level == "differential":
            return RequireDiffTestOp.build(properties={
                "region_ref": SymbolRefAttr(action.region_id),
            })

    return None


def recipe_op_to_action(op: Operation) -> Action | None:
    """Convert a Recipe IR op back to an agent Action.

    Returns None for ops that don't map to agent actions
    (facts, provenance, verification obligations, etc.).
    """
    from compgen.agent.env import (
        AssignDeviceAction,
        FuseAction,
        InsertCopyAction,
        TileAction,
    )

    if isinstance(op, TileOp):
        sizes = tuple(
            a.value.data for a in op.tile_sizes.data
            if isinstance(a, IntegerAttr)
        )
        return TileAction(
            region_id=op.region_ref.root_reference.data,
            tile_sizes=sizes,
        )

    if isinstance(op, PlaceOnDeviceOp):
        return AssignDeviceAction(
            region_id=op.region_ref.root_reference.data,
            device_index=op.device.index.value.data,
        )

    if isinstance(op, FuseOp):
        refs = [
            r.root_reference.data
            for r in op.fuse_regions.data
            if isinstance(r, SymbolRefAttr)
        ]
        return FuseAction(
            region_id=refs[0] if refs else "",
            target_region_id=refs[1] if len(refs) > 1 else "",
        )

    if isinstance(op, InsertCopyBoundaryOp):
        return InsertCopyAction(
            region_id=op.src_region.root_reference.data,
            target_region_id=op.dst_region.root_reference.data,
        )

    return None


__all__ = ["action_to_recipe_op", "recipe_op_to_action"]
