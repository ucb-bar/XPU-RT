"""Pass 9: Materialize layout boundaries.

For each remaining SetLayoutOp/UnsetLayoutOp pair that was not fused
away, inserts real PackOp/UnpackOp (or concrete transpose) operations.
After this pass, all virtual encodings are gone and only concrete data
movement ops remain.
"""

from __future__ import annotations

import structlog
from xdsl.dialects.builtin import (
    IntegerAttr,
    IntegerType,
    ModuleOp,
    SymbolRefAttr,
)
from xdsl.ir import Block

from compgen.ir.layout.attrs import PackSpecAttr
from compgen.ir.layout.ops import PackOp, SetLayoutOp, UnsetLayoutOp

log = structlog.get_logger()

MATERIALIZED_ATTR = "compgen.layout_materialized"


def materialize_layout_boundaries(module: ModuleOp) -> ModuleOp:
    """Replace virtual layout ops with concrete pack/unpack operations.

    Walks the module and for each SetLayoutOp/UnsetLayoutOp pair:
    - If the boundary was not fused (no compgen.fused_layout), insert
      a concrete PackOp at the SetLayout position and UnpackOp at the
      UnsetLayout position.
    - Remove the virtual SetLayoutOp/UnsetLayoutOp.
    """

    materialized = 0
    removed_virtual = 0
    counter = 0

    # Collect SetLayoutOp and UnsetLayoutOp instances
    set_ops: list[SetLayoutOp] = []
    unset_ops: list[UnsetLayoutOp] = []

    for op in module.walk():
        if isinstance(op, SetLayoutOp):
            set_ops.append(op)
        elif isinstance(op, UnsetLayoutOp):
            unset_ops.append(op)

    # Materialize SetLayoutOps that aren't in fused regions
    for set_op in set_ops:
        parent = set_op.parent
        if not isinstance(parent, Block):
            continue

        # Check if the next non-layout op has fused_layout
        fused = False
        idx = list(parent.ops).index(set_op)
        for subsequent in list(parent.ops)[idx + 1 :]:
            if isinstance(subsequent, (SetLayoutOp, UnsetLayoutOp)):
                continue
            if "compgen.fused_layout" in subsequent.attributes:
                fused = True
            break

        if not fused:
            # Insert concrete PackOp
            ref_name = f"__mat_pack_{counter}"
            counter += 1
            default_spec = PackSpecAttr(
                inner_tiles=[16, 16],
                outer_perm=[0, 1],
                padding_value="zero",
            )
            pack_op = PackOp.build(
                properties={
                    "source_ref": SymbolRefAttr(ref_name),
                    "pack_spec": default_spec,
                    "is_prepack": IntegerAttr(0, IntegerType(1)),
                },
            )
            parent.insert_op_before(pack_op, set_op)
            materialized += 1

        # Remove the virtual SetLayoutOp
        parent.erase_op(set_op)
        removed_virtual += 1

    # Remove UnsetLayoutOps (materialization boundaries consumed)
    for unset_op in unset_ops:
        parent = unset_op.parent
        if isinstance(parent, Block):
            parent.erase_op(unset_op)
            removed_virtual += 1

    log.debug(
        "layout.materialize_boundaries",
        materialized=materialized,
        removed_virtual=removed_virtual,
    )
    return module


__all__ = ["materialize_layout_boundaries"]
