"""Pass 7: Introduce prepacking for constant operands.

For each PrepackCandidate identified by the PrepackPlanner, inserts
a ``PackOp`` with ``is_prepack=1`` on the constant operand. Prepacking
happens once at initialization, not per-inference.
"""

from __future__ import annotations

from typing import Any

import structlog
from xdsl.dialects.builtin import (
    IntegerAttr,
    IntegerType,
    ModuleOp,
    StringAttr,
    SymbolRefAttr,
)
from xdsl.ir import Block

from compgen.ir.layout.attrs import PackSpecAttr
from compgen.ir.layout.ops import PackOp

log = structlog.get_logger()

PREPACK_MARKER_ATTR = "compgen.prepack_applied"


def introduce_prepacking(
    module: ModuleOp,
    prepack_candidates: list[Any] | None = None,
) -> ModuleOp:
    """Insert PackOp for constant operands.

    Args:
        module: The xDSL module to transform.
        prepack_candidates: List of PrepackCandidate from analysis.
            Each must have region_id, operand_name, operand_index attributes.
    """
    from xdsl.dialects.func import FuncOp, ReturnOp

    if not prepack_candidates:
        log.debug("layout.introduce_prepacking", msg="no candidates, skipping")
        return module

    # Build lookup set of operand names to prepack
    prepack_names: set[str] = set()
    for candidate in prepack_candidates:
        if hasattr(candidate, "operand_name"):
            prepack_names.add(candidate.operand_name)

    prepacked = 0
    counter = 0

    for op in list(module.walk()):
        if isinstance(op, (ModuleOp, FuncOp, ReturnOp)):
            continue
        if PREPACK_MARKER_ATTR in op.attributes:
            continue

        # Check if this op has a prepack hint
        hint_attr = op.attributes.get("compgen.prepack_hint")
        has_hint = hint_attr and hasattr(hint_attr, "data")

        # Check if any operand matches a prepack candidate
        should_prepack = False
        if has_hint:
            should_prepack = True
        else:
            for operand in op.operands:
                if hasattr(operand, "owner") and hasattr(operand.owner, "name"):
                    # Check if producer is a placeholder (constant weight)
                    if operand.owner.name in prepack_names:
                        should_prepack = True
                        break

        if not should_prepack:
            continue

        # Insert PackOp before this op
        ref_name = f"__prepack_{counter}"
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
                "is_prepack": IntegerAttr(1, IntegerType(1)),
            },
        )

        parent = op.parent
        if isinstance(parent, Block):
            parent.insert_op_before(pack_op, op)
            op.attributes[PREPACK_MARKER_ATTR] = StringAttr("1")
            prepacked += 1

    log.debug("layout.introduce_prepacking", prepacked=prepacked)
    return module


__all__ = ["introduce_prepacking"]
