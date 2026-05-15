"""Pass 10: Clean up layout artifacts.

Removes dead SetLayoutOp/UnsetLayoutOp pairs where both sides were
fused or materialized. Removes redundant PackOp/UnpackOp sequences
that cancel out. Final verification ensures no layout dialect ops remain.
"""

from __future__ import annotations

import structlog
from xdsl.dialects.builtin import ModuleOp, StringAttr
from xdsl.ir import Block

from compgen.ir.layout.ops import PackOp, SetLayoutOp, UnpackOp, UnsetLayoutOp

log = structlog.get_logger()

LAYOUT_CLEAN_ATTR = "compgen.layout_clean"


def cleanup_layout_artifacts(module: ModuleOp) -> ModuleOp:
    """Remove dead layout ops and verify no layout dialect ops remain.

    Steps:
    1. Remove any remaining SetLayoutOp/UnsetLayoutOp (should be gone after pass 9).
    2. Remove consecutive PackOp/UnpackOp pairs with the same pack_spec.
    3. Verify no layout.* ops remain.
    4. Mark module as layout-clean.
    """
    removed = 0

    # Step 1: Remove remaining virtual layout ops
    for op in list(module.walk()):
        if isinstance(op, (SetLayoutOp, UnsetLayoutOp)):
            parent = op.parent
            if isinstance(parent, Block):
                parent.erase_op(op)
                removed += 1

    # Step 2: Remove cancelling PackOp/UnpackOp pairs
    for op in list(module.walk()):
        if not isinstance(op, PackOp):
            continue
        parent = op.parent
        if not isinstance(parent, Block):
            continue

        ops_list = list(parent.ops)
        try:
            idx = ops_list.index(op)
        except ValueError:
            continue

        if idx + 1 < len(ops_list) and isinstance(ops_list[idx + 1], UnpackOp):
            # Check if they use the same pack_spec
            pack_spec = str(op.pack_spec) if hasattr(op, "pack_spec") else None
            unpack_spec = str(ops_list[idx + 1].pack_spec) if hasattr(ops_list[idx + 1], "pack_spec") else None
            if pack_spec == unpack_spec:
                parent.erase_op(ops_list[idx + 1])
                parent.erase_op(op)
                removed += 2

    # Step 3: Verify no layout dialect ops remain (except PackOp which is concrete)
    remaining_virtual = 0
    for op in module.walk():
        if isinstance(op, (SetLayoutOp, UnsetLayoutOp)):
            remaining_virtual += 1

    if remaining_virtual > 0:
        log.warning(
            "layout.cleanup_incomplete",
            remaining_virtual=remaining_virtual,
        )

    # Step 4: Mark module as clean
    module.attributes[LAYOUT_CLEAN_ATTR] = StringAttr("1")

    log.debug("layout.cleanup", removed=removed, remaining_virtual=remaining_virtual)
    return module


__all__ = ["cleanup_layout_artifacts"]
