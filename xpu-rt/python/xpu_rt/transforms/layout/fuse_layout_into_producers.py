"""Pass 6: Fuse layout encodings into producer ops.

When a producer op can directly emit the encoded layout (e.g., matmul
producing tiled output directly consumed by next matmul in same layout),
eliminates the SetLayout/UnsetLayout boundary. The producer absorbs
the consumer's layout request.
"""

from __future__ import annotations

import structlog
from xdsl.dialects.builtin import ModuleOp, StringAttr

log = structlog.get_logger()

FUSED_LAYOUT_ATTR = "compgen.fused_layout"


def fuse_layout_into_producers(module: ModuleOp) -> ModuleOp:
    """Fuse layout encodings into producer ops.

    For each op that produces results consumed by an op with a layout
    encoding:
    - If both producer and consumer have the same encoding, mark
      the boundary as fused (no materialization needed).
    - This eliminates unnecessary pack/unpack at same-layout boundaries.
    """
    from xdsl.dialects.func import FuncOp, ReturnOp

    fused = 0

    # Build producer-consumer encoding map
    op_encoding: dict[int, str] = {}
    for op in module.walk():
        if isinstance(op, (ModuleOp, FuncOp, ReturnOp)):
            continue
        for attr_key in ("compgen.propagated_encoding", "compgen.layout_hint", "compgen.encoding"):
            attr = op.attributes.get(attr_key)
            if attr and hasattr(attr, "data"):
                op_encoding[id(op)] = attr.data
                break

    # Check producer-consumer pairs
    for op in module.walk():
        if isinstance(op, (ModuleOp, FuncOp, ReturnOp)):
            continue

        consumer_enc = op_encoding.get(id(op))
        if not consumer_enc:
            continue

        # Check if all operands' producers have the same encoding
        all_match = True
        for operand in op.operands:
            if hasattr(operand, "owner"):
                producer_enc = op_encoding.get(id(operand.owner))
                if producer_enc and producer_enc == consumer_enc:
                    continue
                all_match = False
                break

        if all_match and op.operands:
            op.attributes[FUSED_LAYOUT_ATTR] = StringAttr(consumer_enc)
            fused += 1

    log.debug("layout.fuse_into_producers", fused=fused)
    return module


__all__ = ["fuse_layout_into_producers"]
