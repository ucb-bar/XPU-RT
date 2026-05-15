"""Pass 5: Hoist layout ops to dominating positions.

Moves SetLayoutOp markers out of local regions to reduce the number
of runtime layout decisions.  If a SetLayoutOp encoding dominates all
uses within a region, hoist it to the region/function entry.

In the current attribute-based representation, this means consolidating
layout attributes: if all ops in a contiguous sequence share the same
encoding, mark the region boundary instead of individual ops.
"""

from __future__ import annotations

import structlog
from xdsl.dialects.builtin import ModuleOp, StringAttr

log = structlog.get_logger()

HOISTED_ENCODING_ATTR = "compgen.hoisted_encoding"


def hoist_layout_ops(module: ModuleOp) -> ModuleOp:
    """Hoist layout encodings to dominating positions.

    For each function in the module:
    - Collect all unique encoding values across ops.
    - If a single encoding dominates (>= 80% of encoded ops), mark the
      function with ``compgen.hoisted_encoding`` for that layout.
    - Individual ops retain their encodings for fine-grained passes.

    Args:
        module: The xDSL ModuleOp to transform.

    Returns:
        The same ModuleOp with hoisted encoding attributes on functions.
    """
    from xdsl.dialects.func import FuncOp, ReturnOp

    hoisted = 0

    for op in module.walk():
        if not isinstance(op, FuncOp):
            continue

        # Collect encodings within this function
        encoding_counts: dict[str, int] = {}
        total_encoded = 0

        for inner_op in op.walk():
            if isinstance(inner_op, (ModuleOp, FuncOp, ReturnOp)):
                continue
            for attr_key in ("compgen.propagated_encoding", "compgen.layout_hint", "compgen.encoding"):
                attr = inner_op.attributes.get(attr_key)
                if attr and hasattr(attr, "data"):
                    enc: str = attr.data
                    encoding_counts[enc] = encoding_counts.get(enc, 0) + 1
                    total_encoded += 1
                    break

        if not encoding_counts or total_encoded == 0:
            continue

        # Find dominant encoding (>= 80% threshold)
        dominant_enc = max(encoding_counts, key=encoding_counts.get)  # type: ignore[arg-type]
        ratio = encoding_counts[dominant_enc] / total_encoded

        if ratio >= 0.8:
            op.attributes[HOISTED_ENCODING_ATTR] = StringAttr(dominant_enc)
            hoisted += 1

    log.debug("layout.hoist_layout_ops", hoisted=hoisted)
    return module


__all__ = ["hoist_layout_ops"]
