"""Pass 4: Propagate virtual layout encodings through transparent ops.

Pushes SetLayoutOp encodings through layout-transparent operations:
- ``arith.*`` (elementwise arithmetic)
- ``math.*`` (elementwise math)
- ``linalg.fill`` (broadcast)
- ``tensor.empty`` (allocation)

These ops are layout-agnostic: they produce output in whatever layout
their input has.  After propagation, entire subgraphs share a single
encoding without materialization.
"""

from __future__ import annotations

import structlog
from xdsl.dialects.builtin import ModuleOp, StringAttr

log = structlog.get_logger()

PROPAGATED_ENCODING_ATTR = "compgen.propagated_encoding"

# Ops that are layout-transparent (output layout = input layout)
_TRANSPARENT_PREFIXES = (
    "arith.",
    "math.",
)

_TRANSPARENT_OPS = frozenset(
    {
        "linalg.fill",
        "tensor.empty",
        "tensor.extract_slice",
        "tensor.insert_slice",
    }
)


def _is_ukernel_transparent(op) -> bool:  # type: ignore[no-untyped-def]
    """Check if an op is a transparent ukernel (compiler can see through it)."""
    attr = op.attributes.get("compgen.ukernel_transparency")
    return bool(attr and hasattr(attr, "data") and attr.data == "transparent")


def _is_transparent(op) -> bool:  # type: ignore[no-untyped-def]
    """Check if an op is layout-transparent."""
    if any(op.name.startswith(prefix) for prefix in _TRANSPARENT_PREFIXES):
        return True
    if op.name in _TRANSPARENT_OPS:
        return True
    # Transparent ukernels participate in layout propagation
    return _is_ukernel_transparent(op)


def _get_encoding(op) -> str | None:  # type: ignore[no-untyped-def]
    """Get the layout encoding from an op (propagated or direct)."""
    for attr_key in (PROPAGATED_ENCODING_ATTR, "compgen.layout_hint", "compgen.encoding"):
        attr = op.attributes.get(attr_key)
        if attr and hasattr(attr, "data"):
            return attr.data
    return None


def propagate_layouts(module: ModuleOp) -> ModuleOp:
    """Propagate layout encodings through transparent ops.

    For each layout-transparent op:
    - Check if any operand's producer has a layout encoding.
    - If so, propagate that encoding to this op.
    - This avoids materialization at transparent boundaries.

    Uses a forward dataflow walk: process ops in program order,
    propagating producer encodings to consumer ops.

    Args:
        module: The xDSL ModuleOp to transform.

    Returns:
        The same ModuleOp with propagated encoding attributes.
    """
    from xdsl.dialects.func import FuncOp, ReturnOp

    propagated = 0

    # Build a map from SSA values to their producing op's encoding
    value_encoding: dict[int, str] = {}

    for op in module.walk():
        if isinstance(op, (ModuleOp, FuncOp, ReturnOp)):
            continue

        # Record encodings for this op's results
        encoding = _get_encoding(op)
        if encoding:
            for result in op.results:
                value_encoding[id(result)] = encoding

        # If this op is transparent, try to inherit encoding from operands
        if _is_transparent(op) and PROPAGATED_ENCODING_ATTR not in op.attributes:
            for operand in op.operands:
                inherited = value_encoding.get(id(operand))
                if inherited:
                    op.attributes[PROPAGATED_ENCODING_ATTR] = StringAttr(inherited)
                    # Also record for this op's results
                    for result in op.results:
                        value_encoding[id(result)] = inherited
                    propagated += 1
                    break

    log.debug("layout.propagate_layouts", propagated=propagated)
    return module


__all__ = ["propagate_layouts"]
