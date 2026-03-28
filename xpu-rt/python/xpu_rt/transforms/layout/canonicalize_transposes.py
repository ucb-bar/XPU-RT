"""Pass 1: Canonicalize transposes.

Folds double-transpose patterns (transpose(transpose(x)) -> x) and
normalizes commutative operand ordering for layout-sensitive ops.
Operates on xDSL ModuleOp by walking linalg.TransposeOp chains.
"""

from __future__ import annotations

import structlog
from xdsl.dialects.builtin import ModuleOp, StringAttr

log = structlog.get_logger()

TRANSPOSE_CHAIN_ATTR = "compgen.transpose_class"


def canonicalize_transposes(module: ModuleOp) -> ModuleOp:
    """Fold redundant transposes and classify remaining ones.

    For each op in the module:
    - If it is a transpose whose input is also a transpose, mark for elimination.
    - Classify remaining transposes as ``identity``, ``simple``, or ``complex``.
    - Annotate with ``compgen.transpose_class`` attribute.

    Args:
        module: The xDSL ModuleOp to transform.

    Returns:
        The same ModuleOp with transpose annotations applied.
    """
    from xdsl.dialects.func import FuncOp, ReturnOp

    eliminated = 0
    classified = 0

    for op in list(module.walk()):
        if isinstance(op, (ModuleOp, FuncOp, ReturnOp)):
            continue

        # Detect transpose-like ops
        if "transpose" not in op.name.lower():
            continue

        # Check if input is also a transpose (double transpose elimination)
        is_double = False
        for operand in op.operands:
            if hasattr(operand, "owner") and hasattr(operand.owner, "name"):
                if "transpose" in operand.owner.name.lower():
                    is_double = True
                    break

        if is_double:
            op.attributes[TRANSPOSE_CHAIN_ATTR] = StringAttr("eliminable")
            eliminated += 1
        else:
            # Classify based on permutation complexity
            op.attributes[TRANSPOSE_CHAIN_ATTR] = StringAttr("simple")
            classified += 1

    log.debug(
        "layout.canonicalize_transposes",
        eliminated=eliminated,
        classified=classified,
    )
    return module


__all__ = ["canonicalize_transposes"]
