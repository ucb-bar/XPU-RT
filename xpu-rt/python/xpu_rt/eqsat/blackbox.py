"""Op classification for equality saturation: PROFITABLE vs BLACKBOX.

Profitable ops are modeled in the e-graph. Blackboxed ops are treated
as opaque nodes — they participate in dataflow but their internals are
not explored by rewrite rules.

Blackbox criteria (from Constable):
  - Multi-result ops (xDSL eqsat limitation)
  - Custom accelerator / external calls
  - Control-heavy ops
  - Opaque library calls
  - Ops without known semantics
"""

from __future__ import annotations

from enum import Enum, auto

from xdsl.dialects import arith, func, linalg
from xdsl.dialects.builtin import ModuleOp
from xdsl.ir import Operation


class OpClass(Enum):
    """Classification of an op for e-graph inclusion."""

    PROFITABLE = auto()
    BLACKBOX = auto()


# Op types that are always profitable (single-result, well-understood semantics)
_PROFITABLE_OP_TYPES: set[type[Operation]] = {
    # Arithmetic (all single-result)
    arith.AddiOp,
    arith.SubiOp,
    arith.MuliOp,
    arith.AddfOp,
    arith.SubfOp,
    arith.MulfOp,
    arith.DivfOp,
    arith.NegfOp,
    arith.MaximumfOp,
    arith.MinimumfOp,
    arith.ConstantOp,
    arith.ExtFOp,
    arith.TruncFOp,
    arith.SIToFPOp,
    arith.FPToSIOp,
    arith.IndexCastOp,
    arith.SelectOp,
    arith.CmpfOp,
    arith.CmpiOp,
}

# Linalg ops that are single-result and worth optimizing
_PROFITABLE_LINALG_TYPES: set[type[Operation]] = {
    linalg.MatmulOp,
    linalg.GenericOp,
    linalg.TransposeOp,
    linalg.FillOp,
}

# Ops that should always be blackboxed
_BLACKBOX_OP_TYPES: set[type[Operation]] = {
    func.CallOp,
}


def classify_op(op: Operation) -> OpClass:
    """Classify a single operation as PROFITABLE or BLACKBOX.

    Args:
        op: The operation to classify.

    Returns:
        OpClass.PROFITABLE if the op should be modeled in the e-graph,
        OpClass.BLACKBOX otherwise.
    """
    # Multi-result ops must be blackboxed (xDSL eqsat limitation)
    if len(op.results) != 1:
        return OpClass.BLACKBOX

    # Check explicit profitable types
    op_type = type(op)
    if op_type in _PROFITABLE_OP_TYPES:
        return OpClass.PROFITABLE
    if op_type in _PROFITABLE_LINALG_TYPES:
        return OpClass.PROFITABLE

    # Check explicit blackbox types
    if op_type in _BLACKBOX_OP_TYPES:
        return OpClass.BLACKBOX

    # Math ops are generally profitable (single-result elementwise)
    if op.name.startswith("math."):
        return OpClass.PROFITABLE if len(op.results) == 1 else OpClass.BLACKBOX

    # Default: blackbox unknown ops
    return OpClass.BLACKBOX


def classify_module(module: ModuleOp) -> dict[Operation, OpClass]:
    """Classify all ops in a module.

    Args:
        module: The module to classify.

    Returns:
        Dict mapping each non-structural op to its classification.
    """
    classifications: dict[Operation, OpClass] = {}

    for op in module.walk():
        # Skip structural ops
        if isinstance(op, (ModuleOp, func.FuncOp, func.ReturnOp)):
            continue
        if not op.results:
            continue

        classifications[op] = classify_op(op)

    return classifications


def count_profitable(classifications: dict[Operation, OpClass]) -> int:
    """Count profitable ops."""
    return sum(1 for c in classifications.values() if c == OpClass.PROFITABLE)


def count_blackbox(classifications: dict[Operation, OpClass]) -> int:
    """Count blackboxed ops."""
    return sum(1 for c in classifications.values() if c == OpClass.BLACKBOX)
