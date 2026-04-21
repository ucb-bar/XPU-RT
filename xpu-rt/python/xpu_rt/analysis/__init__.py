"""CompGen analysis infrastructure.

Wave 2 modules:
  * ``dim_semantics``       — per-dim role tagging (parallel/reduce/broadcast)
  * ``transpose_propagation`` — transpose-chain detection + cancellation
"""

from __future__ import annotations

from compgen.analysis.dim_semantics import (
    DimRole,
    OpDimAnnotation,
    annotate_dim_roles,
    dim_roles_for_op,
)
from compgen.analysis.transpose_propagation import (
    TransposeChain,
    detect_transpose_chains,
    propose_transpose_cancellations,
)

__all__ = [
    "DimRole",
    "OpDimAnnotation",
    "TransposeChain",
    "annotate_dim_roles",
    "detect_transpose_chains",
    "dim_roles_for_op",
    "propose_transpose_cancellations",
]
