"""Ukernel dialect -- stable leaf-call boundary for all kernel backends.

Provides a uniform IR representation for calling into external kernel
implementations, regardless of backend (Triton, CUDA, vendor library,
handwritten assembly, MLIR bodies).

Two execution classes share one unified contract:
    - **Transparent ukernels**: MLIR/xDSL bodies that stay compiler-visible
      for fusion, layout propagation, prepacking.
    - **Opaque ukernels**: C/Triton/library/binary bodies behind the same contract.

Four operation types:
    - ``UkernelDeclOp``: semantic + layout contract
    - ``UkernelMatchOp``: declarative selection constraints
    - ``UkernelBodyOp``: implementation (transparent or opaque)
    - ``UkernelCallOp``: stable call boundary in the graph

Supporting infrastructure:
    - ``UkernelContract``: layout-aware interface contract
    - ``ConstraintContext`` + evaluator: declarative constraint matching
    - ``UkernelRegistry``: selection engine for decl+match+body triples
"""

from __future__ import annotations

__all__: list[str] = []
