"""Reference cuda_tile adapter — in-tree mirror of bridge-validated #144 shape.

Bwell's external ``compgen_cuda_tile`` package validates this lowering
against the real ``cuda-tile-translate`` toolchain. This in-tree
reference reproduces the same MLIR text emission so:

- CompGen's PyPI wheel ships a working ``cuda_tile`` adapter without
  requiring users to clone the bwell-side package.
- Unit tests can regression-check the FFN single-tile MLIR pattern
  without any NVIDIA toolchain on PATH.
- The bundle stage gracefully degrades when ``cuda-tile-translate``
  is not installed: emits ``format="mlir-cuda-tile"`` (text) instead
  of ``format="cuda-tile-bitcode"``.

The lowering is **single-tile FFN matmul-relu-matmul** — the structural
pattern from bridge #144. Multi-tile partitioning (bridge #146 work)
will land as a sibling template once bwell validates the index-space
coord-op syntax against the real toolchain.
"""

from __future__ import annotations

from compgen.extensions.vendor_dialect.builtins.cuda_tile.adapter import (
    CudaTileReferenceAdapter,
    make_adapter,
)

__all__ = ["CudaTileReferenceAdapter", "make_adapter"]
