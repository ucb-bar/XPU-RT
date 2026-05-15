"""Dialect-provider substrate.

A ``DialectProviderCard`` describes an MLIR-dialect lowering backed
by a target-specific compiler (CUDA Tile IR, Hexagon-MLIR, Pallas,
NKI, Exo, Gemmini C, Radiance/Muon, IREE, Triton-MLIR, StableHLO).
The card declares which IR it consumes and emits; the dialect
provider implementation lives in the corresponding extension or
adapter module.
"""

from __future__ import annotations

from compgen.dialects.dialect_provider_types import (
    DialectProviderCard,
    DialectProviderCardError,
)

__all__ = [
    "DialectProviderCard",
    "DialectProviderCardError",
]
