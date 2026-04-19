"""Payload-IR extension types.

Re-exports custom xDSL types that extend the builtin type system. Today
this covers the FP8 variants (``Float8E4M3FNType`` and ``Float8E5M2Type``)
that upstream MLIR ships but xDSL does not.
"""

from __future__ import annotations

from compgen.ir.payload.types.float8 import (
    Float8E4M3FNType,
    Float8E5M2Type,
)

__all__ = [
    "Float8E4M3FNType",
    "Float8E5M2Type",
]
