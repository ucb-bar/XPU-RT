"""Transform generation and application subpackage (Stage 3).

Handles LLM-driven generation of MLIR Transform Dialect scripts,
their application to the payload IR via xDSL/MLIR, and verification
that the transforms preserve semantics.

The LLM generates *transform programs* (scripts that describe what
rewrites to apply and with what parameters), not direct IR mutations.
This keeps the compiler deterministic and the LLM's role bounded.
"""

from __future__ import annotations

__all__: list[str] = []
