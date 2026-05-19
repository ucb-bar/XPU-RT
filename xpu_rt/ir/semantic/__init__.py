"""Semantic IR -- the verification/trust layer.

Layer 3 of the three-layer IR stack. Provides dialect-agnostic verification
tools by making semantics first-class. Inspired by "First-Class Verification
Dialects for MLIR" (PLDI'25).

Capabilities:
    - Translation validation (check lowering preserves semantics)
    - Peephole rewrite verification (prove individual rewrites correct)
    - Dataflow analysis verification (check analysis soundness)

Dialect semantics are defined as lowerings into semantic dialects, which
then lower into SMT queries for automated checking.
"""

from __future__ import annotations

__all__: list[str] = []
