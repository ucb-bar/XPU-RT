"""Canonical form enforcement for CompGen IR.

Canonicalization ensures the LLM always sees a stable, low-entropy IR.
This reduces "almost-right" generations and makes verification deterministic.

MVP implementation: counts ops before/after and returns the module unchanged.
Actual canonicalization patterns (layout normalization, constant folding,
dead code elimination) come in Phase 1.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class CanonicalizationReport:
    """Report from a canonicalization pass.

    Attributes:
        ops_before: Number of ops before canonicalization.
        ops_after: Number of ops after canonicalization.
        transforms_applied: List of transform names applied.
        warnings: Canonicalization warnings (e.g., ambiguous layouts).
    """

    ops_before: int
    ops_after: int
    transforms_applied: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _count_ops(module: Any) -> int:
    """Count all operations in an xDSL module."""
    count = 0
    for op in module.walk():
        count += 1
    return count


@dataclass
class CanonicalizePass:
    """Canonicalization pass for xDSL modules.

    MVP: counts ops and returns the module unchanged. Future phases
    will add layout normalization, constant folding, and dead code
    elimination using xDSL's RewritePattern infrastructure.
    """

    def run(self, module: Any) -> tuple[Any, CanonicalizationReport]:
        """Run canonicalization on an xDSL module.

        Args:
            module: An xDSL module (builtin.ModuleOp).

        Returns:
            Tuple of (module, CanonicalizationReport).
        """
        ops_before = _count_ops(module)

        # MVP: no transformations applied yet
        transforms_applied: list[str] = []

        ops_after = _count_ops(module)

        return module, CanonicalizationReport(
            ops_before=ops_before,
            ops_after=ops_after,
            transforms_applied=transforms_applied,
        )


def canonicalize(module: Any) -> tuple[Any, CanonicalizationReport]:
    """Convenience function: canonicalize an xDSL module."""
    return CanonicalizePass().run(module)


__all__ = ["CanonicalizationReport", "CanonicalizePass", "canonicalize"]
