"""Verify rewrite families via the PDL verification backend.

Entry point for rewrite verification in CompGen's pipeline. Accepts
either callable pairs or xDSL modules and routes to the appropriate
backend.
"""

from __future__ import annotations

from typing import Any, Callable

import structlog
from xdsl.dialects.builtin import ModuleOp

from compgen.semantic.backends.xdsl_smt.pdl_backend import PDLVerificationBackend
from compgen.semantic.backends.xdsl_smt.results import PDLResult

log = structlog.get_logger()


def verify_rewrite_family(
    pattern: Callable[..., Any] | ModuleOp,
    replacement: Callable[..., Any] | ModuleOp | None = None,
    num_operands: int = 2,
    max_bitwidth: int = 32,
) -> PDLResult:
    """Verify a rewrite family across all bitwidths.

    Args:
        pattern: Pattern callable or PDL module.
        replacement: Replacement callable (required for callable path).
        num_operands: Number of symbolic operands.
        max_bitwidth: Maximum bitwidth to check.

    Returns:
        PDLResult with soundness outcome.
    """
    backend = PDLVerificationBackend()

    if callable(pattern) and callable(replacement):
        return backend.verify_arith_rewrite(
            build_pattern=pattern,
            build_replacement=replacement,
            num_operands=num_operands,
            max_bitwidth=max_bitwidth,
        )

    if isinstance(pattern, ModuleOp):
        return backend.verify_pattern(
            pattern_module=pattern,
            max_bitwidth=max_bitwidth,
        )

    log.warning("verify_rewrite_family.unsupported_input")
    return PDLResult(sound=False, status="unknown")


__all__ = ["verify_rewrite_family"]
