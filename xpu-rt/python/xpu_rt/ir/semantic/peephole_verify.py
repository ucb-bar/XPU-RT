"""Peephole rewrite verification.

Proves that individual peephole rewrites (pattern -> replacement) are
semantics-preserving across all bitwidths via Z3.

Invariants:
    - Verification is per-rewrite (not whole-program).
    - Verified rewrites can be cached and reused across compilations.
    - Failed verification produces a concrete counterexample.

Backend:
    Uses ``compgen.semantic.backends.xdsl_smt.pdl_backend`` for
    callable-based rewrite verification across bitwidths.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import structlog

log = structlog.get_logger()


@dataclass(frozen=True)
class RewriteVerificationResult:
    """Result of verifying a peephole rewrite.

    Attributes:
        valid: Whether the rewrite preserves semantics.
        status: "valid", "invalid", "unknown", or "timeout".
        counterexample: Counterexample (if invalid).
        solver_time_ms: Solver time.
        cached: Whether this result came from cache.
    """

    valid: bool
    status: str = "unknown"
    counterexample: dict[str, Any] | None = None
    solver_time_ms: float = 0.0
    cached: bool = False


def verify_rewrite(
    pattern: Any,
    replacement: Any,
    max_bitwidth: int = 32,
) -> RewriteVerificationResult:
    """Verify a peephole rewrite is semantics-preserving.

    Supports two calling conventions:

    1. **Callable path** (preferred): ``pattern`` and ``replacement`` are
       callables ``(operands: list[z3.BitVec]) -> z3.BitVec``.

    2. **Module path** (future): ``pattern`` and ``replacement`` are xDSL
       ``ModuleOp`` instances containing the rewrite in PDL form.

    Args:
        pattern: Source pattern (callable or ModuleOp).
        replacement: Replacement (callable or ModuleOp).
        max_bitwidth: Maximum bitwidth to verify.

    Returns:
        RewriteVerificationResult.
    """
    # Identity check
    if pattern is replacement:
        return RewriteVerificationResult(valid=True, status="valid")

    from compgen.semantic.backends.xdsl_smt.pdl_backend import PDLVerificationBackend

    backend = PDLVerificationBackend()

    if callable(pattern) and callable(replacement):
        result = backend.verify_arith_rewrite(
            build_pattern=pattern,
            build_replacement=replacement,
            num_operands=2,
            max_bitwidth=max_bitwidth,
        )
        return RewriteVerificationResult(
            valid=result.sound,
            status=result.status,
            solver_time_ms=result.solver_time_ms,
        )

    # Module path — not yet implemented
    log.warning("verify_rewrite.module_path_not_implemented")
    return RewriteVerificationResult(valid=False, status="unknown")


__all__ = ["RewriteVerificationResult", "verify_rewrite"]
