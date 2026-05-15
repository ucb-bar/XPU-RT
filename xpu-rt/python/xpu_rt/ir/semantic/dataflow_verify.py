"""Dataflow analysis verification.

Verifies that dataflow analysis results (e.g., known-bits, range analysis,
aliasing) are sound with respect to the concrete semantics.

Invariants:
    - Verification checks that the analysis is an over-approximation.
    - Unsound analyses are flagged with counterexamples.

Backend:
    Uses ``compgen.semantic.backends.xdsl_smt.transfer_backend`` for
    forward-transfer soundness checking via Z3.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import structlog

log = structlog.get_logger()


@dataclass(frozen=True)
class AnalysisVerificationResult:
    """Result of verifying a dataflow analysis.

    Attributes:
        sound: Whether the analysis is a sound over-approximation.
        status: "sound", "unsound", "unknown", or "timeout".
        counterexample: Counterexample (if unsound).
        solver_time_ms: Solver time.
    """

    sound: bool
    status: str = "unknown"
    counterexample: dict[str, Any] | None = None
    solver_time_ms: float = 0.0


def verify_analysis(
    analysis_result: Any,
    ir_module: Any,
    concrete_fn: Callable[..., Any] | None = None,
    transfer_fn: Callable[..., Any] | None = None,
    abstract_constraint: Callable[..., Any] | None = None,
    instance_constraint: Callable[..., Any] | None = None,
    num_operands: int = 2,
    max_bitwidth: int = 32,
) -> AnalysisVerificationResult:
    """Verify a dataflow analysis result is sound.

    Supports two modes:

    1. **Callable path**: Pass ``concrete_fn``, ``transfer_fn``,
       ``abstract_constraint``, and ``instance_constraint`` to verify a
       forward transfer function via the Z3 backend.

    2. **Dict/result path**: Pass a pre-computed result dict or
       ``AnalysisVerificationResult`` directly.

    Args:
        analysis_result: The analysis output to verify (or dict/result).
        ir_module: The IR module the analysis was run on.
        concrete_fn: Concrete operation semantics.
        transfer_fn: Abstract transfer function to verify.
        abstract_constraint: Consistency check (concrete, abstract) -> bool.
        instance_constraint: Abstract domain validity check.
        num_operands: Number of operands.
        max_bitwidth: Maximum bitwidth for verification.

    Returns:
        AnalysisVerificationResult.
    """
    # Direct passthrough of pre-computed results
    if isinstance(analysis_result, AnalysisVerificationResult):
        return analysis_result
    if isinstance(analysis_result, dict) and "sound" in analysis_result:
        return AnalysisVerificationResult(
            sound=bool(analysis_result["sound"]),
            status="sound" if analysis_result["sound"] else "unsound",
            counterexample=analysis_result.get("counterexample"),
            solver_time_ms=float(analysis_result.get("solver_time_ms", 0.0)),
        )

    # Callable path: verify via Z3 backend
    if all(fn is not None for fn in [concrete_fn, transfer_fn, abstract_constraint, instance_constraint]):
        from compgen.semantic.backends.xdsl_smt.transfer_backend import TransferVerificationBackend

        backend = TransferVerificationBackend()
        result = backend.verify_forward_transfer(
            concrete_fn=concrete_fn,
            transfer_fn=transfer_fn,
            abstract_constraint=abstract_constraint,
            instance_constraint=instance_constraint,
            num_operands=num_operands,
            max_bitwidth=max_bitwidth,
        )

        cex_dict: dict[str, Any] | None = None
        if result.counterexample is not None:
            cex_dict = {
                "inputs": result.counterexample.inputs,
                "summary": result.counterexample.summary,
            }

        return AnalysisVerificationResult(
            sound=result.sound,
            status=result.status,
            counterexample=cex_dict,
            solver_time_ms=result.solver_time_ms,
        )

    return AnalysisVerificationResult(sound=False, status="unknown")


__all__ = ["AnalysisVerificationResult", "verify_analysis"]
