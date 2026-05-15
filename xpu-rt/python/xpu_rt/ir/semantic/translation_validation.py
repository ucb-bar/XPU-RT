"""Translation validation for IR lowerings.

Checks that a lowering (e.g., dialect conversion, progressive lowering step)
preserves the semantics of the source program. Uses refinement relations
encoded in the Semantic IR and solved via SMT.

Invariants:
    - Validation is sound (if it says "valid", the lowering is correct).
    - Validation may be incomplete (it may timeout or return "unknown").
    - Validation never modifies the IR.

Backend:
    Uses ``compgen.semantic.backends.xdsl_smt.tv_backend`` which lowers
    arith operations to Z3 bitvector expressions and checks refinement.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import structlog
from xdsl.dialects.builtin import ModuleOp

log = structlog.get_logger()


@dataclass(frozen=True)
class TranslationValidationResult:
    """Result of translation validation.

    Attributes:
        valid: True if the lowering is semantics-preserving.
        status: "valid", "invalid", "unknown", or "timeout".
        counterexample: Counterexample inputs (if invalid).
        solver_time_ms: Time spent in the solver.
    """

    valid: bool
    status: str = "unknown"
    counterexample: dict[str, Any] | None = None
    solver_time_ms: float = 0.0


def validate_translation(
    source_module: Any,
    target_module: Any,
    timeout_ms: int = 30_000,
) -> TranslationValidationResult:
    """Validate that target_module is a correct lowering of source_module.

    Uses the xdsl-smt translation validation backend to build a
    refinement formula and check it with Z3.

    Args:
        source_module: The source (higher-level) xDSL module.
        target_module: The target (lower-level) xDSL module.
        timeout_ms: Z3 solver timeout in milliseconds.

    Returns:
        TranslationValidationResult.
    """
    if source_module is target_module:
        return TranslationValidationResult(valid=True, status="valid")

    # Fast path: identical IR text means trivially valid
    if isinstance(source_module, ModuleOp) and isinstance(target_module, ModuleOp):
        import io

        from xdsl.printer import Printer

        try:
            src_buf = io.StringIO()
            Printer(stream=src_buf).print_op(source_module)
            tgt_buf = io.StringIO()
            Printer(stream=tgt_buf).print_op(target_module)
            if src_buf.getvalue() == tgt_buf.getvalue():
                return TranslationValidationResult(valid=True, status="valid")
        except Exception:
            pass

    # Real verification via SMT backend
    from compgen.semantic.backends.xdsl_smt.tv_backend import TranslationValidationBackend

    backend = TranslationValidationBackend(timeout_ms=timeout_ms)
    tv_result = backend.check_refinement(source_module, target_module)

    cex_dict: dict[str, Any] | None = None
    if tv_result.counterexample is not None:
        cex_dict = {
            "inputs": tv_result.counterexample.inputs,
            "expected": tv_result.counterexample.expected,
            "actual": tv_result.counterexample.actual,
            "summary": tv_result.counterexample.summary,
        }

    return TranslationValidationResult(
        valid=tv_result.ok,
        status=tv_result.status,
        counterexample=cex_dict,
        solver_time_ms=tv_result.solver_time_ms,
    )


__all__ = ["TranslationValidationResult", "validate_translation"]
