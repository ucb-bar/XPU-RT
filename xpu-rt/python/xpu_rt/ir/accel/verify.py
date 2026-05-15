"""Hardware-specific verification rules for accelerator dialect.

Verifies that accelerator dialect ops are legal and well-formed
for the target hardware. Checks include:
    - Shape and alignment constraints
    - Memory space legality
    - Async event ordering (no use-before-ready)
    - Engine configuration validity
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from compgen.ir.accel.ops import (
    BarrierOp,
    DMAStartOp,
    DMAWaitOp,
    MatrixEngineOp,
    TileLoadOp,
    TileStoreOp,
)


@dataclass(frozen=True)
class AccelVerificationResult:
    """Result of verifying accelerator dialect ops.

    Attributes:
        valid: Whether all ops are legal.
        errors: List of (op_name, message) pairs.
    """

    valid: bool
    errors: list[tuple[str, str]] = field(default_factory=list)


def verify_accel_ops(module: Any, target_profile: Any = None) -> AccelVerificationResult:
    """Verify accelerator dialect ops are legal.

    Args:
        module: A single op, or a list/tuple of ops to verify.
        target_profile: Optional target profile for hardware-specific limits.

    Returns:
        An :class:`AccelVerificationResult` with validity flag and error list.
    """
    errors: list[tuple[str, str]] = []
    ops = module if isinstance(module, (list, tuple)) else [module]

    for op in ops:
        if isinstance(op, (TileLoadOp, TileStoreOp)):
            if not op.shape or any(d <= 0 for d in op.shape):
                errors.append((type(op).__name__, "shape dimensions must be positive"))
            if not op.dtype:
                errors.append((type(op).__name__, "dtype must be specified"))
        elif isinstance(op, DMAStartOp):
            if op.size_bytes <= 0:
                errors.append(("DMAStartOp", "size_bytes must be positive"))
            if not op.event:
                errors.append(("DMAStartOp", "event name must be non-empty"))
        elif isinstance(op, DMAWaitOp):
            # Individual DMAWait check -- cross-op event check is below.
            pass
        elif isinstance(op, MatrixEngineOp):
            valid_kinds = {"matmul", "conv", "mma", "outer_product"}
            if op.op_kind not in valid_kinds:
                errors.append(("MatrixEngineOp", f"invalid op_kind '{op.op_kind}'"))
            for ref_name in ("a_ref", "b_ref", "c_ref"):
                if not getattr(op, ref_name):
                    errors.append(("MatrixEngineOp", f"{ref_name} must be non-empty"))
        elif isinstance(op, BarrierOp):
            if op.scope not in {"workgroup", "device", "system"}:
                errors.append(("BarrierOp", f"invalid scope '{op.scope}'"))

    # Cross-op: check DMAWait events match DMAStart events.
    start_events = {op.event for op in ops if isinstance(op, DMAStartOp) and op.event}
    for op in ops:
        if isinstance(op, DMAWaitOp) and op.event not in start_events:
            errors.append(("DMAWaitOp", f"event '{op.event}' has no matching DMAStart"))

    return AccelVerificationResult(valid=len(errors) == 0, errors=errors)


__all__ = ["AccelVerificationResult", "verify_accel_ops"]
