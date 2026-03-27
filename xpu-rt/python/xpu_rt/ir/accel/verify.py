"""Hardware-specific verification rules for accelerator dialect.

Verifies that accelerator dialect ops are legal and well-formed
for the target hardware. Checks include:
    - Shape and alignment constraints
    - Memory space legality
    - Async event ordering (no use-before-ready)
    - Engine configuration validity

TODO: Implement verify_accel_ops() with per-op rules.
TODO: Integrate with target profile for hardware-specific limits.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


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

    TODO: Walk all accel ops and check constraints.
    TODO: Use target_profile for hardware-specific limits.
    """
    raise NotImplementedError("verify_accel_ops is not yet implemented")


__all__ = ["AccelVerificationResult", "verify_accel_ops"]
