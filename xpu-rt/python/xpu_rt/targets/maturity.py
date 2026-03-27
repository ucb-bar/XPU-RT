"""Target maturity levels.

Defines the 4-level maturity model for target enablement:

    L0: Recognized  -- profile parsed, capabilities inferred, validation passes
    L1: Correctness -- fallback/reference path produces correct outputs
    L2: Optimized   -- real recipes, generated kernels, solver plans beat fallback
    L3: Promoted    -- verified recipes, stable, reusable across workloads

Maturity is assessed automatically based on what artifacts exist and
what verification has passed for a target package.

Invariants:
    - Maturity is monotonically increasing (never downgrades silently).
    - Each level requires all previous levels.
    - Assessment is deterministic given the same target package state.

TODO: Implement assess_maturity() from target package state.
TODO: Implement level requirements as checkable predicates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any


class TargetMaturity(IntEnum):
    """Target maturity level."""

    L0_RECOGNIZED = 0
    L1_CORRECTNESS = 1
    L2_OPTIMIZED = 2
    L3_PROMOTED = 3


@dataclass(frozen=True)
class MaturityRequirement:
    """A single requirement for reaching a maturity level.

    Attributes:
        level: The maturity level this requirement belongs to.
        name: Requirement name.
        description: What must be true.
        satisfied: Whether this requirement is currently met.
    """

    level: TargetMaturity
    name: str
    description: str
    satisfied: bool = False


@dataclass(frozen=True)
class MaturityAssessment:
    """Assessment of a target's current maturity.

    Attributes:
        current_level: Highest fully satisfied maturity level.
        requirements: All requirements with satisfaction status.
        blockers: Requirements preventing the next level.
    """

    current_level: TargetMaturity
    requirements: list[MaturityRequirement] = field(default_factory=list)
    blockers: list[MaturityRequirement] = field(default_factory=list)


def assess_maturity(target_package: Any) -> MaturityAssessment:
    """Assess the maturity of a target package.

    L0 requires: profile valid, capabilities inferred.
    L1 requires: L0 + at least one workload produces correct outputs via fallback.
    L2 requires: L1 + at least one optimized recipe beats fallback + solver plan feasible.
    L3 requires: L2 + promoted recipes + verification ladder passes + replay deterministic.

    """
    requirements: list[MaturityRequirement] = []
    current_level = TargetMaturity.L0_RECOGNIZED

    # L0: profile exists and capabilities present
    has_profile = hasattr(target_package, "profile") and target_package.profile is not None
    has_caps = hasattr(target_package, "capabilities") and target_package.capabilities is not None

    requirements.append(MaturityRequirement(
        level=TargetMaturity.L0_RECOGNIZED, name="profile_valid",
        description="Target profile is loaded and valid", satisfied=has_profile,
    ))
    requirements.append(MaturityRequirement(
        level=TargetMaturity.L0_RECOGNIZED, name="capabilities_inferred",
        description="Capability spec is inferred", satisfied=has_caps,
    ))

    l0_ok = has_profile and has_caps

    # L1-L3: not checkable yet (require pipeline stages)
    requirements.append(MaturityRequirement(
        level=TargetMaturity.L1_CORRECTNESS, name="fallback_correct",
        description="At least one workload correct via fallback", satisfied=False,
    ))
    requirements.append(MaturityRequirement(
        level=TargetMaturity.L2_OPTIMIZED, name="recipe_beats_fallback",
        description="At least one optimized recipe beats fallback", satisfied=False,
    ))
    requirements.append(MaturityRequirement(
        level=TargetMaturity.L3_PROMOTED, name="promoted_verified",
        description="Promoted recipes pass full verification ladder", satisfied=False,
    ))

    if l0_ok:
        current_level = TargetMaturity.L0_RECOGNIZED
    # Future: check L1, L2, L3

    blockers = [r for r in requirements if not r.satisfied and r.level.value == current_level.value + 1]

    return MaturityAssessment(
        current_level=current_level,
        requirements=requirements,
        blockers=blockers,
    )


__all__ = ["MaturityAssessment", "MaturityRequirement", "TargetMaturity", "assess_maturity"]
