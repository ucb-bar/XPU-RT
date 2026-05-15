"""L0-L9 verification ladder for generated targets.

Maps fine-grained verification levels to the existing TargetMaturity (L0-L3).
Each level has a set of tests that a target must pass to advance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

from compgen.targetgen.classify import Classification
from compgen.targetgen.hardware_spec import HardwareSpec
from compgen.targetgen.plan import SupportPlan
from compgen.targets.maturity import TargetMaturity


class VerificationLevel(IntEnum):
    """10-level verification ladder."""

    L0_SPEC_SANITY = 0
    L1_PROFILE_VALID = 1
    L2_CAPABILITIES_INFERRED = 2
    L3_STACK_CONSTRUCTED = 3
    L4_CONTRACTS_PASS = 4
    L5_SHARED_PASSES = 5
    L6_PLUGIN_PASSES = 6
    L7_DIFFERENTIAL_CORRECT = 7
    L8_BUNDLE_VALID = 8
    L9_PERF_CHARACTERIZED = 9


# Mapping from verification levels to TargetMaturity
MATURITY_MAP: dict[VerificationLevel, TargetMaturity] = {
    VerificationLevel.L0_SPEC_SANITY: TargetMaturity.L0_RECOGNIZED,
    VerificationLevel.L1_PROFILE_VALID: TargetMaturity.L0_RECOGNIZED,
    VerificationLevel.L2_CAPABILITIES_INFERRED: TargetMaturity.L0_RECOGNIZED,
    VerificationLevel.L3_STACK_CONSTRUCTED: TargetMaturity.L1_CORRECTNESS,
    VerificationLevel.L4_CONTRACTS_PASS: TargetMaturity.L1_CORRECTNESS,
    VerificationLevel.L5_SHARED_PASSES: TargetMaturity.L1_CORRECTNESS,
    VerificationLevel.L6_PLUGIN_PASSES: TargetMaturity.L1_CORRECTNESS,
    VerificationLevel.L7_DIFFERENTIAL_CORRECT: TargetMaturity.L2_OPTIMIZED,
    VerificationLevel.L8_BUNDLE_VALID: TargetMaturity.L2_OPTIMIZED,
    VerificationLevel.L9_PERF_CHARACTERIZED: TargetMaturity.L3_PROMOTED,
}


@dataclass(frozen=True)
class VerificationTest:
    """A single verification test."""

    level: VerificationLevel
    name: str
    description: str
    requires_hardware: bool = False


@dataclass(frozen=True)
class VerificationManifest:
    """Complete verification manifest for a target."""

    target_name: str
    tests: list[VerificationTest] = field(default_factory=list)
    highest_achievable: VerificationLevel = VerificationLevel.L9_PERF_CHARACTERIZED

    @property
    def maturity(self) -> TargetMaturity:
        """Map highest achievable level to TargetMaturity."""
        return MATURITY_MAP.get(self.highest_achievable, TargetMaturity.L0_RECOGNIZED)

    def tests_at_level(self, level: VerificationLevel) -> list[VerificationTest]:
        """Get all tests for a specific level."""
        return [t for t in self.tests if t.level == level]

    def levels_with_tests(self) -> list[VerificationLevel]:
        """Get all levels that have tests."""
        return sorted({t.level for t in self.tests})


def _base_tests() -> list[VerificationTest]:
    """Tests every target must pass."""
    vl = VerificationLevel
    return [
        VerificationTest(vl.L0_SPEC_SANITY, "spec_parses", "HardwareSpec loads"),
        VerificationTest(vl.L0_SPEC_SANITY, "spec_validates", "Validation passes"),
        VerificationTest(vl.L1_PROFILE_VALID, "profile_extracted", "Profile extracts"),
        VerificationTest(vl.L1_PROFILE_VALID, "profile_validates", "Profile validates"),
        VerificationTest(vl.L2_CAPABILITIES_INFERRED, "capabilities_built", "Caps inferred"),
        VerificationTest(vl.L2_CAPABILITIES_INFERRED, "all_ops_mapped", "Ops mapped"),
        VerificationTest(vl.L3_STACK_CONSTRUCTED, "stack_builds", "Stack constructs"),
        VerificationTest(vl.L3_STACK_CONSTRUCTED, "stage_count_matches", "Stages match"),
        VerificationTest(vl.L4_CONTRACTS_PASS, "input_contracts", "Input contracts pass"),
        VerificationTest(vl.L4_CONTRACTS_PASS, "output_contracts", "Output contracts pass"),
        VerificationTest(vl.L5_SHARED_PASSES, "shared_on_arith", "Shared on arith"),
        VerificationTest(vl.L5_SHARED_PASSES, "shared_on_matmul", "Shared on matmul"),
        VerificationTest(vl.L8_BUNDLE_VALID, "bundle_creates", "Bundle created"),
        VerificationTest(vl.L8_BUNDLE_VALID, "bundle_has_payload", "Has payload.mlir"),
    ]


def _plugin_tests(plan: SupportPlan) -> list[VerificationTest]:
    """Plugin-specific tests based on the support plan."""
    tests: list[VerificationTest] = []
    for stage_req in plan.required_stages:
        if stage_req.needs_plugin:
            tests.append(
                VerificationTest(
                    VerificationLevel.L6_PLUGIN_PASSES,
                    f"plugin_{stage_req.stage_name}_runs",
                    f"Plugin for {stage_req.stage_name} runs without error",
                )
            )
    return tests


def _differential_tests(spec: HardwareSpec) -> list[VerificationTest]:
    """Differential correctness tests."""
    tests = [
        VerificationTest(
            VerificationLevel.L7_DIFFERENTIAL_CORRECT,
            "golden_match",
            f"Output matches golden model ({spec.verification_surface.golden_model})",
        ),
    ]
    if spec.verification_surface.has_simulator:
        tests.append(
            VerificationTest(
                VerificationLevel.L7_DIFFERENTIAL_CORRECT,
                "simulator_match",
                "Output matches simulator execution",
                requires_hardware=True,
            )
        )
    return tests


def _perf_tests(spec: HardwareSpec) -> list[VerificationTest]:
    """Performance characterization tests."""
    tests = [
        VerificationTest(
            VerificationLevel.L9_PERF_CHARACTERIZED,
            "cost_model_populated",
            "Cost model has measured data for key ops",
            requires_hardware=True,
        ),
    ]
    if spec.verification_surface.performance_counters:
        tests.append(
            VerificationTest(
                VerificationLevel.L9_PERF_CHARACTERIZED,
                "perf_counters_read",
                f"Read performance counters: {spec.verification_surface.performance_counters[:3]}",
                requires_hardware=True,
            )
        )
    return tests


def generate_verification_manifest(
    spec: HardwareSpec,
    classification: Classification,
    plan: SupportPlan,
) -> VerificationManifest:
    """Generate a verification manifest for a target."""
    tests = _base_tests() + _plugin_tests(plan) + _differential_tests(spec) + _perf_tests(spec)

    # Determine highest achievable without hardware
    highest = VerificationLevel.L9_PERF_CHARACTERIZED
    if not spec.verification_surface.has_simulator and not spec.verification_surface.has_emulator:
        # Can't do L9 without some form of execution
        highest = VerificationLevel.L8_BUNDLE_VALID

    return VerificationManifest(
        target_name=spec.name,
        tests=tests,
        highest_achievable=highest,
    )
