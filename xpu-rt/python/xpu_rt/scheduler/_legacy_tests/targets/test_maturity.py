"""Tests for target maturity levels."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from compgen.targets.capability import infer_capabilities
from compgen.targets.maturity import MaturityAssessment, MaturityRequirement, TargetMaturity, assess_maturity
from compgen.targets.schema import load_profile

PROFILES_DIR = Path(__file__).parent.parent.parent / "examples" / "target_profiles"


def test_maturity_ordering() -> None:
    assert TargetMaturity.L0_RECOGNIZED < TargetMaturity.L1_CORRECTNESS
    assert TargetMaturity.L1_CORRECTNESS < TargetMaturity.L2_OPTIMIZED
    assert TargetMaturity.L2_OPTIMIZED < TargetMaturity.L3_PROMOTED


def test_maturity_requirement() -> None:
    req = MaturityRequirement(
        level=TargetMaturity.L0_RECOGNIZED,
        name="profile_valid",
        description="Profile parses and validates",
        satisfied=True,
    )
    assert req.satisfied


def test_maturity_assessment() -> None:
    assessment = MaturityAssessment(current_level=TargetMaturity.L0_RECOGNIZED, blockers=[])
    assert assessment.current_level == TargetMaturity.L0_RECOGNIZED


def test_assess_l0_maturity() -> None:
    """A package with profile + capabilities should reach L0."""
    profile = load_profile(PROFILES_DIR / "cuda_a100.yaml")
    caps = infer_capabilities(profile)
    pkg = SimpleNamespace(profile=profile, capabilities=caps)
    assessment = assess_maturity(pkg)
    assert assessment.current_level == TargetMaturity.L0_RECOGNIZED
    # Should have blockers for L1
    assert len(assessment.blockers) > 0


def test_assess_no_profile() -> None:
    """A package without profile should still return L0 but with unmet requirements."""
    pkg = SimpleNamespace(profile=None, capabilities=None)
    assessment = assess_maturity(pkg)
    assert assessment.current_level == TargetMaturity.L0_RECOGNIZED
    # Profile requirement should be unsatisfied
    profile_reqs = [r for r in assessment.requirements if r.name == "profile_valid"]
    assert len(profile_reqs) == 1
    assert not profile_reqs[0].satisfied
