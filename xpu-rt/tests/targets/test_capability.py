"""Tests for target capability specification and classification."""

from __future__ import annotations

from pathlib import Path

from compgen.targets.capability import (
    BackendLane,
    OpCapability,
    TargetClass,
    classify_target,
    infer_capabilities,
)
from compgen.targets.schema import load_profile

PROFILES_DIR = Path(__file__).parent.parent.parent / "examples" / "target_profiles"


def test_target_class_values() -> None:
    assert TargetClass.TRITON_FRIENDLY.value == "triton_friendly"
    assert TargetClass.ACCEL_NATIVE.value == "accel_native"
    assert TargetClass.UKERNEL_RUNTIME.value == "ukernel_runtime"
    assert TargetClass.HYBRID.value == "hybrid"


def test_backend_lane_values() -> None:
    assert BackendLane.TRITON.value == "triton"
    assert BackendLane.FALLBACK.value == "fallback"


def test_op_capability() -> None:
    cap = OpCapability(op_name="matmul", preferred_lane=BackendLane.TRITON, estimated_speedup=3.5)
    assert cap.preferred_lane == BackendLane.TRITON
    assert cap.estimated_speedup == 3.5


def test_classify_gpu_target() -> None:
    """GPU with Triton backend should classify as TRITON_FRIENDLY."""
    p = load_profile(PROFILES_DIR / "cuda_a100.yaml")
    tc = classify_target(p)
    assert tc == TargetClass.TRITON_FRIENDLY


def test_classify_accel_target() -> None:
    """Accelerator target should classify as ACCEL_NATIVE."""
    p = load_profile(PROFILES_DIR / "trainium1.yaml")
    tc = classify_target(p)
    assert tc == TargetClass.ACCEL_NATIVE


def test_classify_hybrid_target() -> None:
    """Multi-device CPU+GPU should classify as HYBRID."""
    p = load_profile(PROFILES_DIR / "multi_device.yaml")
    tc = classify_target(p)
    assert tc == TargetClass.HYBRID


def test_infer_capabilities_gpu() -> None:
    """GPU target should get Triton as preferred lane for all ops."""
    p = load_profile(PROFILES_DIR / "cuda_a100.yaml")
    caps = infer_capabilities(p)
    assert caps.target_class == TargetClass.TRITON_FRIENDLY
    assert len(caps.op_capabilities) > 0
    for cap in caps.op_capabilities.values():
        assert cap.preferred_lane == BackendLane.TRITON


def test_infer_capabilities_accel() -> None:
    """Accelerator target should get ACCEL_DIALECT as preferred lane."""
    p = load_profile(PROFILES_DIR / "trainium1.yaml")
    caps = infer_capabilities(p)
    assert caps.target_class == TargetClass.ACCEL_NATIVE
    for cap in caps.op_capabilities.values():
        assert cap.preferred_lane == BackendLane.ACCEL_DIALECT
