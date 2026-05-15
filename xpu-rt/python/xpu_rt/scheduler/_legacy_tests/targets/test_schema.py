"""Tests for target profile schema and loading."""

from __future__ import annotations

from pathlib import Path

from compgen.targets.schema import DeviceSpec, MemoryLevel, TargetProfile, load_profile

PROFILES_DIR = Path(__file__).parent.parent.parent / "examples" / "target_profiles"


def test_target_profile_construction() -> None:
    profile = TargetProfile(name="test-target")
    assert profile.name == "test-target"
    assert profile.schema_version == "1.0"
    assert profile.devices == []


def test_device_spec_construction() -> None:
    device = DeviceSpec(device_type="gpu", name="TestGPU", vendor="test")
    assert device.device_type == "gpu"


def test_memory_level_construction() -> None:
    mem = MemoryLevel(name="hbm", size_bytes=80_000_000_000, bandwidth_gbps=2039.0)
    assert mem.bandwidth_gbps == 2039.0


def test_load_cuda_a100_profile() -> None:
    p = load_profile(PROFILES_DIR / "cuda_a100.yaml")
    assert p.name == "cuda-a100"
    assert len(p.devices) == 1
    assert p.devices[0].device_type == "gpu"
    assert p.devices[0].name == "A100-SXM4-80GB"
    assert len(p.devices[0].compute_units) == 2
    assert len(p.devices[0].memory_hierarchy) == 5
    assert "triton" in p.devices[0].kernel_backends


def test_load_trainium_profile() -> None:
    p = load_profile(PROFILES_DIR / "trainium1.yaml")
    assert p.name == "trainium1"
    assert p.devices[0].device_type == "accelerator"
    assert "nki" in p.devices[0].kernel_backends


def test_load_multi_device_profile() -> None:
    p = load_profile(PROFILES_DIR / "multi_device.yaml")
    assert p.name == "cpu-plus-a100"
    assert len(p.devices) == 2
    assert len(p.interconnects) == 1
    assert p.interconnects[0].topology == "pcie"
    device_types = {d.device_type for d in p.devices}
    assert device_types == {"cpu", "gpu"}


def test_load_profile_preserves_cost_model() -> None:
    p = load_profile(PROFILES_DIR / "cuda_a100.yaml")
    assert "op_latencies" in p.cost_model
    assert "memory_bandwidth" in p.cost_model
