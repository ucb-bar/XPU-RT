"""Tests for target profile validation."""

from __future__ import annotations

from pathlib import Path

from compgen.targets.schema import DeviceSpec, Interconnect, TargetProfile, load_profile
from compgen.targets.validate import validate_profile, validate_profile_file

PROFILES_DIR = Path(__file__).parent.parent.parent / "examples" / "target_profiles"


def test_valid_profile_passes() -> None:
    p = load_profile(PROFILES_DIR / "cuda_a100.yaml")
    result = validate_profile(p)
    assert result.valid
    assert len(result.errors) == 0


def test_empty_devices_fails() -> None:
    p = TargetProfile(name="empty")
    result = validate_profile(p)
    assert not result.valid
    assert any("at least one device" in e.message.lower() for e in result.errors)


def test_invalid_interconnect_device_index_fails() -> None:
    p = TargetProfile(
        name="bad-ic",
        devices=[DeviceSpec(device_type="gpu", name="gpu0")],
        interconnects=[Interconnect(topology="pcie", bandwidth_gbps=31.5, devices=(0, 5))],
    )
    result = validate_profile(p)
    assert not result.valid
    assert any("out of range" in e.message for e in result.errors)


def test_valid_multi_device_passes() -> None:
    p = load_profile(PROFILES_DIR / "multi_device.yaml")
    result = validate_profile(p)
    assert result.valid


def test_validate_profile_file_works() -> None:
    result = validate_profile_file(PROFILES_DIR / "cuda_a100.yaml")
    assert result.valid


def test_validate_missing_file() -> None:
    result = validate_profile_file("/nonexistent/path.yaml")
    assert not result.valid
    assert any("not found" in e.message.lower() for e in result.errors)
