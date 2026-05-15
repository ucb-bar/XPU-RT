"""Tests for targets/calibrate.py -- hardware calibration."""

from __future__ import annotations

from typing import Any

from compgen.targets.calibrate import CalibratedProfile, CalibrationResult, Calibrator, calibrate
from compgen.targets.schema import TargetProfile


class MockDeviceHandle:
    """Fake device handle that returns pre-set benchmark values."""

    def __init__(self, values: dict[str, float] | None = None) -> None:
        self._values = values or {"hbm_bandwidth": 1200.0, "matmul_fp16": 312.0}

    def run_benchmark(self, name: str, params: dict[str, Any] | None = None) -> float:
        return self._values.get(name, 0.0)


def _make_profile(**overrides: Any) -> TargetProfile:
    defaults: dict[str, Any] = {
        "name": "test-target",
        "cost_model": {"hbm_bandwidth": 900.0, "matmul_fp16": 200.0},
    }
    defaults.update(overrides)
    return TargetProfile(**defaults)


def test_calibration_result_construction() -> None:
    r = CalibrationResult(
        benchmark="hbm_bandwidth",
        value=900.0,
        unit="GB/s",
    )
    assert r.benchmark == "hbm_bandwidth"
    assert r.value == 900.0
    assert r.unit == "GB/s"
    assert r.samples == 1
    assert r.std_dev == 0.0


def test_calibrator_defaults() -> None:
    c = Calibrator()
    assert "hbm_bandwidth" in c.benchmarks
    assert "matmul_fp16" in c.benchmarks
    assert c.num_samples == 10


def test_calibrator_custom_params() -> None:
    c = Calibrator(benchmarks=["memory_bw"], num_samples=5)
    assert c.benchmarks == ["memory_bw"]
    assert c.num_samples == 5


def test_calibrator_calibrate() -> None:
    """Calibrator.calibrate should return a CalibratedProfile."""
    profile = _make_profile()
    handle = MockDeviceHandle()
    cal = Calibrator()
    result = cal.calibrate(profile, device_handle=handle)

    assert isinstance(result, CalibratedProfile)
    assert len(result.results) == len(cal.benchmarks)
    # Values come from the mock handle
    bw_result = next(r for r in result.results if r.benchmark == "hbm_bandwidth")
    assert bw_result.value == 1200.0
    assert bw_result.unit == "GB/s"
    assert bw_result.samples == cal.num_samples

    # Updated profile should carry calibration_data
    assert result.profile.calibration_data["hbm_bandwidth"] == 1200.0
    assert result.profile.calibration_data["matmul_fp16"] == 312.0


def test_calibrate_convenience() -> None:
    """calibrate convenience wrapper should use synthetic defaults without a device."""
    profile = _make_profile()
    result = calibrate(profile)

    assert isinstance(result, CalibratedProfile)
    # Without a device handle, values come from profile cost_model
    bw_result = next(r for r in result.results if r.benchmark == "hbm_bandwidth")
    assert bw_result.value == 900.0
    assert bw_result.samples == 0
    assert result.profile.calibration_data["hbm_bandwidth"] == 900.0
