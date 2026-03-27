"""Tests for targets/calibrate.py -- hardware calibration."""

from __future__ import annotations

import pytest
from compgen.targets.calibrate import CalibrationResult, Calibrator


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


@pytest.mark.skip(reason="scaffold only -- implementation pending")
def test_calibrator_calibrate() -> None:
    """Calibrator.calibrate should return a CalibratedProfile."""


@pytest.mark.skip(reason="scaffold only -- implementation pending")
def test_calibrate_convenience() -> None:
    """calibrate should work with default settings."""
