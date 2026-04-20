"""Tests for performance feedback calibration (Unit 15)."""

from __future__ import annotations

import pytest
from compgen.memory.calibration import calibrate_cost, get_calibration_factor, record_calibration


@pytest.fixture
def memory(tmp_path):
    from compgen.memory.store import CompilerMemory

    return CompilerMemory(
        db_path=tmp_path / "test.db",
        blob_root=tmp_path / "blobs",
    )


class TestCalibration:
    def test_record_returns_id(self, memory):
        kid = record_calibration(memory, "gpu_a100", "matmul", estimated_us=100.0, measured_us=120.0)
        assert kid
        assert len(kid) > 0

    def test_single_calibration_factor(self, memory):
        record_calibration(memory, "gpu_a100", "matmul", estimated_us=100.0, measured_us=200.0)
        factor = get_calibration_factor(memory, "gpu_a100", "matmul")
        assert abs(factor - 2.0) < 0.01

    def test_average_calibration_factor(self, memory):
        record_calibration(memory, "gpu_a100", "matmul", estimated_us=100.0, measured_us=150.0)
        record_calibration(memory, "gpu_a100", "matmul", estimated_us=100.0, measured_us=250.0)
        factor = get_calibration_factor(memory, "gpu_a100", "matmul")
        # average of 1.5 and 2.5 = 2.0
        assert abs(factor - 2.0) < 0.01

    def test_no_data_returns_one(self, memory):
        factor = get_calibration_factor(memory, "gpu_a100", "matmul")
        assert factor == 1.0

    def test_calibrate_cost(self):
        assert calibrate_cost(100.0, 2.0) == 200.0
        assert calibrate_cost(50.0, 1.0) == 50.0
