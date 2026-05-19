"""Unit tests for the QRB5165 target spec + roofline bound."""

from __future__ import annotations

import pytest

from xpu_rt.targets.backends.qnn.target_spec import (
    OpFootprint, QRB5165_BACKENDS, analytical_bound_us, backend_dtype,
    is_compute_only,
)


def test_all_published_backends_present():
    for b in ("CPU", "GPU", "DSP", "HTA", "HTP"):
        assert b in QRB5165_BACKENDS


def test_dsp_int8_peak_dominates_cpu_fp32_on_int8_workload():
    """An int8 conv on DSP should be cheaper than its CPU fp32 lower bound."""
    op = OpFootprint(flops=1e9, bytes_read=1e6, bytes_written=1e6)
    dsp_us, _ = analytical_bound_us(op, "DSP")
    cpu_us, _ = analytical_bound_us(op, "CPU")
    assert dsp_us < cpu_us


def test_memory_bound_op_returns_memory_bound_rationale():
    # Tiny compute, big bytes → memory dominates.
    op = OpFootprint(flops=10.0, bytes_read=1e9, bytes_written=1e9)
    us, rationale = analytical_bound_us(op, "GPU")
    assert rationale == "memory-bound"
    # Bandwidth = 32 GB/s, bytes = 2 GB → ~62 ms.
    assert us > 50_000
    assert us < 80_000


def test_compute_bound_op_returns_compute_bound_rationale():
    # Huge compute, tiny bytes → compute dominates.
    op = OpFootprint(flops=1e12, bytes_read=1.0, bytes_written=1.0)
    us, rationale = analytical_bound_us(op, "HTA")
    assert rationale == "compute-bound"
    # HTA 3.4 TOPS → 1e12 / 3.4e12 ≈ 294 ms.
    assert us > 200_000


def test_is_compute_only():
    op_compute = OpFootprint(flops=1e12, bytes_read=1.0, bytes_written=1.0)
    op_memory = OpFootprint(flops=10.0, bytes_read=1e9, bytes_written=1e9)
    assert is_compute_only(op_compute, "GPU")
    assert not is_compute_only(op_memory, "GPU")


def test_unknown_backend_raises():
    with pytest.raises(ValueError):
        analytical_bound_us(OpFootprint(1.0, 1.0, 1.0), "VPU")


def test_backend_dtype_known():
    assert backend_dtype("CPU") == "fp32"
    assert backend_dtype("GPU") == "fp16"
    assert backend_dtype("DSP") == "int8"
    assert backend_dtype("HTA") == "int8"
    assert backend_dtype("HTP") == "int8"
