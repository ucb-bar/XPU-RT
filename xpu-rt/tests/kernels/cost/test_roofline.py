"""Roofline analytical cost-model tests.

Pin down the production-grade contract of :mod:`compgen.kernels.cost.roofline`:

- Compute-bound and memory-bound regimes classify correctly.
- Missing peaks raise :class:`RooflineUnavailableError` rather than
  silently returning 0.0.
- ``predict_fusion_speedup`` compares real predictions on both sides
  (no 1.0 placeholder).
- ``RooflinePrediction.as_measurement`` round-trips into the
  common :class:`KernelMeasurement` type downstream code consumes.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
from compgen.kernels.cost.roofline import (
    RooflinePrediction,
    predict,
    predict_fusion_speedup,
    roofline_latency_us,
)
from compgen.kernels.errors import RooflineUnavailableError
from compgen.targets.schema import ComputeUnit, DeviceSpec, MemoryLevel, TargetProfile


def _a100_profile() -> TargetProfile:
    """A partial A100-ish profile: 312 TFLOPS (TF32), 2 TB/s HBM, 40 MB L2."""
    return TargetProfile(
        name="a100-mini",
        devices=[
            DeviceSpec(
                device_type="gpu",
                name="A100",
                vendor="nvidia",
                compute_units=[
                    ComputeUnit(name="tensor_core", count=1, peak_tflops=312.0),
                ],
                memory_hierarchy=[
                    MemoryLevel(name="l2", size_bytes=40 * 1024 * 1024, bandwidth_gbps=6000.0),
                    MemoryLevel(name="hbm", size_bytes=80 * 1024 * 1024 * 1024, bandwidth_gbps=2000.0),
                ],
            )
        ],
    )


def _contract(flops: int, bytes_read: int, bytes_written: int) -> SimpleNamespace:
    return SimpleNamespace(cost=SimpleNamespace(flops=flops, bytes_read=bytes_read, bytes_written=bytes_written))


class TestRooflineLatencyFormula:
    def test_compute_bound(self) -> None:
        """High FLOPS / low bytes → compute bound."""
        lat, regime = roofline_latency_us(
            flops=312_000_000_000,  # 312 GFLOPs
            bytes_moved=1_000_000,  # 1 MB
            peak_flops_per_s=312e12,
            peak_bandwidth_bps=2e12,
        )
        assert regime == "compute-bound"
        assert lat == pytest.approx(1000.0, rel=0.01)  # 312e9/312e12 = 1ms = 1000us

    def test_memory_bound(self) -> None:
        """Low arithmetic intensity → memory-bound.

        2 GB @ 2 TB/s = 1 ms = 1000 µs. With only 1 M FLOPs the compute
        time is ~3 ns — memory wins.
        """
        lat, regime = roofline_latency_us(
            flops=1_000_000,  # trivial work
            bytes_moved=2_000_000_000,  # 2 GB
            peak_flops_per_s=312e12,
            peak_bandwidth_bps=2e12,
        )
        assert regime == "memory-bound"
        assert lat == pytest.approx(1000.0, rel=0.01)

    def test_zero_peak_flops_raises(self) -> None:
        with pytest.raises(RooflineUnavailableError, match="peak_flops_per_s"):
            roofline_latency_us(
                flops=1_000_000,
                bytes_moved=1_000_000,
                peak_flops_per_s=0.0,
                peak_bandwidth_bps=2e12,
            )

    def test_zero_peak_bandwidth_raises(self) -> None:
        with pytest.raises(RooflineUnavailableError, match="peak_bandwidth_bps"):
            roofline_latency_us(
                flops=1_000_000,
                bytes_moved=1_000_000,
                peak_flops_per_s=312e12,
                peak_bandwidth_bps=0.0,
            )

    def test_zero_work_raises(self) -> None:
        """Contract with zero flops AND zero bytes is a malformed
        contract — roofline must say so, not return 0.0."""
        with pytest.raises(RooflineUnavailableError, match="zero flops and zero bytes"):
            roofline_latency_us(
                flops=0,
                bytes_moved=0,
                peak_flops_per_s=312e12,
                peak_bandwidth_bps=2e12,
            )


class TestPredict:
    def test_predict_from_profile(self) -> None:
        profile = _a100_profile()
        # 312 GFLOPs with enough arithmetic intensity to fit in L2:
        contract = _contract(flops=312_000_000_000, bytes_read=500_000, bytes_written=500_000)
        pred = predict(contract, None, target_profile=profile)
        assert isinstance(pred, RooflinePrediction)
        assert pred.regime == "compute-bound"
        assert pred.latency_us > 0
        # AI = 312e9 / 1e6 = 312 000 → very high, compute bound
        assert pred.arithmetic_intensity == pytest.approx(312_000.0)
        assert pred.peak_flops_per_s == pytest.approx(312e12)

    def test_memory_level_picked_by_size(self) -> None:
        """Bytes that fit in L2 → L2 bandwidth used; bytes > L2 → HBM."""
        profile = _a100_profile()
        small = _contract(flops=1_000_000, bytes_read=1024, bytes_written=1024)
        big = _contract(flops=1_000_000, bytes_read=100 * 1024 * 1024 * 1024, bytes_written=0)
        small_pred = predict(small, None, target_profile=profile)
        big_pred = predict(big, None, target_profile=profile)
        assert small_pred.memory_level == "l2"
        assert big_pred.memory_level == "hbm"

    def test_no_peaks_declared_raises(self) -> None:
        """Profile with empty devices → unavailable, not silent 0.0."""
        bad_profile = TargetProfile(name="empty", devices=[])
        contract = _contract(flops=1_000, bytes_read=1_000, bytes_written=0)
        with pytest.raises(RooflineUnavailableError):
            predict(contract, None, target_profile=bad_profile)

    def test_no_peak_flops_raises(self) -> None:
        """Device with ComputeUnit but no peak_tflops → unavailable."""
        profile = TargetProfile(
            name="flopless",
            devices=[
                DeviceSpec(
                    device_type="gpu",
                    name="noops",
                    compute_units=[ComputeUnit(name="cu", count=1, peak_tflops=None)],
                    memory_hierarchy=[MemoryLevel(name="hbm", size_bytes=1 << 30, bandwidth_gbps=100.0)],
                )
            ],
        )
        contract = _contract(flops=1_000, bytes_read=1_000, bytes_written=0)
        with pytest.raises(RooflineUnavailableError, match="peak FLOPS"):
            predict(contract, None, target_profile=profile)


class TestPredictFusionSpeedup:
    def test_fusion_speedup_vs_separate(self) -> None:
        """Predicted speedup is the ratio of real latency sums; not 1.0."""
        profile = _a100_profile()
        # Two memory-bound parts (each moves 1 GB). Fused they move 1.5 GB
        # (shared intermediate stays in registers).
        part_a = _contract(flops=1_000_000, bytes_read=500_000_000, bytes_written=500_000_000)
        part_b = _contract(flops=1_000_000, bytes_read=500_000_000, bytes_written=500_000_000)
        fused = _contract(flops=2_000_000, bytes_read=500_000_000, bytes_written=500_000_000)
        ratio = predict_fusion_speedup(
            parts=[part_a, part_b],
            fused=fused,
            target_profile=profile,
        )
        # Separate = 2 × 0.5 ms = 1 ms; fused = 0.5 ms. Speedup ≈ 2×.
        assert ratio == pytest.approx(2.0, rel=0.05)
        assert not math.isnan(ratio)
        assert ratio != 1.0  # explicit: no placeholder

    def test_empty_parts_rejected(self) -> None:
        with pytest.raises(ValueError, match="parts must be non-empty"):
            predict_fusion_speedup(
                parts=[],
                fused=_contract(1, 1, 1),
                target_profile=_a100_profile(),
            )


class TestAsMeasurement:
    def test_prediction_adapts_to_measurement(self) -> None:
        """RooflinePrediction.as_measurement returns a KernelMeasurement
        with source=\"roofline\" — downstream uniform code works on both."""
        from compgen.kernels.measure import KernelMeasurement

        profile = _a100_profile()
        contract = _contract(flops=312_000_000_000, bytes_read=500_000, bytes_written=500_000)
        pred = predict(contract, None, target_profile=profile)
        m = pred.as_measurement()
        assert isinstance(m, KernelMeasurement)
        assert m.source == "roofline"
        assert m.latency_us == pred.latency_us
        assert m.device == "analytical"
