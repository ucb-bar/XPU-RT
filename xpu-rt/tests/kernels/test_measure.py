"""Kernel measurement tests.

Pin down the production-grade contract of ``measure_kernel``:
- Real timing on real callables; returned numbers are non-zero.
- ``UnmeasurableKernelError`` on every missing precondition
  (no runnable, no inputs, CUDA requested but unavailable, kernel
  raises during warmup).
- CUDA + CPU paths produce grounded numbers.
- ``bandwidth_gbps`` / ``flops_per_s`` derive from the contract when
  one is supplied.
- Speedup computation refuses to lie when either side is zero.
"""

from __future__ import annotations

import math

import pytest
import torch
from compgen.kernels.errors import UnmeasurableKernelError
from compgen.kernels.measure import (
    KernelMeasurement,
    iqr_filtered,
    measure_kernel,
    speedup,
)


class TestMeasureKernelPreconditions:
    def test_rejects_none_runnable(self) -> None:
        with pytest.raises(UnmeasurableKernelError, match="not callable"):
            measure_kernel(runnable=None, golden_inputs=(torch.zeros(2),))  # type: ignore[arg-type]

    def test_rejects_non_callable(self) -> None:
        with pytest.raises(UnmeasurableKernelError, match="not callable"):
            measure_kernel(runnable="not a function", golden_inputs=(torch.zeros(2),))  # type: ignore[arg-type]

    def test_rejects_none_inputs(self) -> None:
        with pytest.raises(UnmeasurableKernelError, match="golden_inputs"):
            measure_kernel(runnable=lambda x: x, golden_inputs=None)

    def test_rejects_iters_lt_1(self) -> None:
        with pytest.raises(ValueError, match="iters"):
            measure_kernel(
                runnable=lambda x: x,
                golden_inputs=(torch.zeros(2),),
                iters=0,
            )

    def test_rejects_negative_warmup(self) -> None:
        with pytest.raises(ValueError, match="warmup"):
            measure_kernel(
                runnable=lambda x: x,
                golden_inputs=(torch.zeros(2),),
                warmup=-1,
            )

    def test_cuda_requested_but_unavailable(self) -> None:
        if torch.cuda.is_available():
            pytest.skip("CUDA available; can't exercise the unavailable-path")
        with pytest.raises(UnmeasurableKernelError, match="torch.cuda"):
            measure_kernel(
                runnable=lambda x: x,
                golden_inputs=(torch.zeros(2),),
                device="cuda:0",
            )

    def test_kernel_raising_in_warmup(self) -> None:
        def boom(_x: torch.Tensor) -> torch.Tensor:
            raise RuntimeError("deliberate")

        with pytest.raises(UnmeasurableKernelError, match="warmup"):
            measure_kernel(
                runnable=boom,
                golden_inputs=(torch.zeros(2),),
                warmup=1,
                iters=1,
            )


class TestMeasureKernelCpuPath:
    def test_cpu_path_produces_real_timing(self) -> None:
        """CPU path returns a positive, non-zero mean latency."""
        x = torch.randn(64, 64)
        # Something measurable but quick.
        m = measure_kernel(
            runnable=lambda t: t @ t,
            golden_inputs=(x,),
            device="cpu",
            warmup=2,
            iters=10,
        )
        assert isinstance(m, KernelMeasurement)
        assert m.latency_us > 0.0
        assert m.iters == 10
        assert m.warmup == 2
        assert m.device == "cpu"
        assert m.source == "measured_cpu"
        # stddev is a non-negative real number.
        assert m.latency_stddev_us >= 0.0

    def test_cpu_path_fills_flops_and_bandwidth(self) -> None:
        """When a contract is supplied, flops/bandwidth derive from the
        measured latency — no placeholders."""
        from types import SimpleNamespace

        contract = SimpleNamespace(cost=SimpleNamespace(flops=1_000_000, bytes_read=4096, bytes_written=4096))
        m = measure_kernel(
            runnable=lambda t: t + t,
            golden_inputs=(torch.randn(64, 64),),
            contract=contract,
            device="cpu",
            warmup=1,
            iters=5,
        )
        assert m.flops_per_s > 0.0
        assert m.bandwidth_gbps > 0.0

    def test_cpu_path_zero_contract_fields(self) -> None:
        """Contract with zero flops/bytes yields zero-rate measurements
        (honest: no work was declared, no rate can be computed)."""
        from types import SimpleNamespace

        contract = SimpleNamespace(cost=SimpleNamespace(flops=0, bytes_read=0, bytes_written=0))
        m = measure_kernel(
            runnable=lambda t: t + t,
            golden_inputs=(torch.randn(8, 8),),
            contract=contract,
            device="cpu",
            warmup=1,
            iters=3,
        )
        assert m.latency_us > 0.0
        assert m.flops_per_s == 0.0
        assert m.bandwidth_gbps == 0.0


@pytest.mark.requires_gpu
class TestMeasureKernelCudaPath:
    def test_cuda_path_produces_real_timing(self) -> None:
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        x = torch.randn(64, 64, device="cuda")
        m = measure_kernel(
            runnable=lambda t: t @ t,
            golden_inputs=(x,),
            device="cuda:0",
            warmup=2,
            iters=5,
        )
        assert m.latency_us > 0.0
        assert m.device.startswith("cuda")
        assert m.source == "measured_gpu"


class TestSpeedup:
    def test_real_speedup(self) -> None:
        b = KernelMeasurement(latency_us=100.0, source="measured_gpu")
        c = KernelMeasurement(latency_us=25.0, source="measured_gpu")
        assert speedup(b, c) == pytest.approx(4.0)

    def test_zero_baseline_returns_nan(self) -> None:
        """A zero-latency "baseline" is a lie; return NaN instead of inf."""
        b = KernelMeasurement(latency_us=0.0, source="unknown")
        c = KernelMeasurement(latency_us=25.0, source="measured_gpu")
        result = speedup(b, c)
        assert math.isnan(result)

    def test_zero_candidate_returns_nan(self) -> None:
        b = KernelMeasurement(latency_us=100.0, source="measured_gpu")
        c = KernelMeasurement(latency_us=0.0, source="unknown")
        assert math.isnan(speedup(b, c))


class TestIqrFilter:
    def test_small_sample_passes_through(self) -> None:
        samples = [1.0, 2.0, 3.0]
        assert iqr_filtered(samples) == samples

    def test_drops_outliers(self) -> None:
        samples = [1.0, 1.1, 1.0, 1.2, 1.05, 100.0]
        kept = iqr_filtered(samples)
        # 100.0 should be dropped as an IQR outlier.
        assert 100.0 not in kept

    def test_empty_fence_returns_original(self) -> None:
        """If the IQR fence drops everything, keep original rather than
        return []."""
        samples = [1.0, 1.0, 1.0, 1.0]  # IQR = 0 → fence == [1,1], everything kept
        assert iqr_filtered(samples) == samples
