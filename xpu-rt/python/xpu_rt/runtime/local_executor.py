"""Local execution engine — run PyTorch models and collect real measurements.

Executes models on CPU/GPU, collects timing, memory, and per-op profiling data.
This is the **real reward signal** for the agent — actual hardware measurements
rather than cost model estimates.

Two modes:
    - Eager: run the original PyTorch model (baseline)
    - Compiled: run via torch.compile (comparison target)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn


@dataclass(frozen=True)
class BenchmarkResult:
    """Real hardware measurements from execution.

    This is the agent's ground-truth reward signal.
    """

    latency_median_us: float
    latency_p99_us: float
    throughput_samples_per_sec: float
    peak_memory_bytes: int
    device: str
    mode: str                        # "eager", "compiled", "compgen"
    num_iterations: int
    warmup_iterations: int
    per_run_us: list[float] = field(default_factory=list)


@dataclass(frozen=True)
class ComparisonResult:
    """Side-by-side comparison of different execution modes."""

    eager_cpu: BenchmarkResult | None = None
    eager_gpu: BenchmarkResult | None = None
    compiled_gpu: BenchmarkResult | None = None
    speedup_compile_vs_eager: float = 0.0
    speedup_gpu_vs_cpu: float = 0.0


class LocalExecutor:
    """Execute models and collect real measurements."""

    def benchmark(
        self,
        model: nn.Module,
        sample_inputs: tuple[Any, ...],
        device: str = "cpu",
        mode: str = "eager",
        num_iterations: int = 100,
        warmup: int = 10,
    ) -> BenchmarkResult:
        """Run a benchmark on a model.

        Args:
            model: PyTorch model.
            sample_inputs: Sample input tensors.
            device: "cpu" or "cuda".
            mode: "eager" or "compiled".
            num_iterations: Number of timed iterations.
            warmup: Warmup iterations before timing.

        Returns:
            BenchmarkResult with real measurements.
        """
        model = model.eval()

        # Move to device
        if device == "cuda" and torch.cuda.is_available():
            model = model.to("cuda")
            inputs = tuple(
                x.to("cuda") if isinstance(x, torch.Tensor) else x
                for x in sample_inputs
            )
        else:
            device = "cpu"
            inputs = sample_inputs

        # Compile if requested
        run_fn: Any = model
        if mode == "compiled":
            run_fn = torch.compile(model)

        # Warmup
        with torch.no_grad():
            for _ in range(warmup):
                run_fn(*inputs)

        if device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

        # Timed runs
        timings: list[float] = []
        with torch.no_grad():
            for _ in range(num_iterations):
                if device == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                run_fn(*inputs)
                if device == "cuda":
                    torch.cuda.synchronize()
                timings.append((time.perf_counter() - t0) * 1e6)

        timings.sort()
        median = timings[len(timings) // 2]
        p99 = timings[int(len(timings) * 0.99)]

        peak_mem = 0
        if device == "cuda":
            peak_mem = torch.cuda.max_memory_allocated()

        batch_size = 1
        if isinstance(inputs[0], torch.Tensor):
            batch_size = inputs[0].shape[0]

        throughput = batch_size / (median / 1e6) if median > 0 else 0

        return BenchmarkResult(
            latency_median_us=median,
            latency_p99_us=p99,
            throughput_samples_per_sec=throughput,
            peak_memory_bytes=peak_mem,
            device=device,
            mode=mode,
            num_iterations=num_iterations,
            warmup_iterations=warmup,
            per_run_us=timings,
        )

    def compare(
        self,
        model: nn.Module,
        sample_inputs: tuple[Any, ...],
        num_iterations: int = 100,
    ) -> ComparisonResult:
        """Run full comparison: eager CPU, eager GPU, compiled GPU."""
        eager_cpu = self.benchmark(model, sample_inputs, device="cpu", mode="eager",
                                   num_iterations=num_iterations)

        eager_gpu = None
        compiled_gpu = None
        speedup_compile = 0.0
        speedup_gpu = 0.0

        if torch.cuda.is_available():
            eager_gpu = self.benchmark(model, sample_inputs, device="cuda", mode="eager",
                                       num_iterations=num_iterations)
            compiled_gpu = self.benchmark(model, sample_inputs, device="cuda", mode="compiled",
                                          num_iterations=num_iterations)

            if eager_gpu.latency_median_us > 0:
                speedup_compile = eager_gpu.latency_median_us / compiled_gpu.latency_median_us
            if eager_cpu.latency_median_us > 0 and eager_gpu.latency_median_us > 0:
                speedup_gpu = eager_cpu.latency_median_us / eager_gpu.latency_median_us

        return ComparisonResult(
            eager_cpu=eager_cpu,
            eager_gpu=eager_gpu,
            compiled_gpu=compiled_gpu,
            speedup_compile_vs_eager=speedup_compile,
            speedup_gpu_vs_cpu=speedup_gpu,
        )


__all__ = ["BenchmarkResult", "ComparisonResult", "LocalExecutor"]
