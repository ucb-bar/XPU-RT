"""Local execution engine — run PyTorch models and collect real measurements.

Executes models on CPU/GPU, collects timing, memory, and per-op profiling data.
This is the **real reward signal** for the agent — actual hardware measurements
rather than cost model estimates.

Three modes:
    - ``eager``:      run the original PyTorch model (baseline)
    - ``compiled``:   run via ``torch.compile`` (comparison target)
    - ``compgen_ir``: run the compiled xDSL payload IR through
      :func:`compgen.runtime.cpu_executor.execute`. This is the path that
      actually exercises CompGen's compile output.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
from xdsl.dialects.builtin import ModuleOp


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
    mode: str  # "eager" | "compiled" | "compgen_ir"
    num_iterations: int
    warmup_iterations: int
    per_run_us: list[float] = field(default_factory=list)
    #: Output tensor from one post-warmup run. Populated for every mode so
    #: callers can diff ``mode="eager"`` vs ``mode="compgen_ir"`` outputs
    #: without re-running. ``None`` if the run raised before completion.
    sample_output: torch.Tensor | None = None


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
        *,
        payload_module: ModuleOp | None = None,
        exported_program: Any = None,
    ) -> BenchmarkResult:
        """Run a benchmark on a model.

        Args:
            model: PyTorch model.
            sample_inputs: Sample input tensors.
            device: ``"cpu"`` or ``"cuda"``.
            mode: ``"eager"``, ``"compiled"``, or ``"compgen_ir"``.
            num_iterations: Number of timed iterations.
            warmup: Warmup iterations before timing.
            payload_module: CompGen xDSL ``ModuleOp``. Required when
                ``mode="compgen_ir"``. Executed through
                :func:`compgen.runtime.cpu_executor.execute`.
            exported_program: ``torch.export.ExportedProgram`` that
                produced ``payload_module``. Required when
                ``mode="compgen_ir"`` — the executor needs the
                graph-signature + state-dict.

        Returns:
            BenchmarkResult with real measurements and a captured
            ``sample_output`` for correctness verification.
        """
        model = model.eval()

        # mode="compgen_ir" is CPU-only today (cpu_executor is pure torch).
        if mode == "compgen_ir":
            if payload_module is None or exported_program is None:
                raise ValueError("mode='compgen_ir' requires payload_module and exported_program")
            if device == "cuda":
                # cpu_executor runs on CPU regardless; keep inputs on CPU
                # to match. GPU dispatch will land with the CUDA driver.
                device = "cpu"

        # Move to device
        if device == "cuda" and torch.cuda.is_available():
            model = model.to("cuda")
            inputs = tuple(x.to("cuda") if isinstance(x, torch.Tensor) else x for x in sample_inputs)
        else:
            device = "cpu"
            inputs = sample_inputs

        # Build the per-call function for the selected mode.
        run_fn: Any
        if mode == "compgen_ir":
            # Import lazily — keeps the module importable even when xDSL
            # cpu_executor isn't available (e.g. in minimal installs).
            from compgen.runtime.cpu_executor import execute as _compgen_execute

            # Narrow the optional kwargs for the closure. The None-check
            # above already guarantees these are populated.
            assert payload_module is not None
            assert exported_program is not None
            _pm: ModuleOp = payload_module
            _ep: Any = exported_program

            def _run_compgen_ir(*call_inputs: Any) -> torch.Tensor:
                return _compgen_execute(_pm, _ep, call_inputs)

            run_fn = _run_compgen_ir
        elif mode == "compiled":
            run_fn = torch.compile(model)
        else:
            run_fn = model

        # Warmup
        with torch.no_grad():
            for _ in range(warmup):
                run_fn(*inputs)

        if device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

        # Timed runs
        timings: list[float] = []
        sample_output: torch.Tensor | None = None
        with torch.no_grad():
            for i in range(num_iterations):
                if device == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                result = run_fn(*inputs)
                if device == "cuda":
                    torch.cuda.synchronize()
                timings.append((time.perf_counter() - t0) * 1e6)
                if i == num_iterations - 1:
                    # Capture the last iteration's output for diffing.
                    if isinstance(result, torch.Tensor):
                        sample_output = result.detach().clone()
                    elif isinstance(result, (tuple, list)) and result and isinstance(result[0], torch.Tensor):
                        sample_output = result[0].detach().clone()

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
            sample_output=sample_output,
        )

    def compare(
        self,
        model: nn.Module,
        sample_inputs: tuple[Any, ...],
        num_iterations: int = 100,
    ) -> ComparisonResult:
        """Run full comparison: eager CPU, eager GPU, compiled GPU."""
        eager_cpu = self.benchmark(model, sample_inputs, device="cpu", mode="eager", num_iterations=num_iterations)

        eager_gpu = None
        compiled_gpu = None
        speedup_compile = 0.0
        speedup_gpu = 0.0

        if torch.cuda.is_available():
            eager_gpu = self.benchmark(model, sample_inputs, device="cuda", mode="eager", num_iterations=num_iterations)
            compiled_gpu = self.benchmark(
                model, sample_inputs, device="cuda", mode="compiled", num_iterations=num_iterations
            )

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
