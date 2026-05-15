"""Benchmarking utilities for ``compile_through_pipeline`` runs.

Measures:

- **compile_time_ms** -- wall-clock time for one
  ``compile_through_pipeline`` run (cold; no cache reuse).
- **compile_time_cached_ms** -- second compile with the same
  ``CompGenOptions`` via ``PipelineCache`` (measures cache hit
  cost, should be near-zero).
- **executor_time_ms** -- CPU executor runtime over ``n_iter``
  runs (reports ``min`` / ``median`` / ``max``).
- **eager_time_ms** -- same measurements on the eager PyTorch
  reference for speedup comparison.
- **memory_bytes** -- approx peak memory via `resource.getrusage`
  on POSIX; falls back to 0 elsewhere.

No external dependencies beyond torch + stdlib.
"""

from __future__ import annotations

import gc
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import structlog

from compgen.options import CompGenOptions
from compgen.pipeline import PipelineCache, PipelineResult, compile_through_pipeline

log = structlog.get_logger()


@dataclass
class BenchmarkReport:
    fixture_name: str = ""
    compile_time_ms: float = 0.0
    compile_time_cached_ms: float = 0.0
    executor_time_ms_min: float = 0.0
    executor_time_ms_median: float = 0.0
    executor_time_ms_max: float = 0.0
    eager_time_ms_min: float = 0.0
    eager_time_ms_median: float = 0.0
    eager_time_ms_max: float = 0.0
    compile_memory_delta_bytes: int = 0
    executor_ops_run: int = 0
    executor_ops_skipped: int = 0
    pipeline_stages_run: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def compile_speedup(self) -> float:
        """Cache hit speedup: cold compile / warm compile."""
        if self.compile_time_cached_ms <= 0:
            return float("inf")
        return self.compile_time_ms / self.compile_time_cached_ms

    @property
    def executor_vs_eager(self) -> float:
        """Eager-median / executor-median. ``1.0`` = same speed. ``>1`` = executor faster."""
        if self.executor_time_ms_median <= 0:
            return 0.0
        return self.eager_time_ms_median / self.executor_time_ms_median


# --- helpers ----------------------------------------------------------------


def _now_ms() -> float:
    return time.perf_counter() * 1000.0


def _peak_memory_bytes() -> int:
    try:
        import resource

        r = resource.getrusage(resource.RUSAGE_SELF)
        return int(r.ru_maxrss) * 1024  # kB on linux
    except Exception:  # noqa: BLE001
        return 0


def _time_runs(fn: Callable[[], Any], n_iter: int) -> list[float]:
    samples: list[float] = []
    for _ in range(n_iter):
        gc.collect()
        start = _now_ms()
        fn()
        samples.append(_now_ms() - start)
    return samples


# --- main API ---------------------------------------------------------------


def measure_pipeline(
    model: Any,
    example_inputs: tuple[Any, ...],
    *,
    options: CompGenOptions | None = None,
    fixture_name: str = "",
    n_iter: int = 5,
    exported_program: Any = None,
) -> BenchmarkReport:
    """Measure one workload through the CompGen pipeline + CPU executor."""
    report = BenchmarkReport(fixture_name=fixture_name)
    if options is None:
        options = CompGenOptions()

    # --- Compile cold ------------------------------------------------
    mem_before = _peak_memory_bytes()
    cold_start = _now_ms()
    pr: PipelineResult = compile_through_pipeline(
        model,
        example_inputs=example_inputs,
        options=options,
        workload_name=fixture_name,
    )
    report.compile_time_ms = _now_ms() - cold_start
    mem_after = _peak_memory_bytes()
    report.compile_memory_delta_bytes = max(0, mem_after - mem_before)
    report.pipeline_stages_run = pr.stages_run

    if pr.module is None:
        report.notes.append("bridge failed; skipping executor timing")
        return report

    # --- Compile cached ---------------------------------------------
    cache = PipelineCache()
    cache.compile(model, example_inputs, options=options)  # warm
    warm_start = _now_ms()
    cache.compile(model, example_inputs, options=options)  # hit
    report.compile_time_cached_ms = _now_ms() - warm_start

    # --- Executor timing --------------------------------------------
    import torch

    try:
        from compgen.runtime.cpu_executor import ExecutorStats, execute

        ep = exported_program
        if ep is None:
            ep = torch.export.export(model, example_inputs)

        stats = ExecutorStats()

        def _run_executor():
            execute(pr.module, ep, example_inputs, stats=stats)

        samples = _time_runs(_run_executor, n_iter)
        report.executor_time_ms_min = min(samples)
        report.executor_time_ms_median = statistics.median(samples)
        report.executor_time_ms_max = max(samples)
        report.executor_ops_run = stats.ops_executed
        report.executor_ops_skipped = stats.ops_skipped
    except Exception as exc:  # noqa: BLE001
        report.notes.append(f"executor timing skipped: {exc}")

    # --- Eager timing -----------------------------------------------
    try:
        import torch

        was_training = getattr(model, "training", None)
        if was_training is True:
            model.eval()

        def _run_eager():
            with torch.no_grad():
                model(*example_inputs)

        samples = _time_runs(_run_eager, n_iter)
        report.eager_time_ms_min = min(samples)
        report.eager_time_ms_median = statistics.median(samples)
        report.eager_time_ms_max = max(samples)
        if was_training is True:
            model.train()
    except Exception as exc:  # noqa: BLE001
        report.notes.append(f"eager timing skipped: {exc}")

    log.info(
        "bench.done",
        fixture=fixture_name,
        compile_ms=report.compile_time_ms,
        compile_cached_ms=report.compile_time_cached_ms,
        exec_ms=report.executor_time_ms_median,
        eager_ms=report.eager_time_ms_median,
    )
    return report


def measure_pipeline_suite(
    fixtures: list[Any],
    *,
    options: CompGenOptions | None = None,
    n_iter: int = 5,
) -> list[BenchmarkReport]:
    """Run ``measure_pipeline`` over a list of fixtures."""
    reports: list[BenchmarkReport] = []
    for fixture_fn in fixtures:
        fx = fixture_fn()
        reports.append(
            measure_pipeline(
                fx.model,
                fx.example_inputs,
                options=options,
                fixture_name=fx.name,
                n_iter=n_iter,
                exported_program=fx.exported,
            )
        )
    return reports


__all__ = [
    "BenchmarkReport",
    "measure_pipeline",
    "measure_pipeline_suite",
]
