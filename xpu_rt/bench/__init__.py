"""Benchmarking harness for ``compile_through_pipeline`` + executor."""

from __future__ import annotations

from xpu_rt.bench.measure import (
    BenchmarkReport,
    measure_pipeline,
    measure_pipeline_suite,
)

__all__ = [
    "BenchmarkReport",
    "measure_pipeline",
    "measure_pipeline_suite",
]
