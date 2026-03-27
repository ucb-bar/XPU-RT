"""torch.compile baseline and diagnostics.

Runs the model through TorchDynamo/torch.compile to establish a performance
baseline and collect diagnostic information (graph breaks, op coverage,
compilation time, recompilation behavior).

This baseline is what CompGen must beat -- or at least match -- while adding
heterogeneous target support.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn


@dataclass(frozen=True)
class BaselineReport:
    """Performance baseline from torch.compile.

    Attributes:
        cold_compile_ms: First compilation time in milliseconds.
        warm_run_ms: Median warm run time in milliseconds.
        num_graph_breaks: Number of graph breaks.
        compiled_op_fraction: Fraction of ops in compiled regions.
        backend: Backend used (e.g., "inductor").
    """

    cold_compile_ms: float
    warm_run_ms: float
    num_graph_breaks: int
    compiled_op_fraction: float
    backend: str = "inductor"


@dataclass(frozen=True)
class DynamoReport:
    """Detailed diagnostics from TorchDynamo.

    Attributes:
        graph_breaks: List of (location, reason) tuples.
        guard_failures: Number of guard failures causing recompilation.
        op_coverage: Dict mapping op name to compiled/not-compiled status.
        warnings: Diagnostic warnings.
    """

    graph_breaks: list[tuple[str, str]] = field(default_factory=list)
    guard_failures: int = 0
    op_coverage: dict[str, bool] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def compile_baseline(
    model: nn.Module,
    sample_inputs: tuple[Any, ...],
    backend: str = "inductor",
    num_warmup: int = 3,
    num_runs: int = 10,
) -> BaselineReport:
    """Run torch.compile and collect baseline metrics."""
    model.eval()

    # Cold compile: first invocation triggers compilation
    compiled = torch.compile(model, backend=backend)
    t0 = time.perf_counter()
    with torch.no_grad():
        compiled(*sample_inputs)
    cold_ms = (time.perf_counter() - t0) * 1000

    # Warm runs
    with torch.no_grad():
        for _ in range(num_warmup):
            compiled(*sample_inputs)

    timings = []
    with torch.no_grad():
        for _ in range(num_runs):
            t0 = time.perf_counter()
            compiled(*sample_inputs)
            timings.append((time.perf_counter() - t0) * 1000)

    timings.sort()
    warm_ms = timings[len(timings) // 2]  # median

    return BaselineReport(
        cold_compile_ms=cold_ms,
        warm_run_ms=warm_ms,
        num_graph_breaks=0,  # TODO: extract from Dynamo logs
        compiled_op_fraction=1.0,  # TODO: measure actual coverage
        backend=backend,
    )


def collect_diagnostics(model: nn.Module, sample_inputs: tuple[Any, ...]) -> DynamoReport:
    """Collect detailed TorchDynamo diagnostics.

    MVP: Run torch.compile and capture basic info. Full Dynamo log parsing
    is deferred to Phase 1.
    """
    warnings_list: list[str] = []

    try:
        compiled = torch.compile(model, backend="inductor")
        with torch.no_grad():
            compiled(*sample_inputs)
    except Exception as e:
        warnings_list.append(f"torch.compile failed: {e}")

    return DynamoReport(
        graph_breaks=[],
        guard_failures=0,
        op_coverage={},
        warnings=warnings_list,
    )


__all__ = ["BaselineReport", "DynamoReport", "collect_diagnostics", "compile_baseline"]
