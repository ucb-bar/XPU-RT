"""torch.compile baseline and diagnostics.

Runs the model through TorchDynamo/torch.compile to establish a performance
baseline and collect diagnostic information (graph breaks, guards, op coverage,
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
        guard_observations: Guard metadata observed by TorchDynamo.
        graph_count: Number of captured graphs.
        op_count: Number of compiled ops across graphs.
        warnings: Diagnostic warnings.
    """

    graph_breaks: list[tuple[str, str]] = field(default_factory=list)
    guard_failures: int = 0
    op_coverage: dict[str, bool] = field(default_factory=dict)
    guard_observations: list["GuardObservation"] = field(default_factory=list)
    graph_count: int = 0
    op_count: int = 0
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class GuardObservation:
    """One guard observed during a TorchDynamo explain run."""

    name: str
    source: str
    create_fn: str = ""
    guard_types: tuple[str, ...] = ()
    code: tuple[str, ...] = ()
    stack: tuple[str, ...] = ()


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

    # Extract graph break and op coverage diagnostics
    num_graph_breaks = 0
    compiled_op_fraction = 1.0
    try:
        diag = collect_diagnostics(model, sample_inputs)
        num_graph_breaks = len(diag.graph_breaks)
        total_ops = len(diag.op_coverage)
        if total_ops > 0:
            compiled_ops = sum(1 for v in diag.op_coverage.values() if v)
            compiled_op_fraction = compiled_ops / total_ops
    except Exception:
        pass

    return BaselineReport(
        cold_compile_ms=cold_ms,
        warm_run_ms=warm_ms,
        num_graph_breaks=num_graph_breaks,
        compiled_op_fraction=compiled_op_fraction,
        backend=backend,
    )


def _extract_break_reason(reason: Any) -> tuple[str, str]:
    """Convert a TorchDynamo break reason object into stable text."""

    location = ""
    reason_text = str(getattr(reason, "reason", reason))
    user_stack = getattr(reason, "user_stack", None)
    if user_stack:
        frame = user_stack[0]
        filename = getattr(frame, "filename", "")
        lineno = getattr(frame, "lineno", "")
        if filename:
            location = f"{filename}:{lineno}" if lineno else str(filename)
    return location, reason_text


def _extract_guard_observation(guard: Any) -> GuardObservation:
    """Normalize a TorchDynamo guard object into a serializable dataclass."""

    guard_types = tuple(
        str(item) for item in (getattr(guard, "guard_types", None) or ())
    )
    code = tuple(str(item) for item in (getattr(guard, "code_list", None) or ()))
    user_stack = tuple(str(frame) for frame in (getattr(guard, "user_stack", None) or ()))
    create_fn = getattr(guard, "create_fn_name", "") or ""
    if not create_fn:
        raw_create_fn = getattr(guard, "create_fn", None)
        create_fn = getattr(raw_create_fn, "__name__", "")
    return GuardObservation(
        name=str(getattr(guard, "name", "") or ""),
        source=str(getattr(guard, "source", "") or ""),
        create_fn=create_fn,
        guard_types=guard_types,
        code=code,
        stack=user_stack,
    )


def collect_diagnostics(model: nn.Module, sample_inputs: tuple[Any, ...]) -> DynamoReport:
    """Collect detailed TorchDynamo diagnostics.

    Uses ``torch._dynamo.explain`` when available so the export boundary can
    record graph-break and guard information from the installed runtime.
    """
    warnings_list: list[str] = []
    graph_breaks: list[tuple[str, str]] = []
    guard_observations: list[GuardObservation] = []
    op_coverage: dict[str, bool] = {}
    graph_count = 0
    op_count = 0

    try:
        explain = torch._dynamo.explain(model)
        with torch.no_grad():
            output = explain(*sample_inputs)

        graph_count = int(getattr(output, "graph_count", 0) or 0)
        op_count = int(getattr(output, "op_count", 0) or 0)

        for reason in getattr(output, "break_reasons", ()) or ():
            graph_breaks.append(_extract_break_reason(reason))

        for guard in getattr(output, "out_guards", ()) or ():
            guard_observations.append(_extract_guard_observation(guard))

        for ops in getattr(output, "ops_per_graph", ()) or ():
            for op in ops:
                op_coverage[str(op)] = True
    except Exception as e:
        warnings_list.append(f"torch._dynamo.explain failed: {e}")
        try:
            compiled = torch.compile(model, backend="inductor")
            with torch.no_grad():
                compiled(*sample_inputs)
        except Exception as compile_exc:
            warnings_list.append(f"torch.compile failed: {compile_exc}")

    return DynamoReport(
        graph_breaks=graph_breaks,
        guard_failures=0,
        op_coverage=op_coverage,
        guard_observations=guard_observations,
        graph_count=graph_count,
        op_count=op_count,
        warnings=warnings_list,
    )


__all__ = [
    "BaselineReport",
    "DynamoReport",
    "GuardObservation",
    "collect_diagnostics",
    "compile_baseline",
]
