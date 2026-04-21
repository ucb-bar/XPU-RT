"""Per-kernel GPU microbenchmark harness.

Takes a :class:`KernelContractV3` + its generated Triton source + an
eager-PyTorch reference and runs:

  1. **Correctness**  — our kernel output vs eager reference (max abs /
     max rel error, with configurable ``atol`` / ``rtol``).
  2. **Latency**       — CUDA-event timed median of N runs on our kernel,
     on eager, and on ``torch.compile``'d eager (when supported). All on
     the same inputs, same device.
  3. **Speedup**       — ours/eager and ours/torch.compile ratios.

The harness returns a :class:`BenchResult` dataclass that callers can
write to disk, feed into the calibration loop, or render as a table.

Design choices:

* **CUDA events, not wall clock** — avoids CPU scheduler noise; the
  events are recorded on the default stream and ``torch.cuda.synchronize``
  bookends the measurement window.
* **Warmup before timing** — first N calls are thrown away so JIT
  compilation, allocator bring-up, and clock ramp don't poison the
  median. Default warmup=10, timed=100.
* **Pluggable eager reference** — the caller provides ``eager_ref``
  as a plain ``callable(*inputs) -> Tensor``, so we don't have to guess
  dtype / shape semantics.
"""

from __future__ import annotations

import statistics
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass(frozen=True)
class BenchResult:
    """Outcome of one kernel microbench."""

    name: str
    device: str
    dtype_in: str
    dtype_out: str
    # Correctness
    max_abs_err: float
    max_rel_err: float
    passed: bool
    # Latency (microseconds, median of ``timed`` runs after ``warmup`` warmups)
    our_us: float
    eager_us: float
    torch_compile_us: float | None  # None when torch.compile unavailable or skipped
    # Derived ratios (ours/eager, ours/torch_compile) — >1 means we're slower
    us_ratio_vs_eager: float
    us_ratio_vs_torch_compile: float | None
    # Provenance
    warmup_iters: int
    timed_iters: int
    input_shapes: list[list[int]] = field(default_factory=list)
    notes: str = ""


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------


def _cuda_event_time_us(fn: Callable[[], Any], *, warmup: int, timed: int) -> float:
    """Return the *median* latency of ``fn`` in microseconds.

    Uses ``torch.cuda.Event`` pairs per iteration. The returned median is
    a robust estimator — min is too optimistic (GPU clock-boost hit),
    mean is contaminated by the occasional jitter tail.
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    samples_ms: list[float] = []
    for _ in range(timed):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        samples_ms.append(start.elapsed_time(end))   # ms
    return statistics.median(samples_ms) * 1_000.0   # → μs


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


def run_microbench(
    name: str,
    *,
    our_fn: Callable[[], torch.Tensor],
    eager_ref: Callable[[], torch.Tensor],
    atol: float = 1e-3,
    rtol: float = 1e-3,
    torch_compile_fn: Callable[[], torch.Tensor] | None = None,
    warmup: int = 10,
    timed: int = 100,
    input_shapes: list[list[int]] | None = None,
    notes: str = "",
) -> BenchResult:
    """Drive one microbench.

    Args:
        name: Human label (e.g. ``"matmul_512x1024x512_fp16_turing"``).
        our_fn: Zero-arg callable that invokes OUR kernel and returns the
            output tensor. Must be idempotent across calls.
        eager_ref: Zero-arg callable running the eager PyTorch reference
            on the SAME inputs ``our_fn`` uses. Returns the reference
            tensor.
        atol / rtol: Passed to ``torch.testing.assert_close`` for the
            correctness check.
        torch_compile_fn: Optional zero-arg callable running a
            ``torch.compile``'d version of ``eager_ref``. Timed when
            present; skipped otherwise.
        warmup: Iters discarded before timing.
        timed: Iters used for median-latency estimation.
    """
    assert torch.cuda.is_available(), "kernel_bench requires a CUDA device"

    # Correctness against eager.
    ours = our_fn()
    ref = eager_ref()
    torch.cuda.synchronize()
    diff = (ours.float() - ref.float()).abs()
    max_abs_err = float(diff.max().item())
    max_rel_err = float((diff / (ref.float().abs() + 1e-6)).max().item())
    try:
        torch.testing.assert_close(ours, ref, atol=atol, rtol=rtol)
        passed = True
    except AssertionError:
        passed = False

    # Latency — ours
    our_us = _cuda_event_time_us(our_fn, warmup=warmup, timed=timed)
    # Latency — eager
    eager_us = _cuda_event_time_us(eager_ref, warmup=warmup, timed=timed)
    # Latency — torch.compile (optional)
    tc_us: float | None = None
    if torch_compile_fn is not None:
        try:
            tc_us = _cuda_event_time_us(torch_compile_fn, warmup=warmup, timed=timed)
        except Exception:
            tc_us = None

    return BenchResult(
        name=name,
        device=str(torch.cuda.get_device_name(0)),
        dtype_in=str(ours.dtype).replace("torch.", ""),
        dtype_out=str(ref.dtype).replace("torch.", ""),
        max_abs_err=max_abs_err,
        max_rel_err=max_rel_err,
        passed=passed,
        our_us=our_us,
        eager_us=eager_us,
        torch_compile_us=tc_us,
        us_ratio_vs_eager=our_us / eager_us if eager_us > 0 else float("inf"),
        us_ratio_vs_torch_compile=(our_us / tc_us) if tc_us is not None and tc_us > 0 else None,
        warmup_iters=warmup,
        timed_iters=timed,
        input_shapes=input_shapes or [],
        notes=notes,
    )


def format_bench_result(r: BenchResult) -> str:
    """One-line tabular render."""
    tc_col = f"{r.torch_compile_us:>7.1f}" if r.torch_compile_us is not None else "    n/a"
    ratio_tc = (
        f"{r.us_ratio_vs_torch_compile:>5.2f}x" if r.us_ratio_vs_torch_compile is not None else "  n/a"
    )
    status = "PASS" if r.passed else "FAIL"
    return (
        f"[{status}] {r.name:45s}  "
        f"ours={r.our_us:>7.1f}μs  eager={r.eager_us:>7.1f}μs  "
        f"tc={tc_col}μs  "
        f"vs_eager={r.us_ratio_vs_eager:>5.2f}x  vs_tc={ratio_tc}  "
        f"max_abs_err={r.max_abs_err:.2e}"
    )


__all__ = ["BenchResult", "format_bench_result", "run_microbench"]
