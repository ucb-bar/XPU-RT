"""Production-grade kernel measurement.

Every placeholder ``latency_us = 0.0`` in the codebase is a bug — it
makes selectors compare real latencies against a lie. This module
provides the single blessed way to measure a kernel's cost:

- **GPU path**: ``torch.cuda.Event`` pairs bracketing
  ``torch.cuda.synchronize()``. Accurate to the microsecond; the
  event timers return milliseconds so we convert.
- **CPU path**: ``time.perf_counter_ns`` with warmup + averaged iters.
- **Refuses to guess**: if we can't run the kernel (no callable, no
  golden inputs, no compatible device), :class:`UnmeasurableKernelError`.

Every measurement carries:

- ``latency_us``: mean latency across ``iters`` timed runs, after
  ``warmup`` un-timed runs.
- ``latency_stddev_us``: sample stddev so callers can see noise.
- ``bandwidth_gbps``: derived from ``contract.cost.bytes_read +
  bytes_written`` divided by mean latency. 0.0 if bytes unknown.
- ``flops_per_s``: derived from ``contract.cost.flops`` / mean
  latency. 0.0 if FLOPS unknown.
- ``device``: the device the measurement ran on ("cuda:0", "cpu").

Typical usage::

    from compgen.kernels.measure import measure_kernel
    from compgen.kernels.errors import UnmeasurableKernelError

    try:
        m = measure_kernel(
            runnable=compiled_triton_kernel,
            contract=contract,
            golden_inputs=(x, y),
            warmup=5,
            iters=50,
        )
        provider_result.latency_us = m.latency_us
    except UnmeasurableKernelError:
        # Fall back to analytical roofline; never set 0.0.
        m = roofline.predict(contract, device_traits)
        provider_result.latency_us = m.latency_us
"""

from __future__ import annotations

import math
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from compgen.kernels.errors import UnmeasurableKernelError

if TYPE_CHECKING:
    from compgen.kernels.contracts import KernelContract


@dataclass(frozen=True)
class KernelMeasurement:
    """One measurement of a kernel's cost.

    ``source`` captures how the number was obtained so downstream
    reports can say "measured" vs "analytical" instead of lying about
    provenance.
    """

    latency_us: float
    latency_stddev_us: float = 0.0
    bandwidth_gbps: float = 0.0
    flops_per_s: float = 0.0
    device: str = ""
    iters: int = 0
    warmup: int = 0
    source: str = "unknown"  # "measured_gpu" | "measured_cpu" | "roofline" | ...


def measure_kernel(
    *,
    runnable: Callable[..., Any],
    contract: KernelContract | None = None,
    golden_inputs: tuple[Any, ...] | None = None,
    warmup: int = 5,
    iters: int = 50,
    device: str | None = None,
) -> KernelMeasurement:
    """Measure ``runnable(*golden_inputs)`` and return the timings.

    Args:
        runnable: A Python callable. For Triton kernels this is the
            compiled ``@triton.jit`` function (after grid binding);
            for torch paths this is a module's ``forward``, a compiled
            function, or any callable. Must accept ``*golden_inputs``.
        contract: Optional :class:`KernelContract`. When supplied, its
            ``cost.flops`` / ``bytes_read`` / ``bytes_written`` fields
            populate ``flops_per_s`` and ``bandwidth_gbps`` in the
            measurement.
        golden_inputs: Arguments to call ``runnable`` with. Required.
        warmup: Number of un-timed runs before measurement begins.
            Absorbs JIT warmup, CUDA context lazy-init, cache warming.
        iters: Number of timed runs contributing to the mean + stddev.
        device: Target device ("cuda:0", "cuda:1", "cpu"). When ``None``
            we pick "cuda:0" if CUDA is available AND any tensor in
            ``golden_inputs`` is on CUDA, else "cpu".

    Returns:
        :class:`KernelMeasurement`.

    Raises:
        UnmeasurableKernelError: ``runnable`` is not callable, or
            ``golden_inputs`` is missing, or CUDA is requested but
            unavailable, or the kernel raises during warmup.
    """
    if not callable(runnable):
        raise UnmeasurableKernelError(
            f"runnable is not callable (got {type(runnable).__name__!r}); "
            "pass a compiled kernel, nn.Module, or plain function"
        )
    if golden_inputs is None:
        raise UnmeasurableKernelError("measure_kernel requires golden_inputs; cannot time a kernel with no inputs")
    if iters < 1:
        raise ValueError(f"iters must be >= 1, got {iters}")
    if warmup < 0:
        raise ValueError(f"warmup must be >= 0, got {warmup}")

    resolved_device = device or _pick_device(golden_inputs)
    use_cuda = resolved_device.startswith("cuda")
    if use_cuda and not torch.cuda.is_available():
        raise UnmeasurableKernelError(f"device={resolved_device!r} requested but torch.cuda.is_available() is False")

    # ---------------- Warmup (un-timed) ----------------
    try:
        for _ in range(warmup):
            runnable(*golden_inputs)
        if use_cuda:
            torch.cuda.synchronize()
    except Exception as exc:
        raise UnmeasurableKernelError(f"kernel raised during warmup: {exc!r}") from exc

    # ---------------- Timed iterations ----------------
    samples_us: list[float] = []
    try:
        if use_cuda:
            samples_us = _time_cuda(runnable, golden_inputs, iters)
        else:
            samples_us = _time_cpu(runnable, golden_inputs, iters)
    except Exception as exc:
        raise UnmeasurableKernelError(f"kernel raised during timed run: {exc!r}") from exc

    mean_us = statistics.fmean(samples_us)
    stddev_us = statistics.pstdev(samples_us) if len(samples_us) > 1 else 0.0

    flops_per_s = 0.0
    bandwidth_gbps = 0.0
    if contract is not None and mean_us > 0.0:
        flops = int(contract.cost.flops) if contract.cost.flops > 0 else 0
        bytes_total = int(contract.cost.bytes_read) + int(contract.cost.bytes_written)
        if flops > 0:
            flops_per_s = flops / (mean_us * 1e-6)
        if bytes_total > 0:
            bandwidth_gbps = (bytes_total / (mean_us * 1e-6)) / 1e9

    return KernelMeasurement(
        latency_us=mean_us,
        latency_stddev_us=stddev_us,
        bandwidth_gbps=bandwidth_gbps,
        flops_per_s=flops_per_s,
        device=resolved_device,
        iters=iters,
        warmup=warmup,
        source="measured_gpu" if use_cuda else "measured_cpu",
    )


def _pick_device(golden_inputs: tuple[Any, ...]) -> str:
    """Pick a device from the inputs. Prefers CUDA when any input is on it."""
    for x in golden_inputs:
        if isinstance(x, torch.Tensor) and x.is_cuda:
            return str(x.device)
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


def _time_cuda(runnable: Callable[..., Any], inputs: tuple[Any, ...], iters: int) -> list[float]:
    """Time ``iters`` runs using ``torch.cuda.Event``.

    One event pair per iteration — the paper-standard pattern. We
    synchronise once at the end to flush the stream, then read each
    pair's elapsed time. Event timing is reported in milliseconds at
    0.5 µs resolution; convert to µs.
    """
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        runnable(*inputs)
        ends[i].record()
    torch.cuda.synchronize()
    return [float(starts[i].elapsed_time(ends[i])) * 1000.0 for i in range(iters)]


def _time_cpu(runnable: Callable[..., Any], inputs: tuple[Any, ...], iters: int) -> list[float]:
    """Time ``iters`` runs using ``time.perf_counter_ns``."""
    out: list[float] = []
    for _ in range(iters):
        t0 = time.perf_counter_ns()
        runnable(*inputs)
        t1 = time.perf_counter_ns()
        out.append((t1 - t0) / 1000.0)  # ns → µs
    return out


def iqr_filtered(samples_us: list[float]) -> list[float]:
    """Drop samples outside the IQR×1.5 fence.

    Useful when the first few "warmup" iterations bleed into timing
    because CUDA context init or kernel auto-tuning happened mid-run.
    Not used by default; exposed for callers that want robust-mean
    semantics.
    """
    if len(samples_us) < 4:
        return list(samples_us)
    sorted_s = sorted(samples_us)
    q1 = sorted_s[len(sorted_s) // 4]
    q3 = sorted_s[(3 * len(sorted_s)) // 4]
    iqr = q3 - q1
    lo = q1 - 1.5 * iqr
    hi = q3 + 1.5 * iqr
    kept = [x for x in samples_us if lo <= x <= hi]
    return kept or list(samples_us)


def speedup(baseline: KernelMeasurement, candidate: KernelMeasurement) -> float:
    """``baseline.latency_us / candidate.latency_us``.

    0.0 baseline or candidate is treated as unknown and returns NaN —
    downstream code must check ``math.isnan`` and NOT substitute a 1.0.
    """
    if baseline.latency_us <= 0.0 or candidate.latency_us <= 0.0:
        return math.nan
    return baseline.latency_us / candidate.latency_us


__all__ = [
    "KernelMeasurement",
    "iqr_filtered",
    "measure_kernel",
    "speedup",
]
