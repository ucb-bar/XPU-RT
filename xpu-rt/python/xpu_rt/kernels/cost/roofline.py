"""Analytical roofline cost model.

Reference: Williams, Waterman, Patterson. "Roofline: An Insightful
Visual Performance Model for Multicore Architectures". CACM 2009.

The roofline bounds kernel throughput by ``min(peak_flops,
bandwidth × arithmetic_intensity)``, where arithmetic_intensity =
flops/bytes. For our compile-time cost estimation we invert this to
compute the **lower-bound latency** achievable by a kernel with the
given FLOPs + bytes-moved on a target with the given peak numbers.

This module is the ONLY supported way to synthesize a cost number when
measurement is unavailable. It always produces a grounded figure or
raises :class:`RooflineUnavailableError`. No placeholder ``0.0``.

Usage::

    from compgen.kernels.cost import predict
    from compgen.runtime.traits import DeviceTraits

    prediction = predict(contract, DeviceTraits.from_target_profile(profile))
    latency_us = prediction.latency_us
    regime = prediction.regime  # "compute-bound" | "memory-bound"

For fusion decisions::

    from compgen.kernels.cost import predict_fusion_speedup

    # Predicted speedup of a fused kernel over separate kernels, using
    # shared-memory reuse factor. Derived, not placeholder.
    speedup = predict_fusion_speedup(
        parts=[matmul_contract, gelu_contract],
        fused=matmul_gelu_contract,
        device_traits=traits,
    )
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from compgen.kernels.errors import RooflineUnavailableError
from compgen.kernels.measure import KernelMeasurement

if TYPE_CHECKING:
    from compgen.kernels.contracts import KernelContract
    from compgen.runtime.traits import DeviceTraits
    from compgen.targets.schema import TargetProfile


@dataclass(frozen=True)
class RooflinePrediction:
    """A roofline-derived latency prediction with provenance.

    The prediction is grounded in two peak numbers from the target:

    - ``peak_flops_per_s`` — from the sum of
      ``DeviceSpec.compute_units[i].count * peak_tflops * 1e12``.
    - ``peak_bandwidth_bps`` — the ``bandwidth_gbps * 1e9`` of the
      innermost memory level that at least one of the kernel's
      operands fits in, falling back to the outermost DRAM-level
      bandwidth when the operands don't fit anywhere in fast memory.

    ``regime`` is either ``"compute-bound"`` (arithmetic_intensity >=
    machine balance) or ``"memory-bound"``. The inverse of the
    binding rate gives ``latency_us``.
    """

    latency_us: float
    flops: int
    bytes_moved: int
    arithmetic_intensity: float
    peak_flops_per_s: float
    peak_bandwidth_bps: float
    regime: str
    memory_level: str
    source: str = "roofline"

    def as_measurement(self) -> KernelMeasurement:
        """Adapt to a :class:`KernelMeasurement` so downstream code
        that consumes measurements uniformly works on both paths."""
        return KernelMeasurement(
            latency_us=self.latency_us,
            latency_stddev_us=0.0,  # analytical, no variance
            bandwidth_gbps=self.peak_bandwidth_bps / 1e9,
            flops_per_s=self.peak_flops_per_s,
            device="analytical",
            iters=0,
            warmup=0,
            source="roofline",
        )


def roofline_latency_us(
    *,
    flops: int,
    bytes_moved: int,
    peak_flops_per_s: float,
    peak_bandwidth_bps: float,
) -> tuple[float, str]:
    """Core roofline formula. Returns ``(latency_us, regime)``.

    ``flops`` = kernel FLOPs. ``bytes_moved`` = bytes transferred across
    the limiting memory level. ``peak_flops_per_s`` and
    ``peak_bandwidth_bps`` are device peak rates in FLOP/s and B/s.

    Raises :class:`RooflineUnavailableError` when either peak is
    non-positive (target declares no capability → no grounded
    prediction).
    """
    if peak_flops_per_s <= 0.0:
        raise RooflineUnavailableError(
            f"peak_flops_per_s must be > 0, got {peak_flops_per_s}; declare peak_tflops on at least one ComputeUnit"
        )
    if peak_bandwidth_bps <= 0.0:
        raise RooflineUnavailableError(
            f"peak_bandwidth_bps must be > 0, got {peak_bandwidth_bps}; "
            "declare bandwidth_gbps on at least one MemoryLevel"
        )
    # Latency = max(flop_time, byte_time), expressed in seconds then µs.
    flop_time_s = flops / peak_flops_per_s if flops > 0 else 0.0
    byte_time_s = bytes_moved / peak_bandwidth_bps if bytes_moved > 0 else 0.0
    if flop_time_s == 0.0 and byte_time_s == 0.0:
        raise RooflineUnavailableError("contract declared zero flops and zero bytes_moved — no work to model")
    latency_s = max(flop_time_s, byte_time_s)
    regime = "compute-bound" if flop_time_s >= byte_time_s else "memory-bound"
    return latency_s * 1e6, regime


def predict(
    contract: KernelContract,
    device_traits: DeviceTraits | None = None,
    *,
    target_profile: TargetProfile | None = None,
    device_index: int = 0,
) -> RooflinePrediction:
    """Predict kernel latency from a contract + device capability.

    One of ``device_traits`` or ``target_profile`` must be supplied.
    Prefers ``device_traits`` when both are present (those are the
    live-probe-preferred values).

    Args:
        contract: The kernel contract; its ``cost`` drives the
            prediction.
        device_traits: Traits derived from a live probe.
        target_profile: Declared profile. Used to extract peak_flops +
            peak_bandwidth when traits aren't supplied.
        device_index: Which device in the profile's ``devices`` list
            to model for (default 0).

    Returns:
        :class:`RooflinePrediction`.

    Raises:
        RooflineUnavailableError: neither source declares usable peak
            numbers.
    """
    flops = int(contract.cost.flops)
    bytes_moved = int(contract.cost.bytes_read) + int(contract.cost.bytes_written)

    peak_flops, peak_bandwidth, mem_level_name = _resolve_peaks(
        device_traits=device_traits,
        target_profile=target_profile,
        device_index=device_index,
        bytes_needed=bytes_moved,
    )

    latency_us, regime = roofline_latency_us(
        flops=flops,
        bytes_moved=bytes_moved,
        peak_flops_per_s=peak_flops,
        peak_bandwidth_bps=peak_bandwidth,
    )

    ai = float(flops) / float(bytes_moved) if bytes_moved > 0 else 0.0

    return RooflinePrediction(
        latency_us=latency_us,
        flops=flops,
        bytes_moved=bytes_moved,
        arithmetic_intensity=ai,
        peak_flops_per_s=peak_flops,
        peak_bandwidth_bps=peak_bandwidth,
        regime=regime,
        memory_level=mem_level_name,
    )


def _resolve_peaks(
    *,
    device_traits: DeviceTraits | None,
    target_profile: TargetProfile | None,
    device_index: int,
    bytes_needed: int,
) -> tuple[float, float, str]:
    """Return (peak_flops_per_s, peak_bandwidth_bps, memory_level_name).

    Tries ``device_traits`` first (it carries live-probed values when
    they disagree with the profile), falls back to
    ``target_profile.devices[device_index]`` for peak_flops and the
    best-fitting memory level for bandwidth.
    """
    peak_flops = 0.0
    peak_bandwidth = 0.0
    mem_level = "unknown"

    if device_traits is not None:
        # DeviceTraits may carry peak numbers in ``metadata`` for
        # probed devices; honour them if present.
        meta = getattr(device_traits, "metadata", {}) or {}
        if "peak_flops_per_s" in meta:
            peak_flops = float(meta["peak_flops_per_s"])
        if "peak_bandwidth_bps" in meta:
            peak_bandwidth = float(meta["peak_bandwidth_bps"])
            mem_level = str(meta.get("peak_bandwidth_level", "device_traits"))

    if (peak_flops <= 0.0 or peak_bandwidth <= 0.0) and target_profile is not None:
        devices = target_profile.devices
        if not devices or device_index >= len(devices):
            raise RooflineUnavailableError(
                f"target_profile has no device at index {device_index}; found {len(devices)} device(s)"
            )
        device = devices[device_index]

        if peak_flops <= 0.0:
            flops_sum = 0.0
            for cu in device.compute_units:
                if cu.peak_tflops is not None and cu.peak_tflops > 0.0:
                    flops_sum += cu.count * cu.peak_tflops * 1e12
            peak_flops = flops_sum

        if peak_bandwidth <= 0.0:
            # Prefer the finest-grain memory level whose size fits the
            # operand footprint. Fall back to the outermost level
            # (DRAM) when nothing fits.
            candidate = _pick_memory_level(device.memory_hierarchy, bytes_needed)
            if candidate is not None:
                peak_bandwidth = float(candidate.bandwidth_gbps or 0.0) * 1e9
                mem_level = candidate.name

    if peak_flops <= 0.0:
        raise RooflineUnavailableError(
            "no peak FLOPS declared on device_traits or target_profile; roofline prediction requires a grounded peak"
        )
    if peak_bandwidth <= 0.0:
        raise RooflineUnavailableError(
            "no peak bandwidth declared on device_traits or target_profile; "
            "roofline prediction requires a grounded peak"
        )
    return peak_flops, peak_bandwidth, mem_level


def _pick_memory_level(levels, bytes_needed: int):  # noqa: ANN001
    """Pick the finest-grain memory level whose size fits ``bytes_needed``.

    Assumes ``levels`` is ordered fastest→slowest (the schema's
    documented convention). Skips levels with unknown bandwidth.
    """
    last_with_bandwidth = None
    for level in levels:
        if level.bandwidth_gbps is None or level.bandwidth_gbps <= 0.0:
            continue
        last_with_bandwidth = level
        if bytes_needed <= 0 or level.size_bytes >= bytes_needed:
            return level
    return last_with_bandwidth


def predict_fusion_speedup(
    *,
    parts: list[KernelContract],
    fused: KernelContract,
    device_traits: DeviceTraits | None = None,
    target_profile: TargetProfile | None = None,
    device_index: int = 0,
) -> float:
    """Predict speedup of ``fused`` over running ``parts`` separately.

    Derivation (not placeholder): sum the individual roofline
    latencies, divide by the fused roofline latency.

    Returns the ratio ``separate_latency_us / fused_latency_us``.
    Values > 1 mean fusion wins; < 1 mean separate kernels would be
    faster (fusion is a pessimization — rare, but possible when
    register pressure eliminates compute overlap).

    Raises :class:`RooflineUnavailableError` on missing peaks.
    """
    if not parts:
        raise ValueError("parts must be non-empty to compute fusion speedup")
    separate_latency_us = 0.0
    for p in parts:
        pred = predict(p, device_traits, target_profile=target_profile, device_index=device_index)
        separate_latency_us += pred.latency_us
    fused_pred = predict(fused, device_traits, target_profile=target_profile, device_index=device_index)
    if fused_pred.latency_us <= 0.0:
        raise RooflineUnavailableError("fused contract produced zero latency — malformed cost")
    return separate_latency_us / fused_pred.latency_us


__all__ = [
    "RooflinePrediction",
    "predict",
    "predict_fusion_speedup",
    "roofline_latency_us",
]
