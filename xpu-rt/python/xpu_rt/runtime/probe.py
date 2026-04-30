"""Live device-traits probe — Phase 6.

Probes a CUDA device's capability bits + peak rates and returns a
JSON-serialisable dict suitable for
:meth:`compgen.runtime.traits.DeviceTraits.with_probe`. The probe is
the bridge between the static :class:`TargetProfile` YAML (compile
time) and the actual silicon under the runtime (compile-and-run time).

Two implementations:

- :func:`probe_via_torch` — uses ``torch.cuda.get_device_properties``
  + a small ``cudaDeviceGetAttribute`` shim through ``cuda.bindings``
  (when ``cuda-python`` is installed). Available now; works without
  the Phase-4 native HAL.
- :func:`probe_via_native_hal` — placeholder for the C-side
  ``cg_rt_cuda_probe_device`` primitive landing in Phase 4. Returns
  the same dict shape as ``probe_via_torch`` so callers don't
  branch.

The single-entry helper :func:`probe_cuda_device` picks the best
available implementation and falls back to torch when the native HAL
isn't present.

Output shape (every key optional — populated when probable on the
host):

```python
{
    # Identity
    "device_name": str,
    "compute_capability_major": int,
    "compute_capability_minor": int,

    # Counts + sizes
    "sm_count": int,
    "max_threads_per_block": int,
    "max_threads_per_multiprocessor": int,
    "max_shared_memory_per_block_optin_bytes": int,
    "max_grid_dim_x": int,
    "max_grid_dim_y": int,
    "max_grid_dim_z": int,
    "warp_size": int,
    "l2_cache_bytes": int,

    # Architectural Blackwell / Hopper booleans
    "supports_clusters": bool,
    "supports_tma": bool,
    "supports_fp8": bool,
    "supports_fp4": bool,
    "supports_ondevice_scheduler": bool,

    # Roofline-cost inputs
    "peak_flops_per_s": float,           # estimated from clock × cores
    "peak_bandwidth_bps": float,         # global memory bandwidth
    "peak_bandwidth_level": str,         # "hbm" / "gddr7" / "gddr6x"

    # Topology
    "interconnect_topology": str,        # "nvlink" | "pcie" | "shared_memory"
    "interconnect_bandwidth_gbps": float,
    "num_visible_devices": int,

    # Provenance
    "probe_source": str,                 # "torch" | "native_hal" | "fallback"
    "driver_version": str,
    "runtime_version": str,
}
```

Phase-4's C probe will populate the same keys; downstream code
(Phase 5 emitter, Phase 2 cost model, conformance harness) reads
this dict via ``DeviceTraits.metadata``.
"""

from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger(__name__)


# Architectural-feature gate by compute capability major. These mirror
# the values the YAML profiles encode statically and the probe sets
# dynamically; kept here so probe + YAML stay in lockstep.
_CC_FEATURE_GATES = {
    "supports_tma": 9,  # Hopper+
    "supports_clusters": 9,  # Hopper+
    "supports_fp8": 9,  # Hopper+ (e4m3 / e5m2)
    "supports_fp4": 10,  # Datacenter Blackwell+ (sm_100/sm_120)
    "supports_ondevice_scheduler": 9,  # Hopper+ (cooperative groups + atomics)
}


def probe_cuda_device(device_index: int = 0) -> dict[str, Any]:
    """Pick the best-available probe and return its result.

    Order: native HAL (Phase 4 ``cg_rt_cuda_probe_device``) → torch
    fallback. Always returns a dict; on a CPU host returns a
    minimally-populated record with ``probe_source="fallback"`` and a
    reason in ``probe_error``.
    """
    try:
        return probe_via_native_hal(device_index)
    except _NativeHalUnavailable:
        pass
    return probe_via_torch(device_index)


def probe_via_torch(device_index: int = 0) -> dict[str, Any]:
    """Probe using ``torch.cuda`` + an optional ``cuda.bindings`` shim.

    Returns the canonical probe dict (see module docstring). Missing
    bits land as ``0``, ``False``, or ``""`` — callers consume via
    :meth:`DeviceTraits.with_probe` which treats the probe values as
    **overriding** profile-derived ones, so omitted keys preserve the
    profile's values.
    """
    out: dict[str, Any] = {"probe_source": "torch"}
    try:
        import torch
    except Exception as exc:
        out["probe_source"] = "fallback"
        out["probe_error"] = f"torch not importable: {exc!r}"
        return out

    if not torch.cuda.is_available():
        out["probe_source"] = "fallback"
        out["probe_error"] = "torch.cuda.is_available() is False"
        return out

    if device_index >= torch.cuda.device_count():
        out["probe_source"] = "fallback"
        out["probe_error"] = f"device_index={device_index} but only {torch.cuda.device_count()} CUDA device(s) visible"
        return out

    props = torch.cuda.get_device_properties(device_index)
    cc_major = int(props.major)
    cc_minor = int(props.minor)

    out["device_name"] = props.name
    out["compute_capability_major"] = cc_major
    out["compute_capability_minor"] = cc_minor
    out["sm_count"] = int(props.multi_processor_count)
    out["num_visible_devices"] = torch.cuda.device_count()
    out["max_threads_per_block"] = int(getattr(props, "max_threads_per_multi_processor", 0))
    out["max_threads_per_multiprocessor"] = int(getattr(props, "max_threads_per_multi_processor", 0))
    out["warp_size"] = int(getattr(props, "warp_size", 32))

    # torch.version.cuda is the toolkit baseline; live driver version
    # via cuda.bindings when available.
    out["runtime_version"] = str(getattr(torch.version, "cuda", ""))

    # Architectural booleans from compute capability.
    for key, threshold in _CC_FEATURE_GATES.items():
        out[key] = cc_major >= threshold

    # Memory bandwidth + global memory size — torch exposes total mem
    # but not bandwidth; device-name + cc give us a first cut.
    total_mem_bytes = int(getattr(props, "total_memory", 0))
    out["max_device_memory_bytes"] = total_mem_bytes
    out.update(_estimate_peak_bandwidth(props.name, cc_major, total_mem_bytes))
    out.update(_estimate_peak_flops(props, cc_major))
    out.update(_infer_interconnect(props.name, out["num_visible_devices"]))

    # Try cuda.bindings for the higher-fidelity attribute set.
    out.update(_probe_via_cuda_bindings(device_index))

    return out


class _NativeHalUnavailable(RuntimeError):
    """Raised by :func:`probe_via_native_hal` when the C probe isn't
    on this build. Internal — callers go through :func:`probe_cuda_device`."""


def probe_via_native_hal(device_index: int = 0) -> dict[str, Any]:
    """Probe via the Phase-4 ``cg_rt_cuda_probe_device`` C primitive.

    Wraps :class:`compgen.runtime.native.cuda.CudaDeviceProbe`. Raises
    :class:`_NativeHalUnavailable` when:

    - the CUDA-built ``libcompgen_rt`` isn't on this install (wheel
      built without ``make build-cuda-rt``), or
    - the C API returns an error (CUDA driver missing, no devices, etc).

    On success returns a dict with ``probe_source="native_hal"`` and
    the same field set as :func:`probe_via_torch` plus the
    higher-fidelity attributes (``cluster_launch``, ``cooperative_launch``,
    ``concurrent_kernels``, ``concurrent_managed_access``).
    """
    try:
        from compgen.runtime.native.cuda import (
            CudaDeviceProbe,
            CudaUnavailableError,
        )
    except Exception as exc:
        raise _NativeHalUnavailable(f"compgen.runtime.native.cuda not importable: {exc!r}") from exc
    try:
        return CudaDeviceProbe().probe(device_index)
    except CudaUnavailableError as exc:
        raise _NativeHalUnavailable(str(exc)) from exc


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


# Conservative peak-bandwidth lookup by (name fragment, cc_major).
# Sources: NVIDIA spec sheets + datacenter / workstation product
# pages as of 2026-04. When the device name doesn't match any
# pattern, fall back to a low-bound ``0.0`` and let the caller
# detect via the boolean ``peak_bandwidth_bps > 0``.
_BW_TABLE = (
    ("B200", 8.0e12, "hbm3e"),  # 8 TB/s
    ("B100", 7.7e12, "hbm3e"),
    ("H200", 4.8e12, "hbm3e"),
    ("H100", 3.35e12, "hbm3"),
    ("RTX PRO 6000 Blackwell", 1.79e12, "gddr7"),  # 1.79 TB/s, 96 GB GDDR7
    ("RTX 5090", 1.79e12, "gddr7"),
    ("L40", 0.864e12, "gddr6"),
    ("L4", 0.3e12, "gddr6"),
    ("A100", 1.555e12, "hbm2e"),
    ("V100", 0.9e12, "hbm2"),
)


def _estimate_peak_bandwidth(name: str, cc_major: int, total_mem: int) -> dict[str, Any]:
    name_l = name.lower()
    for fragment, bps, level in _BW_TABLE:
        if fragment.lower() in name_l:
            return {
                "peak_bandwidth_bps": float(bps),
                "peak_bandwidth_level": level,
            }
    # No name match — leave 0.0 + an empty level so callers can detect.
    return {"peak_bandwidth_bps": 0.0, "peak_bandwidth_level": ""}


def _estimate_peak_flops(props: Any, cc_major: int) -> dict[str, Any]:
    """Best-effort peak FLOPS estimate from SM count + compute capability.

    Datacenter Blackwell (sm_100) tensor-core FP8 peak ≈ 18.9 TFLOPS/SM
    (B200 has 132 SMs ⇒ ~2500 TFLOPS = 2.5 PFLOPS). Workstation
    Blackwell (sm_120) tensor-core FP8 peak ≈ 17 TFLOPS/SM
    (RTX PRO 6000 has 188 SMs ⇒ ~3200 TFLOPS).

    For Hopper (sm_90) tensor-core FP8 peak ≈ 15 TFLOPS/SM
    (H100 132 SMs ⇒ ~1979 TFLOPS).

    These are conservative; the real numbers depend on clock + power
    limits. Phase-4's C probe will read ``cudaDevAttrClockRate`` for
    a tighter estimate.
    """
    sm_count = int(getattr(props, "multi_processor_count", 0))
    if sm_count == 0 or cc_major < 7:
        return {"peak_flops_per_s": 0.0}

    if cc_major >= 12:  # workstation Blackwell (sm_120)
        per_sm = 17.0e12
    elif cc_major >= 10:  # datacenter Blackwell (sm_100)
        per_sm = 18.9e12
    elif cc_major >= 9:  # Hopper (sm_90)
        per_sm = 15.0e12
    elif cc_major >= 8:  # Ampere (sm_80/sm_86)
        per_sm = 4.9e12  # FP8 not native; use BF16 tensor-core peak.
    else:  # Volta / Turing (sm_70/sm_75)
        per_sm = 1.0e12
    return {"peak_flops_per_s": per_sm * sm_count}


def _infer_interconnect(name: str, num_devices: int) -> dict[str, Any]:
    """Infer GPU-to-GPU interconnect from form factor.

    Datacenter cards (B200/B100/H200/H100/A100/V100) → NVLink/NVSwitch.
    Workstation Blackwell (RTX PRO 6000) → PCIe (NVLink dropped from
    workstation segment per RTX PRO 6000 announcement).
    Single-GPU host → "single".
    """
    if num_devices < 2:
        return {"interconnect_topology": "single", "interconnect_bandwidth_gbps": 0.0}
    name_l = name.lower()
    is_workstation = any(tag in name_l for tag in ("rtx", "workstation"))
    is_datacenter = any(tag in name_l for tag in ("b200", "b100", "h200", "h100", "a100", "v100"))
    if is_datacenter:
        return {"interconnect_topology": "nvlink", "interconnect_bandwidth_gbps": 900.0}
    if is_workstation:
        # PCIe Gen5 x16 = 64 GB/s effective. RTX PRO 6000 has no
        # NVLink, just PCIe.
        return {"interconnect_topology": "pcie", "interconnect_bandwidth_gbps": 64.0}
    return {"interconnect_topology": "unknown", "interconnect_bandwidth_gbps": 0.0}


def _probe_via_cuda_bindings(device_index: int) -> dict[str, Any]:
    """Optional higher-fidelity probe via ``cuda-python``.

    When the package is installed (it is in the ``[cuda]`` extra), we
    read ``cudaDevAttrClusterLaunch``, ``cudaDevAttrMaxBlocksPerCluster``,
    ``cudaDevAttrL2CacheSize``, etc. — the bits torch doesn't surface.

    Returns an empty dict when the bindings aren't importable; the
    torch path still produced the cc-derived booleans so the caller
    has a usable record either way.
    """
    out: dict[str, Any] = {}
    try:
        # cuda-python 13.x layout.
        from cuda.bindings import driver, runtime  # type: ignore[import-not-found]
    except Exception:
        try:
            # cuda-python 12.x layout.
            from cuda import cuda as driver  # type: ignore[import-not-found]
            from cuda import cudart as runtime  # type: ignore[import-not-found]
        except Exception:
            return out  # No bindings; torch values stand.

    try:
        # Attribute IDs vary slightly between bindings versions.
        # Probe the canonical names; missing ones silently skip.
        attr_map = {
            "max_blocks_per_cluster": "cudaDevAttrMaxBlocksPerCluster",
            "cluster_launch": "cudaDevAttrClusterLaunch",
            "l2_cache_bytes": "cudaDevAttrL2CacheSize",
            "max_grid_dim_x": "cudaDevAttrMaxGridDimX",
            "max_grid_dim_y": "cudaDevAttrMaxGridDimY",
            "max_grid_dim_z": "cudaDevAttrMaxGridDimZ",
            "max_threads_per_block": "cudaDevAttrMaxThreadsPerBlock",
            "max_shared_memory_per_block_optin_bytes": "cudaDevAttrMaxSharedMemoryPerBlockOptin",
        }
        for out_key, attr_name in attr_map.items():
            attr_enum = _resolve_devattr(runtime, attr_name)
            if attr_enum is None:
                continue
            value = _safe_get_attr(runtime, attr_enum, device_index)
            if value is not None:
                out[out_key] = int(value)

        # Driver version (live).
        if hasattr(driver, "cuDriverGetVersion"):
            try:
                rc, ver = driver.cuDriverGetVersion()
                if int(rc) == 0:
                    out["driver_version"] = str(int(ver))
            except Exception:
                pass

        # If we got cluster_launch back, refine the cc-derived
        # supports_clusters with the real answer (sm_120 fuzziness
        # was the whole reason we pinned this for runtime probing).
        if "cluster_launch" in out:
            out["supports_clusters"] = bool(out["cluster_launch"])
            out["supports_ondevice_scheduler"] = bool(out["cluster_launch"])
    except Exception as exc:
        log.debug("cuda.bindings probe partial: %s", exc)
    return out


def _resolve_devattr(runtime: Any, name: str) -> Any | None:
    """Look up a ``cudaDevAttr*`` enum across bindings versions."""
    for attr_holder_name in (
        "cudaDeviceAttr",  # cuda-python 12 cudart
        "DeviceAttribute",  # cuda-python 13 runtime
        "cudaDevAttr",
    ):
        holder = getattr(runtime, attr_holder_name, None)
        if holder is None:
            continue
        candidate = getattr(holder, name.replace("cudaDevAttr", ""), None) or getattr(holder, name, None)
        if candidate is not None:
            return candidate
    return None


def _safe_get_attr(runtime: Any, attr_enum: Any, device: int) -> int | None:
    """Call ``cudaDeviceGetAttribute`` across bindings versions."""
    fn = getattr(runtime, "cudaDeviceGetAttribute", None) or getattr(runtime, "cuDeviceGetAttribute", None)
    if fn is None:
        return None
    try:
        rc, value = fn(attr_enum, device)
        if int(rc) == 0:
            return int(value)
    except Exception:
        return None
    return None


__all__ = [
    "probe_cuda_device",
    "probe_via_native_hal",
    "probe_via_torch",
]
