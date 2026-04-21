"""Bridge ``compgen.targets.TargetProfile`` → ``HardwareEnvelope``.

The YAML-loaded :class:`TargetProfile` is CompGen's source-of-truth for
hardware specs. The :class:`HardwareEnvelope` is the kernel-facing
read-through that lives inside a :class:`KernelContractV3`. Today
envelopes are hand-constructed in ``contract_v3_references.py`` for
demo purposes; this bridge derives them automatically from the target
profile so real compiles stay drift-free as YAML specs evolve.

The bridge also attaches **codegen hints** — authored-per-target
strings the kernel codegen (Claude Code) reads as prompt context.
Think of these as the autocomp ``get_hw_config_specific_rules``
equivalent, but structured and owned by us:

    cuda_a100:        "Use tl.dot with bf16 inputs + f32 accumulate ..."
    openq_5165rb:     "Hexagon HVX vectors are 128 bytes ..."
    cpu_host:         "Fall back to stdlib math; fast_math controls fma"
    trainium1:        "NKI expects 128×512 blocks; cache_mode='streaming' ..."
    riscv_soc:        "RVV vector register group LMUL=2 for matmul ..."
"""

from __future__ import annotations

from compgen.kernels.contract_v3 import HardwareEnvelope
from compgen.targets.schema import ComputeUnit, DeviceSpec, MemoryLevel, TargetProfile


# ---------------------------------------------------------------------------
# Per-target codegen hints — authored; short list per target
# ---------------------------------------------------------------------------


CODEGEN_HINTS: dict[str, tuple[str, ...]] = {
    # NVIDIA A100 (cuda_a100.yaml)
    "cuda-a100": (
        "Use ``tl.dot(..., allow_tf32=False)`` with bf16/f16 inputs and f32 accumulator for the TensorCore path.",
        "Shared-memory (SMEM) is 164 KB per SM — keep tiles ≤128 KB to leave margin for double-buffering.",
        "Prefer ``cp.async`` (triton ``tl.load`` with ``eviction_policy='evict_first'``) for >=1 KB copies; synchronous LDS for smaller.",
        "Bank conflicts: pad LDS by 8B on the inner stride when K is a multiple of 32 and dtype is 4B.",
        "Warps are 32-wide; align ``BLOCK_N`` to 32 for coalesced 128-byte stores.",
    ),
    # openq_5165rb (Hexagon-style NPU)
    "openq_5165rb": (
        "HVX vector register is 128 bytes (1024 bits, 128 × i8 / 64 × i16 / 32 × i32).",
        "Use ``vmpyubacc`` for int8×int8→int32 accumulated matmul inner loop; one instruction per 128 MACs.",
        "VTCM (scratchpad) is 8 MB — keep working sets ≤1 MB per slice to allow double-buffer + DMA in flight.",
        "DMA engine is async; issue ``memcpy_async`` then do work; poll on event tensor before touching target buffer.",
        "f16 / bf16 are native on HVX v69+; fp32 is emulated and ~4× slower — keep acc in f16 when tolerance allows.",
    ),
    # CPU host (generic)
    "cpu-host": (
        "Use stdlib ``math.*`` for transcendentals; no tensor-core / intrinsic path available.",
        "``fast_math`` maps to reorder-tolerant FP code (associativity + reciprocal-mul instead of div).",
        "Prefer AoS layout for cache-line packing on sequential scans; SoA for gather/scatter.",
        "Align buffers to 64 B (L1 line); misaligned loads split into 2 µops on modern x86.",
    ),
    # AWS Trainium (trainium1.yaml)
    "trainium1": (
        "NKI expects 128×512 tile blocks on the primary tensor engine; smaller blocks waste cycles.",
        "Set ``cache_mode='streaming'`` on activation tensors that fit on-chip once — avoids HBM roundtrip.",
        "bf16 is the native dtype; f32 is emulated and ~2× slower.",
        "Use ``nki.iota`` for arange-style index tensors; avoid constructing them as tensors on host.",
    ),
    # RISC-V SoC with Vector Extension (riscv_soc.yaml)
    "riscv-soc": (
        "RVV LMUL=2 doubles vector register-group width; use for matmul inner loop, LMUL=1 for reductions.",
        "No hardware transpose; emit explicit strided loads via ``vle32.v`` with stride parameter.",
        "f16 requires Zfh extension — check target.features before emitting f16 intrinsics.",
        "VLEN is configurable (typically 128 bits on this SoC); query at runtime via ``vsetvli`` if variable.",
    ),
}


# ---------------------------------------------------------------------------
# Bridge
# ---------------------------------------------------------------------------


def _pick_primary_compute_unit(device: DeviceSpec) -> ComputeUnit | None:
    """Prefer a tensor-core / matmul-accelerator unit, fall back to the
    first compute unit defined on the device."""
    if not device.compute_units:
        return None
    preferred_names = ("tensor_core", "mma", "tpu", "vpu", "gemm")
    for unit in device.compute_units:
        if any(p in unit.name.lower() for p in preferred_names):
            return unit
    return device.compute_units[0]


def _scratchpad_bytes(device: DeviceSpec) -> int:
    """Return scratchpad-equivalent memory size (shared mem / VTCM / L1).

    Walks ``memory_hierarchy`` looking for the most-on-chip tier.
    """
    preferred = ("shared_memory", "scratchpad", "vtcm", "sram", "local_memory", "l1")
    candidates: list[MemoryLevel] = []
    for level in device.memory_hierarchy:
        lname = level.name.lower()
        if any(p in lname for p in preferred):
            candidates.append(level)
    if candidates:
        # Pick the smallest (on-chip tier is the one the kernel writes into).
        return min(c.size_bytes for c in candidates)
    # Fall back to the smallest memory tier overall.
    if device.memory_hierarchy:
        return min(m.size_bytes for m in device.memory_hierarchy)
    return 0


def _register_bytes(device: DeviceSpec) -> int:
    """Per-thread register file bytes. Looks for a tier named "registers"."""
    for level in device.memory_hierarchy:
        if "register" in level.name.lower():
            return level.size_bytes
    # Some profiles don't model registers explicitly; default to 256 B
    # (64 × 32-bit) which is a safe lower bound for most ISAs.
    return 256


def _peak_bandwidth_gbps(device: DeviceSpec) -> float:
    """Bandwidth of the outermost (DRAM) tier."""
    if not device.memory_hierarchy:
        return 0.0
    # Pick the memory level with the largest size_bytes; that's DRAM/HBM.
    outer = max(device.memory_hierarchy, key=lambda m: m.size_bytes)
    return outer.bandwidth_gbps or 0.0


def _native_dtypes(device: DeviceSpec) -> tuple[str, ...]:
    """Union of supported_dtypes across compute units, de-duplicated."""
    seen: set[str] = set()
    ordered: list[str] = []
    for unit in device.compute_units:
        for dt in unit.supported_dtypes:
            if dt not in seen:
                seen.add(dt)
                ordered.append(dt)
    return tuple(ordered)


def envelope_from_target_profile(
    profile: TargetProfile,
    *,
    device_index: int = 0,
    extra_hints: tuple[str, ...] = (),
) -> HardwareEnvelope:
    """Derive a :class:`HardwareEnvelope` from ``profile``.

    Args:
        profile: The YAML-loaded target profile.
        device_index: Which device in ``profile.devices`` to describe.
            Multi-device profiles derive envelopes per device.
        extra_hints: Caller-supplied hints appended to the authored
            ``CODEGEN_HINTS[profile.name]``. Lets recipe passes append
            per-kernel guidance without rewriting the target library.

    Returns:
        A :class:`HardwareEnvelope` ready to drop into a
        :class:`ExecutionEnvelope`.
    """
    if not profile.devices:
        raise ValueError(
            f"TargetProfile {profile.name!r} has no devices; can't derive envelope"
        )
    if device_index >= len(profile.devices):
        raise IndexError(
            f"device_index={device_index} out of range "
            f"({len(profile.devices)} devices in {profile.name!r})"
        )
    device = profile.devices[device_index]
    unit = _pick_primary_compute_unit(device)
    vector_lanes = unit.count if unit is not None else 1

    base_hints = CODEGEN_HINTS.get(profile.name, ())
    hints = tuple((*base_hints, *extra_hints))

    return HardwareEnvelope(
        target_name=profile.name,
        vector_lanes=vector_lanes,
        scratchpad_bytes=_scratchpad_bytes(device),
        register_bytes=_register_bytes(device),
        native_dtypes=_native_dtypes(device),
        peak_bandwidth_gbps=_peak_bandwidth_gbps(device),
        codegen_hints=hints,
    )


__all__ = [
    "CODEGEN_HINTS",
    "envelope_from_target_profile",
]
