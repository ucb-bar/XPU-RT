"""GPU target class — the abstractions any GPU vendor implements.

Per the unified target hierarchy (see
``docs/architecture/target-hierarchy-inventory.md``), this directory
holds:

- ``contracts.py`` — Protocols every GPU vendor's package must
  satisfy. Universal compile/dispatch paths import these and never
  reach into vendor-specific types.
- ``nvidia/`` — any-NVIDIA-GPU code (NVRTC, CUDA driver, etc.) +
  per-arch leaves under ``blackwell/``, ``hopper/``, ``ampere/``.
- ``amd/`` — placeholder; future HIP/ROCm.
- ``intel/`` — placeholder; future XPU.

Class-level invariants every GPU target shares (encoded in
``contracts.py``):

- Per-block parallelism with cooperative thread groups.
- Shared memory + global memory hierarchy.
- Event-tensor synchronization is meaningful (cross-block sync
  requires hardware support).
- Compile-time tile-shape selection matters for tensor-core
  engagement.

Things that are NOT class-level (and live in vendor or arch):

- Specific GEMM library (cuBLASDx for NVIDIA; rocBLAS/CK for AMD).
- Specific JIT toolchain (NVRTC for NVIDIA; comgr for AMD).
- Specific tensor-core MMA atom (mma.sync, wgmma, tcgen05; HIP
  has its own mfma family).
- Specific arch tag (sm_100 vs gfx942 vs xe_hpc).
"""

from __future__ import annotations

from compgen.targets.gpu.contracts import (
    DEFAULT_SCHEDULING_OVERHEAD_US,
    Device,
    EventTimer,
    GpuBodyEmitter,
    GpuCostModel,
    GpuProbe,
    GpuRuntime,
)

__all__ = [
    "DEFAULT_SCHEDULING_OVERHEAD_US",
    "Device",
    "EventTimer",
    "GpuBodyEmitter",
    "GpuCostModel",
    "GpuProbe",
    "GpuRuntime",
]
