"""NVIDIA-vendor common code — shared across all NVIDIA arches.

Holds anything that applies to every NVIDIA GPU regardless of
arch:

- ``sm_tag.py`` — map NVRTC arch flag to cuBLASDx ``SM<...>`` tag.
- ``discovery.py`` — find cuBLASDx, libcudacxx, CUTLASS header paths.
- ``probe.py`` — :class:`CudaDeviceProbe` (Wave 1.14 migration target).
- ``module.py`` — :class:`CudaModule` NVRTC + driver wrapper (Wave 1.14).
- ``launcher.py`` — cooperative-launch wrapper (Wave 1.14).
- ``primitives.py`` — event tensors / dynamic queues (Wave 1.14).

Per-arch leaves under ``blackwell/``, ``hopper/``, ``ampere/``
inherit from these via the registry's vendor-common fallback.
"""

from __future__ import annotations

from compgen.targets.gpu.nvidia.common.sm_tag import arch_to_cublasdx_sm

__all__ = ["arch_to_cublasdx_sm"]
