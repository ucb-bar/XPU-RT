"""NVIDIA-vendor GPU code under the unified target hierarchy.

Holds anything shared across all NVIDIA GPU arches (CUDA driver
wrappers, NVRTC interface, generic cooperative-launch glue,
discovery helpers for cuBLASDx / libcudacxx / CUTLASS) plus per-arch
leaves under ``blackwell/``, ``hopper/``, ``ampere/``.

Registration: this package's ``__init__.py`` calls
:func:`register_target` at import time for the vendor-common entry
(``"gpu.nvidia"``). The arch leaves (``blackwell``, etc.) register
themselves for ``"gpu.nvidia.blackwell"`` etc.

The registry's vendor-common fallback (``registry.get`` walks up
to the vendor entry when an arch isn't registered) means a leaf
package can ship just the arch-specific overrides and inherit the
vendor's probe / runtime.

Files under this package:
- ``common/`` — shared utilities (CUDA driver wrappers, NVRTC,
  generic cooperative launch). All NVIDIA arches use these.
- ``blackwell/`` — sm_100/sm_120 specifics: cuBLASDx SM<1000>,
  cu13 NVRTC, mma.sync at 64×64×16, tcgen05, cluster-launch.
- ``hopper/`` — sm_90 specifics: wgmma, no cluster.
- ``ampere/`` — sm_80/sm_86 specifics: older mma atoms.
"""

from __future__ import annotations

from compgen.targets.registry import register_target


def _register_vendor_common() -> None:
    """Register the vendor-common ``gpu.nvidia`` entry. Currently a
    placeholder — arch-leaf packages override it. Wave 1.14 fills
    in the adapters when migration moves the existing CUDA/NVRTC
    code into ``targets/gpu/nvidia/common/``."""
    register_target(
        target_class="gpu",
        vendor="nvidia",
        arch="",  # vendor-common
        rationale=(
            "NVIDIA GPU vendor-common entry. Holds CUDA driver "
            "wrapper + NVRTC + cooperative launch primitives shared "
            "across all arches. Arch-specific specializations land "
            "in `gpu.nvidia.blackwell`, `gpu.nvidia.hopper`, etc. "
            "Per the unified target hierarchy: see "
            "docs/architecture/target-hierarchy-inventory.md."
        ),
        registration_path="in_tree",
        metadata={
            "vendor_url": "https://developer.nvidia.com/cuda",
            "jit_toolchain": "NVRTC",
            "memory_model": "unified-virtual-addressing",
        },
    )


_register_vendor_common()
