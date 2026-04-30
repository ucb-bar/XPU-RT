"""NVIDIA Blackwell arch-leaf — sm_100 / sm_120.

Per the unified target hierarchy, this is THE ONLY place
Blackwell-specific code lives. cuBLASDx SM<1000>, cu13 NVRTC,
mma.sync at 64×64×16, tcgen05, cluster-launch wiring (Wave 1.6
lands here).

Wave 1.14 will migrate existing code from these locations:
- ``runtime/native/cuda.py::_resolve_cu13_nvrtc_lib_path`` →
  ``cu13_nvrtc.py``
- ``runtime/lowering/fx_to_megakernel.py::_cublasdx_gemm_body`` →
  ``body_emitter.py``
- ``runtime/lowering/fx_to_megakernel.py::_TILE_M_CUBLASDX = 64`` →
  ``tile_shape.py``
- ``runtime/autotune/__init__.py`` decision tree for Blackwell →
  ``decision.py``
- Per-arch TFLOPS table from ``kernels/cost/etc_predict.py`` →
  ``cost.py``

Until Wave 1.14, this module's ``__init__.py`` only registers the
target with placeholder adapters. The migration is mechanical and
non-breaking — universal modules continue importing from the old
locations until the move + re-export shim lands.
"""

from __future__ import annotations

from compgen.targets.registry import register_target


def _register_blackwell() -> None:
    register_target(
        target_class="gpu",
        vendor="nvidia",
        arch="blackwell",
        rationale=(
            "NVIDIA Blackwell datacenter (B100/B200, sm_100) + "
            "workstation (RTX PRO 6000, sm_120). cuBLASDx with "
            "Arrangement<row_major, row_major, row_major> at "
            "Size<64, 64, 16> + Precision<bf16, bf16, fp32> "
            "engages mma.sync (per bridge #095 PTX dump). "
            "Requires cu13 NVRTC for __CUDA_ARCH__ == 1000 "
            "(per #089 — cu12 silently SIMTs). "
            "Cluster-launch (Wave 1.6) is the cooperative-grid-sync "
            "fix that bwell #108 perf data validates as the next "
            "perf lever. Per the unified target hierarchy: see "
            "docs/architecture/target-hierarchy-inventory.md."
        ),
        registration_path="in_tree",
        metadata={
            "compute_capability_major": 10,
            "compute_capability_minor": 0,  # also sm_120 = (12, 0)
            "supports_clusters": True,
            "supports_tensor_cores": True,
            "supports_tcgen05_mma": True,
            "supports_cu13_nvrtc": True,
            "default_tile_shape": [64, 64, 16],
            "preferred_precision": "bf16_fp32",
            "cublasdx_sm_tag": 1000,
            "sm_count_per_chip": {"sm_100": 132, "sm_120": 188},
        },
    )


_register_blackwell()
