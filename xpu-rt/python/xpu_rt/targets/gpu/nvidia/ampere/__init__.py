"""NVIDIA Ampere arch-leaf — sm_80 / sm_86.

Pre-Hopper consumer + datacenter (A100, RTX 30/40 series). Older
mma.sync atoms (m16n8k16 fp16/bf16 at fp32 acc), no clusters, no
tcgen05, no wgmma. cuBLASDx supports it but tile sizes are
smaller; defaults to fp32 SIMT or fp16 tensor-core paths.

Roadmap-tracked; most active development is on Blackwell.
"""

from __future__ import annotations

from compgen.targets.registry import register_target


def _register_ampere() -> None:
    register_target(
        target_class="gpu",
        vendor="nvidia",
        arch="ampere",
        rationale=(
            "NVIDIA Ampere (A100 / RTX 30 series, sm_80/86). Pre-"
            "Hopper architecture: mma.sync at m16n8k16, no clusters, "
            "no tcgen05, no wgmma. Cu12 NVRTC sufficient. Per the "
            "unified target hierarchy: see "
            "docs/architecture/target-hierarchy-inventory.md."
        ),
        registration_path="in_tree",
        metadata={
            "compute_capability_major": 8,
            "compute_capability_minor": 0,  # also sm_86 = (8, 6)
            "supports_clusters": False,
            "supports_tensor_cores": True,
            "supports_wgmma_async": False,
            "supports_tcgen05_mma": False,
            "default_tile_shape": [32, 32, 32],
            "preferred_precision": "fp32",
            "cublasdx_sm_tag": 800,
            "sm_count": 108,  # A100; RTX-series differ
        },
    )


_register_ampere()
