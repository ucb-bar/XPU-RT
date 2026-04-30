"""NVIDIA Hopper arch-leaf — sm_90 / sm_90a.

Holds Hopper-specific specializations: ``wgmma.async`` MMA atoms,
no cluster-launch (cluster_dim=None for the static schedule),
TMA bulk copies, no tcgen05 (Blackwell-only), cu12 NVRTC sufficient
for the SM<900> dispatcher.

Wave 1.14 migrates existing Hopper-relevant code (currently in
``runtime/lowering/fx_to_megakernel.py``'s ``_arch_to_cublasdx_sm
("sm_90") == 900`` mapping + the per-arch TFLOPS table).

This package is mostly a placeholder today since most of the
project's perf characterization has been on Blackwell. Hopper
support is roadmap-tracked but not actively driven.
"""

from __future__ import annotations

from compgen.targets.registry import register_target


def _register_hopper() -> None:
    register_target(
        target_class="gpu",
        vendor="nvidia",
        arch="hopper",
        rationale=(
            "NVIDIA Hopper (H100, sm_90). cuBLASDx with SM<900> "
            "uses wgmma.async MMA atoms (different from Blackwell's "
            "tcgen05.mma family). Cluster-launch supported but not "
            "yet wired through the matcher (Wave 1.6 lands "
            "Blackwell first). Per the unified target hierarchy: "
            "see docs/architecture/target-hierarchy-inventory.md."
        ),
        registration_path="in_tree",
        metadata={
            "compute_capability_major": 9,
            "compute_capability_minor": 0,
            "supports_clusters": True,
            "supports_tensor_cores": True,
            "supports_wgmma_async": True,
            "supports_tcgen05_mma": False,
            "default_tile_shape": [32, 32, 32],
            "preferred_precision": "fp32",  # bf16 path future
            "cublasdx_sm_tag": 900,
            "sm_count": 132,
        },
    )


_register_hopper()
