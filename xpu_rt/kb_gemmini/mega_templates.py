"""Backwards-compat shim — content moved to
:mod:`xpu_rt.spike_harness.templates.mega_gemmini`."""

from __future__ import annotations

from xpu_rt.spike_harness.templates.mega_gemmini import (  # noqa: F401
    FusedKernelArtifacts,
    render_fused_artifacts,
    render_fused_driver_c,
    render_fused_init_c,
    stage_mega_contract_dir,
)

__all__ = [
    "FusedKernelArtifacts",
    "render_fused_artifacts",
    "render_fused_driver_c",
    "render_fused_init_c",
    "stage_mega_contract_dir",
]
