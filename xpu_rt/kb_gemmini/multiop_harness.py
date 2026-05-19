"""Backwards-compat shim — content moved to
:mod:`xpu_rt.spike_harness.multiop_harness`."""

from __future__ import annotations

from xpu_rt.spike_harness.multiop_harness import (  # noqa: F401
    KernelBinding,
    PipelineHarnessSpec,
    render_pipeline_driver_c,
    stage_pipeline_harness,
)

__all__ = [
    "KernelBinding",
    "PipelineHarnessSpec",
    "render_pipeline_driver_c",
    "stage_pipeline_harness",
]
