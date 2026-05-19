"""Canonical model catalog for benchmark and graph-analysis workloads."""

from __future__ import annotations

from xpu_rt.models.catalog import build_default_model_catalog
from xpu_rt.models.core import CaptureMode, ModelCatalog, ModelSource, ModelSpec, ReadinessLevel
from xpu_rt.models.robotics import (
    get_graph_op_summary,
    load_smolvla,
    load_smolvla_bundle,
    load_smolvla_quantized,
    load_smolvla_quantized_bundle,
)

__all__ = [
    "CaptureMode",
    "ModelCatalog",
    "ModelSource",
    "ModelSpec",
    "ReadinessLevel",
    "build_default_model_catalog",
    "get_graph_op_summary",
    "load_smolvla",
    "load_smolvla_bundle",
    "load_smolvla_quantized",
    "load_smolvla_quantized_bundle",
]
