"""Canonical model catalog for benchmark and graph-analysis workloads."""

from __future__ import annotations

from compgen.models.core import CaptureMode, ModelCatalog, ModelSource, ModelSpec, ReadinessLevel
from compgen.models.frontier import build_frontier_model_specs
from compgen.models.robotics import build_robotics_model_specs, get_graph_op_summary, load_smolvla, load_smolvla_bundle


def build_default_model_catalog() -> ModelCatalog:
    """Build the default catalog of heavyweight frontier models."""

    catalog = ModelCatalog()
    for spec in [*build_frontier_model_specs(), *build_robotics_model_specs()]:
        catalog.register(spec)
    return catalog


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
]
