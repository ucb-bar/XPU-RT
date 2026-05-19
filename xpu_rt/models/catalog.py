"""Default model catalog builder."""

from __future__ import annotations

from xpu_rt.models.core import ModelCatalog
from xpu_rt.models.frontier import build_frontier_model_specs
from xpu_rt.models.robotics import build_robotics_model_specs


def build_default_model_catalog() -> ModelCatalog:
    """Build the default catalog of heavyweight frontier models."""

    catalog = ModelCatalog()
    for spec in [*build_frontier_model_specs(), *build_robotics_model_specs()]:
        catalog.register(spec)
    return catalog
