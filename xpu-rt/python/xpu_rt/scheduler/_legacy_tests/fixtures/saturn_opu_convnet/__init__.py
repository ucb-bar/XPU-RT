"""Saturn OPU ConvNet fixture package."""

from __future__ import annotations

from tests.fixtures.saturn_opu_convnet.model import (
    ConvNet,
    build_model,
    default_inputs,
)

__all__ = ["ConvNet", "build_model", "default_inputs"]
