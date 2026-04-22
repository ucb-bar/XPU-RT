"""Backcompat shim — the ConvNet demo now lives inside the installed
``compgen`` package at :mod:`compgen.examples.saturn_opu_convnet.model`.

This file re-exports the same names so existing test imports
(``tests.fixtures.saturn_opu_convnet.model``) keep working without a
source-tree copy.
"""

from compgen.examples.saturn_opu_convnet.model import (
    ConvNet,
    build_model,
    default_inputs,
)

__all__ = ["ConvNet", "build_model", "default_inputs"]
