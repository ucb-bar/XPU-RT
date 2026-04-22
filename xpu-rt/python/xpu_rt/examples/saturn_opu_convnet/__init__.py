"""Saturn OPU ConvNet — small ResNet-style CNN demo for the RISC-V +
V-extension + xopu target.

The model and its canonical input tuple live in :mod:`.model`, exposing
:func:`build_model` and :func:`default_inputs`. Resolve via
:func:`compgen.examples.resolve_demo_module` when calling tools that
accept a dotted module path (e.g. ``compile_embedded(model_module=...)``).

This package also ships the canonical golden input/output tensors
(``golden_inputs.pt``, ``golden_outputs.pt``) so downstream users can
run numerical-diff checks after a Spike / on-device run without needing
the CompGen source tree. Resolve via :func:`golden_input_path` and
:func:`golden_output_path`.
"""

from importlib import resources
from pathlib import Path

from .model import ConvNet, build_model, default_inputs


def golden_input_path() -> Path:
    """Filesystem path to the shipped ``golden_inputs.pt`` tensor dump."""
    return Path(str(resources.files(__package__).joinpath("golden_inputs.pt")))


def golden_output_path() -> Path:
    """Filesystem path to the shipped ``golden_outputs.pt`` tensor dump."""
    return Path(str(resources.files(__package__).joinpath("golden_outputs.pt")))


__all__ = [
    "ConvNet",
    "build_model",
    "default_inputs",
    "golden_input_path",
    "golden_output_path",
]
