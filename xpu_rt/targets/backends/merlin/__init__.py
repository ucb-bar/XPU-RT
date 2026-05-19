"""Merlin compiler driver — XPU-RT-owned vendor flow.

Mirrors the ``backends/qnn/`` shape: this directory does NOT implement
``TargetBackendProtocol`` (the 4-stage lower/tile/decompose/emit
pipeline). Instead it owns typed data models and a thin Python bridge
to the merlin compiler at ``$MERLIN_ROOT`` (default
``/scratch2/agustin/merlin``).

Pipeline driven from this module::

    MerlinBridge.call("compile", ...)          # one (target, source) → vmfb
    MerlinBridge.call("compile_dispatch_matrix", ...)  # multi-target sweep
    MerlinBridge.call("chipyard", ...)         # chipyard image build
    MerlinBridge.call("onnx_to_mlir", ...)     # ONNX → payload MLIR
    MerlinBridge.call("profile_dispatch_matrix", ...)  # runner profiling

Resolution order in :class:`MerlinBridge`:

1. Python import (``from tools.<name> import main, setup_parser``).
2. Subprocess (``python -m tools.<name>``) under ``cwd=$MERLIN_ROOT``.
3. :class:`MerlinUnavailableError` if both fail.

Merlin's ``pyproject.toml`` exposes its top-level dirs as packages
(``tools``, ``compiler``, ``runtime``, ``target_specs``, ...), so the
import path works whenever merlin is pip-installed in the active env
OR ``$MERLIN_ROOT`` is on ``sys.path``.
"""

from .bridge import MerlinBridge, MerlinCallResult, MerlinUnavailableError
from .chipyard import MerlinChipyardImage, build_chipyard_image
from .compile import MerlinCompileResult, compile_program
from .dispatch_matrix import MerlinDispatchMatrix, compile_dispatch_matrix
from .onnx_bridge import OnnxImportResult, onnx_to_payload_mlir
from .profile import MerlinProfileResult, profile_dispatch_matrix
from .runners import FiresimRunner, HostRunner, NoopRunner, Runner, SpikeRunner
from .target_spec import MerlinTargetSpec, list_targets, load_target_spec

__all__ = [
    "MerlinBridge",
    "MerlinUnavailableError",
    "MerlinCallResult",
    "MerlinCompileResult",
    "compile_program",
    "MerlinDispatchMatrix",
    "compile_dispatch_matrix",
    "MerlinChipyardImage",
    "build_chipyard_image",
    "OnnxImportResult",
    "onnx_to_payload_mlir",
    "MerlinProfileResult",
    "profile_dispatch_matrix",
    "Runner",
    "SpikeRunner",
    "HostRunner",
    "FiresimRunner",
    "NoopRunner",
    "MerlinTargetSpec",
    "list_targets",
    "load_target_spec",
]
