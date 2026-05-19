"""Torch / ONNX → payload MLIR bridge — QNN-facing alias.

The implementation now lives under
``xpu_rt.targets.backends.merlin.onnx_bridge`` (lifted there because
the importer is a merlin feature reused by saturn_opu / gemmini /
spacemit flows, not QNN-specific). This module re-exports the public
API so existing callers (``heterogeneous_loop.py``,
``scripts/qnn_paper_figure_demo.py``, …) keep working unchanged.
"""

from __future__ import annotations

from xpu_rt.targets.backends.merlin.onnx_bridge import (
    OnnxImportResult,
    onnx_to_payload_mlir,
)

__all__ = ["OnnxImportResult", "onnx_to_payload_mlir"]
