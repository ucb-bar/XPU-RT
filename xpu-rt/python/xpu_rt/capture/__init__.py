"""Stage 0 -- Model capture and baselining.

This subpackage handles:

- ``torch.export`` capture to produce an ExportedProgram
- ``torch.compile`` baseline with diagnostics (graph breaks, op coverage)
- TorchAO quantization pipeline integration

The output of this stage is the golden reference (inputs/outputs for
correctness testing) and the exported program that feeds into IR construction.
"""

from __future__ import annotations

from compgen.capture.dynamo_baseline import (
    BaselineReport,
    DynamoReport,
    GuardObservation,
    collect_diagnostics,
    compile_baseline,
)
from compgen.capture.torch_export import (
    CaptureArtifact,
    ExportValidation,
    RangeConstraint,
    capture_dynamo_partitions,
    capture_frontend,
    capture_frontend_artifact,
    capture_model,
    validate_export,
)
from compgen.capture.torchao_pipeline import AccuracyReport, QuantizationConfig, apply_quantization, verify_quant_accuracy

__all__ = [
    "AccuracyReport",
    "BaselineReport",
    "CaptureArtifact",
    "DynamoReport",
    "ExportValidation",
    "GuardObservation",
    "QuantizationConfig",
    "RangeConstraint",
    "apply_quantization",
    "capture_dynamo_partitions",
    "capture_frontend",
    "capture_frontend_artifact",
    "capture_model",
    "collect_diagnostics",
    "compile_baseline",
    "validate_export",
    "verify_quant_accuracy",
]
