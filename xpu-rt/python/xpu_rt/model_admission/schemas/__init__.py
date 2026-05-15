"""Schemas: dataclasses + on-disk JSON Schemas for model_admission contracts.

The dataclasses (``ModelConfig``, ``SliceConfig``, ``AdmissionReport``, ...)
live in :mod:`compgen.model_admission.schemas._contract` and are re-exported
here. JSON Schemas are under ``v1/*.schema.json`` and are loaded by
downstream tooling that wants raw schema files (e.g. cross-language
contract checkers).
"""

from __future__ import annotations

from compgen.model_admission.schemas._contract import (
    ADMISSION_REPORT_SCHEMA,
    AdmissionReport,
    AdmissionStatus,
    CompileConfig,
    DYNAMO_REPORT_SCHEMA,
    DynamoCaptureReport,
    EAGER_REPORT_SCHEMA,
    EXPORT_REPORT_SCHEMA,
    EagerReport,
    ExpectedOutcomes,
    ExportReport,
    HardwareRequirements,
    FX_REPORT_SCHEMA,
    FxReport,
    InputsSpec,
    MODEL_CONFIG_SCHEMA,
    ModelConfig,
    ModelLoaderConfig,
    ModelSource,
    REGISTRY_SCHEMA,
    SLICE_CONFIG_SCHEMA,
    SUITE_CONFIG_SCHEMA,
    SUITE_SUMMARY_SCHEMA,
    SliceConfig,
    StageStatus,
    SuiteConfig,
    SuiteEntry,
    SuiteSummary,
    SuiteSummaryRow,
    SupportPolicy,
    TORCH_COMPILE_REPORT_SCHEMA,
    TorchCompileReport,
    _expect_schema,
)

__all__ = [
    "ADMISSION_REPORT_SCHEMA",
    "AdmissionReport",
    "AdmissionStatus",
    "CompileConfig",
    "DYNAMO_REPORT_SCHEMA",
    "DynamoCaptureReport",
    "EAGER_REPORT_SCHEMA",
    "EXPORT_REPORT_SCHEMA",
    "EagerReport",
    "ExpectedOutcomes",
    "ExportReport",
    "HardwareRequirements",
    "FX_REPORT_SCHEMA",
    "FxReport",
    "InputsSpec",
    "MODEL_CONFIG_SCHEMA",
    "ModelConfig",
    "ModelLoaderConfig",
    "ModelSource",
    "REGISTRY_SCHEMA",
    "SLICE_CONFIG_SCHEMA",
    "SUITE_CONFIG_SCHEMA",
    "SUITE_SUMMARY_SCHEMA",
    "SliceConfig",
    "StageStatus",
    "SuiteConfig",
    "SuiteEntry",
    "SuiteSummary",
    "SuiteSummaryRow",
    "SupportPolicy",
    "TORCH_COMPILE_REPORT_SCHEMA",
    "TorchCompileReport",
]
