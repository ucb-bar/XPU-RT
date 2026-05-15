"""Model admission registry and torch.compile suite.

Public surface for the model admission track. This package provides a
YAML-driven model registry, slice declarations, a torch.compile admission
probe, and a deterministic suite runner that emits typed
``unavailable / failed / pass`` reports for every (model, slice) pair.

The probe is a thin wrapper around the existing
:func:`compgen.capture.dynamo_baseline.compile_baseline` and
:func:`compgen.capture.dynamo_baseline.collect_diagnostics`. It does not
duplicate torch.compile / TorchDynamo plumbing.

Entry points:

- ``python -m compgen.model_admission validate-registry``
- ``python -m compgen.model_admission run-suite --suite ... --out ...``
- ``python -m compgen.model_admission torch-compile --model ... --slice ... --out ...``

Public API:

- :class:`ModelConfig`, :class:`SliceConfig`, :class:`SuiteConfig`
- :class:`AdmissionReport`, :class:`EagerReport`, :class:`DynamoCaptureReport`,
  :class:`TorchCompileReport`, :class:`SuiteSummary`
- :class:`AdmissionStatus`
- :func:`run_admission` -- top-level probe entry point
- :func:`load_registry` -- discover and validate every YAML config
"""

from __future__ import annotations

from compgen.model_admission.registry import (
    Registry,
    RegistryError,
    load_registry,
)
from compgen.model_admission.schemas import (
    AdmissionReport,
    AdmissionStatus,
    DynamoCaptureReport,
    EagerReport,
    ExportReport,
    FxReport,
    ModelConfig,
    SliceConfig,
    SuiteConfig,
    SuiteEntry,
    SuiteSummary,
    SuiteSummaryRow,
    TorchCompileReport,
)
from compgen.model_admission.torch_compile_probe import run_admission

__all__ = [
    "AdmissionReport",
    "AdmissionStatus",
    "DynamoCaptureReport",
    "EagerReport",
    "ExportReport",
    "FxReport",
    "ModelConfig",
    "Registry",
    "RegistryError",
    "SliceConfig",
    "SuiteConfig",
    "SuiteEntry",
    "SuiteSummary",
    "SuiteSummaryRow",
    "TorchCompileReport",
    "load_registry",
    "run_admission",
]
