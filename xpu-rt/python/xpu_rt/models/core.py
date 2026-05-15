"""Model catalog and workload metadata for heavyweight benchmark models."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any

import torch.nn as nn

if TYPE_CHECKING:
    from benchmarks.spec import WorkspaceConfig
else:
    WorkspaceConfig = Any


ModelLoader = Callable[[WorkspaceConfig | None], tuple[nn.Module, tuple[Any, ...]]]


class CaptureMode(StrEnum):
    """Supported frontend capture modes."""

    TORCH_EXPORT = "torch_export"
    TORCH_DYNAMO_PARTITIONED = "torch_dynamo_partitioned"


class ReadinessLevel(StrEnum):
    """Expected level of benchmark support for a workload."""

    FULL_PIPELINE = "full_pipeline"
    ANALYSIS_ONLY = "analysis_only"
    PROBE_ONLY = "probe_only"


@dataclass(frozen=True)
class ModelSource:
    """Origin metadata for a model."""

    kind: str
    identifier: str
    repo_name: str = ""
    notes: str = ""


@dataclass(frozen=True)
class ModelSpec:
    """Canonical definition of a benchmarkable model workload."""

    model_id: str
    family: str
    description: str
    loader: ModelLoader
    source: ModelSource
    source_model_id: str = ""
    capture_mode: CaptureMode = CaptureMode.TORCH_EXPORT
    readiness: ReadinessLevel = ReadinessLevel.FULL_PIPELINE
    expected_status: str = "pass"
    dynamic_shapes: dict[str, Any] = field(default_factory=dict)
    tags: tuple[str, ...] = ()
    requirements: tuple[str, ...] = ()

    def load(self, workspace: WorkspaceConfig | None = None) -> tuple[nn.Module, tuple[Any, ...]]:
        """Load the model and sample inputs."""

        return self.loader(workspace)


@dataclass
class ModelCatalog:
    """Registry of model specifications."""

    models: dict[str, ModelSpec] = field(default_factory=dict)

    def register(self, spec: ModelSpec) -> None:
        self.models[spec.model_id] = spec

    def get(self, model_id: str) -> ModelSpec:
        return self.models[model_id]

    def items(self) -> list[tuple[str, ModelSpec]]:
        return sorted(self.models.items())
