"""Tests for torch.export capture path."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from compgen.capture.torch_export import (
    CaptureArtifact,
    ExportValidation,
    capture_dynamo_partitions,
    capture_frontend_artifact,
    capture_model,
    validate_export,
)

# Add examples to path for model imports
EXAMPLES_DIR = Path(__file__).parent.parent.parent / "examples" / "models"


def _get_simple_mlp():
    sys.path.insert(0, str(EXAMPLES_DIR))
    from simple_mlp import SimpleMLP, get_sample_inputs

    return SimpleMLP(), get_sample_inputs()


def test_capture_simple_mlp() -> None:
    """capture_model should produce an ExportedProgram for SimpleMLP."""
    model, inputs = _get_simple_mlp()
    ep = capture_model(model, inputs)
    assert ep is not None
    assert hasattr(ep, "graph")
    # Should have nodes in the graph
    nodes = list(ep.graph.nodes)
    assert len(nodes) > 0


def test_capture_from_module_directly() -> None:
    """capture_model should accept a live nn.Module."""
    model, inputs = _get_simple_mlp()
    ep = capture_model(model, inputs)
    # Should have call_function nodes (the actual ops)
    call_nodes = [n for n in ep.graph.nodes if n.op == "call_function"]
    assert len(call_nodes) >= 2  # at least linear + gelu


def test_validate_export_passes() -> None:
    """validate_export should pass on a valid ExportedProgram."""
    model, inputs = _get_simple_mlp()
    ep = capture_model(model, inputs)
    validation = validate_export(ep)
    assert validation.valid
    assert validation.round_trip_ok
    assert validation.num_ops >= 2  # at least 2 call_function ops
    assert len(validation.graph_breaks) == 0


def test_export_validation_fields() -> None:
    """ExportValidation dataclass fields should work correctly."""
    v = ExportValidation(valid=True, round_trip_ok=True, num_ops=5)
    assert v.valid
    assert v.num_ops == 5
    assert v.graph_breaks == []
    assert v.warnings == []


def test_capture_frontend_artifact_collects_boundary_metadata() -> None:
    """capture_frontend_artifact should record the prepared export boundary."""
    model, inputs = _get_simple_mlp()
    artifact = capture_frontend_artifact(model, inputs)

    assert isinstance(artifact, CaptureArtifact)
    assert artifact.validation.valid
    assert artifact.capture_mode == "torch_export"
    assert artifact.analysis_success is True
    assert artifact.graph_count == 1
    assert "torch" in artifact.runtime_versions
    assert len(artifact.decomposition_targets) > 0
    assert artifact.diagnostics.graph_count >= 0

    prepared_targets = [
        str(node.target) for node in artifact.exported_program.graph.nodes if node.op == "call_function"
    ]
    assert "aten.addmm.default" in prepared_targets
    assert "aten.permute.default" in prepared_targets


class _ErfModel(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.erf(x)


def test_capture_frontend_artifact_recovers_simple_unsupported_ops() -> None:
    """Simple unsupported ATen ops should get a synthesized recovery plan."""
    artifact = capture_frontend_artifact(_ErfModel(), (torch.randn(4, 8),))

    assert len(artifact.unsupported_resolutions) >= 1
    resolution = next(r for r in artifact.unsupported_resolutions if r.target == "aten.erf.default")
    assert resolution.classification.strategy == "synthesized_external_call"
    assert resolution.translation is not None
    assert resolution.verification.schema_ok
    assert resolution.verification.eager_reference_ok


def test_capture_dynamo_partitions_collects_fx_graphs() -> None:
    """TorchDynamo partition capture should return FX graph modules."""
    model, inputs = _get_simple_mlp()
    artifact = capture_dynamo_partitions(model, inputs)

    assert artifact.capture_mode == "torch_dynamo_partitioned"
    assert artifact.exported_program is None
    assert artifact.analysis_success is True
    assert artifact.graph_count >= 1
    assert len(artifact.graphs) >= 1
    assert artifact.validation.num_ops >= 2
