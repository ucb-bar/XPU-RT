"""Tests for torch.export capture path."""

from __future__ import annotations

import sys
from pathlib import Path

from compgen.capture.torch_export import ExportValidation, capture_model, validate_export

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
