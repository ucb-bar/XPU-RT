"""Tests for FX graph import."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

from compgen.capture.torch_export import capture_frontend_artifact, capture_model
from compgen.ir.payload.import_fx import FXImporter, ImportDiagnostic, fx_to_xdsl

EXAMPLES_DIR = Path(__file__).parent.parent.parent.parent / "examples" / "models"


def _get_simple_mlp():
    sys.path.insert(0, str(EXAMPLES_DIR))
    from simple_mlp import SimpleMLP, get_sample_inputs
    return SimpleMLP(), get_sample_inputs()


def test_import_diagnostic_fields() -> None:
    diag = ImportDiagnostic(fx_node="add_1", level="warning", message="unsupported attr")
    assert diag.fx_node == "add_1"
    assert diag.level == "warning"


def test_fx_importer_default_diagnostics() -> None:
    importer = FXImporter()
    assert importer.diagnostics == []


def test_fx_importer_import_graph() -> None:
    """import_graph should convert SimpleMLP FX graph to a valid xDSL module."""
    model, inputs = _get_simple_mlp()
    ep = capture_model(model, inputs)
    importer = FXImporter()
    module = importer.import_graph(ep)

    # Module should exist and be verified
    assert module is not None

    # Should have mapped ops (no errors)
    errors = [d for d in importer.diagnostics if d.level == "error"]
    assert len(errors) == 0

    # Should have at least 3 mapped ops (linear, gelu, linear)
    infos = [d for d in importer.diagnostics if d.level == "info"]
    assert len(infos) >= 3


def test_fx_to_xdsl_convenience() -> None:
    """fx_to_xdsl should return module and diagnostics."""
    model, inputs = _get_simple_mlp()
    ep = capture_model(model, inputs)
    module, diags = fx_to_xdsl(ep)
    assert module is not None
    assert isinstance(diags, list)


def test_ir_text_contains_ops() -> None:
    """Generated IR should contain real linalg ops (not opaque func.call)."""
    model, inputs = _get_simple_mlp()
    ep = capture_model(model, inputs)
    module, _ = fx_to_xdsl(ep)
    importer = FXImporter()
    ir_text = importer.get_ir_text(module)
    assert "linalg.matmul" in ir_text
    assert "linalg.transpose" in ir_text
    assert "func.func @forward" in ir_text
    # Region IDs should be present
    assert "compgen.region_id" in ir_text


def test_ir_text_has_tensor_types() -> None:
    """Generated IR should have correct tensor shapes."""
    model, inputs = _get_simple_mlp()
    ep = capture_model(model, inputs)
    module, _ = fx_to_xdsl(ep)
    ir_text = FXImporter().get_ir_text(module)
    assert "tensor<8x768xf32>" in ir_text
    assert "tensor<8x3072xf32>" in ir_text


class _SinModel(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(x)


def test_strict_import_uses_synthesized_translation_from_capture_artifact() -> None:
    """Strict import should consume synthesized payload translations from the artifact."""
    artifact = capture_frontend_artifact(_SinModel(), (torch.randn(4, 8),))
    module, diags = fx_to_xdsl(
        artifact.exported_program,
        **artifact.strict_import_options(),
    )

    assert module is not None
    errors = [diag for diag in diags if diag.level == "error"]
    assert errors == []
    ir_text = FXImporter().get_ir_text(module)
    assert "func.call @aten_sin_default" in ir_text
