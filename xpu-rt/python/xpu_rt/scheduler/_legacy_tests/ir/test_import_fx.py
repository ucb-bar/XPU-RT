"""Tests for FX graph to xDSL conversion."""

from __future__ import annotations

import pytest


def test_fx_to_xdsl_simple_mlp() -> None:
    """fx_to_xdsl should convert SimpleMLP FX graph to xDSL module."""
    torch = pytest.importorskip("torch")

    from compgen.ir.payload.import_fx import FXImporter, fx_to_xdsl

    # Define a simple 2-layer MLP
    class SimpleMLP(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc1 = torch.nn.Linear(16, 32)
            self.fc2 = torch.nn.Linear(32, 8)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.fc2(torch.nn.functional.gelu(self.fc1(x)))

    model = SimpleMLP()
    model.eval()
    example_input = torch.randn(4, 16)

    exported = torch.export.export(model, (example_input,))
    module, diagnostics = fx_to_xdsl(exported)

    # The module should have been created and should verify
    from xdsl.dialects.builtin import ModuleOp

    assert isinstance(module, ModuleOp)

    # Should have a "forward" function inside
    importer = FXImporter()
    ir_text = importer.get_ir_text(module)
    assert "forward" in ir_text

    # At least some ops should have been decomposed (linear, gelu)
    decomposed = [d for d in diagnostics if d.level == "info" and "Decomposed" in d.message]
    assert len(decomposed) > 0, "Expected at least one decomposed op"


def test_fx_importer_diagnostics() -> None:
    """FXImporter should produce diagnostics for unsupported ops."""
    torch = pytest.importorskip("torch")

    from compgen.ir.payload.import_fx import FXImporter

    # A model with an op unlikely to be in the decomposition table
    class CustomModel(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.sigmoid(x)

    model = CustomModel()
    model.eval()
    example_input = torch.randn(2, 4)

    exported = torch.export.export(model, (example_input,))

    importer = FXImporter(allow_opaque_fallback=True)
    module = importer.import_graph(exported)

    # Should produce diagnostics -- at minimum info-level for decomposed/opaque ops
    assert len(importer.diagnostics) > 0

    # The module should still be valid
    from xdsl.dialects.builtin import ModuleOp

    assert isinstance(module, ModuleOp)

    # Coverage should be reported
    coverage = importer.decomposition_coverage
    assert 0.0 <= coverage <= 1.0
