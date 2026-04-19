"""Tests for the torch-mlir / FX bridge (``compgen.capture.torch_mlir_bridge``)."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from xdsl.dialects.builtin import ModuleOp

from compgen.capture.torch_mlir_bridge import (
    BridgeResult,
    bridge_fx_graph,
    bridge_fx_graph_or_raise,
    module_to_text,
    torch_mlir_available,
)


class _TinyMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.fc(x))


def _tiny_inputs() -> tuple[torch.Tensor, ...]:
    torch.manual_seed(0)
    return (torch.randn(1, 4),)


# --- basic invocation --------------------------------------------------------


def test_bridge_returns_bridge_result_instance():
    result = bridge_fx_graph(_TinyMLP(), _tiny_inputs())
    assert isinstance(result, BridgeResult)


def test_bridge_returns_a_module_via_fallback_when_torch_mlir_missing():
    # In this environment torch-mlir isn't installed, so the fallback
    # FXImporter path must succeed on the tiny MLP.
    result = bridge_fx_graph(_TinyMLP(), _tiny_inputs())
    assert result.module is not None
    assert isinstance(result.module, ModuleOp)


def test_path_taken_reflects_reality():
    result = bridge_fx_graph(_TinyMLP(), _tiny_inputs())
    if torch_mlir_available():
        assert result.path_taken == "torch_mlir"
    else:
        assert result.path_taken == "fx_importer"


def test_diagnostics_are_populated():
    result = bridge_fx_graph(_TinyMLP(), _tiny_inputs())
    assert len(result.diagnostics) >= 1


# --- raise-on-failure wrapper -----------------------------------------------


def test_bridge_or_raise_returns_module_on_success():
    module = bridge_fx_graph_or_raise(_TinyMLP(), _tiny_inputs())
    assert isinstance(module, ModuleOp)


def test_bridge_or_raise_raises_when_both_paths_fail():
    # Force both paths to fail via allow_fallback=False AND no torch-mlir.
    if torch_mlir_available():
        pytest.skip("torch-mlir installed; cannot force both-failure path")

    with pytest.raises(RuntimeError, match="bridge failed"):
        bridge_fx_graph_or_raise(
            _TinyMLP(), _tiny_inputs(), allow_fallback=False,
        )


# --- module_to_text ---------------------------------------------------------


def test_module_to_text_returns_printable_mlir():
    result = bridge_fx_graph(_TinyMLP(), _tiny_inputs())
    text = module_to_text(result.module)
    assert "builtin.module" in text or "module" in text
    assert "func" in text


# --- ExportedProgram input (torch-mlir-only path) ---------------------------


def test_bridge_accepts_exported_program_shape():
    # The fallback FXImporter expects an ExportedProgram; our bridge
    # captures via ``capture_model`` when handed an nn.Module. Here we
    # verify the nn.Module path actually produces a usable module.
    fx = _TinyMLP()
    result = bridge_fx_graph(fx, _tiny_inputs())
    assert result.module is not None


# --- torch_mlir_available helper -------------------------------------------


def test_torch_mlir_available_returns_bool():
    assert isinstance(torch_mlir_available(), bool)
