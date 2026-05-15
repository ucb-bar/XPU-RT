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
    # Either the wave-7 typed decomp (``@aten_sin`` + ``compgen._pattern_hint``)
    # or the legacy opaque-fallback shape (``@aten_sin_default``) satisfies
    # the contract: a synthesized translation lands for sin.
    assert "func.call @aten_sin" in ir_text


# ---------------------------------------------------------------------------
# Func-signature reconciliation: the FX output-node's recorded dtype can
# diverge from what the body actually produces (e.g. HF Llama declares
# bf16 outputs but the attention math upcasts to f32 at the return).
# Without reconciliation, xDSL's verifier rejects the module with
# "Expected arguments to have the same types as the function output types".
# ---------------------------------------------------------------------------


class _UpcastReturnModel(torch.nn.Module):
    """Body upcasts the bf16 input to f32 before returning.

    Isolates the func-signature reconciliation path: no matmul mixed-dtype
    interactions, just a single op that changes the return dtype between
    the FX output-node metadata and the produced SSA value.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.to(torch.float32)


def test_import_reconciles_func_signature_with_body_return_type() -> None:
    model = _UpcastReturnModel().eval()
    sample = (torch.randn(2, 8, dtype=torch.bfloat16),)
    ep = capture_model(model, sample)
    module, _ = fx_to_xdsl(ep)
    # Must verify — if reconciliation regressed, this raises
    # VerifyException("Expected arguments to have the same types as ...").
    module.verify()


def test_coerce_static_dim_keeps_concrete_dims() -> None:
    from compgen.ir.payload.import_fx import _coerce_static_dim

    assert _coerce_static_dim(7) == 7
    assert _coerce_static_dim(0) == 0


def test_coerce_static_dim_falls_back_to_negative_one_for_symbolic_dim() -> None:
    """Models with dynamic shapes (e.g. SmolVLA's image tile counts) carry
    SymInt dims that ``int(...)`` cannot specialize. We emit -1 so xDSL's
    TensorType verifier accepts them as dynamic, and capture continues."""
    from compgen.ir.payload.import_fx import _coerce_static_dim

    class _Symish:
        def __int__(self) -> int:
            raise RuntimeError("data-dependent SymInt — can't specialize")

    assert _coerce_static_dim(_Symish()) == -1
