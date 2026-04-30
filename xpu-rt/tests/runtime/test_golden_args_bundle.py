"""Tests for REQ-024 — ``bundle/golden_args.pt`` carries the unified
function-arg list (parameters + user inputs) in IR-arglist order.

The IR's ``func.func @forward`` arglist contains every parameter,
buffer, constant tensor, and user input — in the order PyTorch's
``ExportedProgram.graph_signature.input_specs`` enumerates them.
``golden_inputs.pt`` only has the user inputs (back-compat surface);
``golden_args.pt`` has the full list so consumer-side composers can
zip(args, golden_args) without independently parsing
``exported_program.pt2``.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
from compgen.capture.torch_export import capture_frontend_artifact
from compgen.runtime.bundle_emit import emit_extended_artefacts


def _emit(model: nn.Module, sample_inputs: tuple[torch.Tensor, ...], tmp_path: Path) -> Path:
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    (bundle_dir / "manifest.json").write_text(json.dumps({"artifacts": {}}))

    capture = capture_frontend_artifact(model, sample_inputs)
    emit_extended_artefacts(
        bundle_dir,
        capture_artifact=capture,
        sample_inputs=sample_inputs,
        model=model,
        run_compile_baseline=False,
    )
    return bundle_dir


def test_linear_golden_args_matches_func_forward_arglist(tmp_path: Path) -> None:
    """``nn.Linear(8, 4)`` → 3 args (W, bias, x); golden_args has 3 entries."""
    m = nn.Linear(8, 4).eval()
    x = torch.randn(1, 8)
    bundle = _emit(m, (x,), tmp_path)

    golden_args = torch.load(bundle / "golden_args.pt", weights_only=False)
    assert len(golden_args) == 3, len(golden_args)

    # Order matches input_specs order: PARAMETERS first (weight + bias),
    # then USER_INPUT (x). Identify each by shape.
    shapes = [tuple(t.shape) for t in golden_args]
    # weight is (4, 8); bias is (4,); x is (1, 8).
    assert (4, 8) in shapes
    assert (4,) in shapes
    assert (1, 8) in shapes


def test_golden_args_baked_param_values_match_state_dict(tmp_path: Path) -> None:
    """The PARAMETER entries in golden_args carry the model's weights —
    not random tensors that happen to share shape."""
    m = nn.Linear(2, 2).eval()
    # Pin known values so comparison is exact.
    with torch.no_grad():
        m.weight.copy_(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
        m.bias.copy_(torch.tensor([0.5, -0.5]))

    bundle = _emit(m, (torch.zeros(1, 2),), tmp_path)
    golden_args = torch.load(bundle / "golden_args.pt", weights_only=False)

    by_shape = {tuple(t.shape): t for t in golden_args}
    assert torch.allclose(by_shape[(2, 2)], torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
    assert torch.allclose(by_shape[(2,)], torch.tensor([0.5, -0.5]))


def test_golden_args_status_reported_in_manifest(tmp_path: Path) -> None:
    m = nn.Linear(8, 4).eval()
    bundle = _emit(m, (torch.randn(1, 8),), tmp_path)
    manifest = json.loads((bundle / "manifest.json").read_text())
    block = manifest["extended_artifacts"]
    assert block["golden_args"]["status"] == "ok"
    assert block["golden_args"]["path"] == "golden_args.pt"


def test_golden_inputs_remains_user_only(tmp_path: Path) -> None:
    """REQ-024 keeps ``golden_inputs.pt`` user-only for back-compat;
    only ``golden_args.pt`` contains the unified bundle."""
    m = nn.Linear(8, 4).eval()
    bundle = _emit(m, (torch.randn(1, 8),), tmp_path)

    golden_inputs = torch.load(bundle / "golden_inputs.pt", weights_only=False)
    assert len(golden_inputs) == 1, len(golden_inputs)
    assert tuple(golden_inputs[0].shape) == (1, 8)


def test_no_exported_program_skips_golden_args(tmp_path: Path) -> None:
    from compgen.runtime.bundle_emit import emit_extended_artefacts

    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    (bundle_dir / "manifest.json").write_text(json.dumps({"artifacts": {}}))

    class _FakeCapture:
        exported_program = None
        diagnostics = None

    report = emit_extended_artefacts(
        bundle_dir,
        capture_artifact=_FakeCapture(),
        sample_inputs=(torch.randn(2, 2),),
    )
    statuses = {s.name: s.status for s in report.statuses}
    assert statuses["golden_args"] == "skipped"


def test_golden_args_handles_param_only_models(tmp_path: Path) -> None:
    """A trivial elementwise model has no parameters — golden_args == golden_inputs."""

    class Add(nn.Module):
        def forward(self, a, b):
            return a + b

    bundle = _emit(Add(), (torch.randn(4), torch.randn(4)), tmp_path)
    golden_args = torch.load(bundle / "golden_args.pt", weights_only=False)
    golden_inputs = torch.load(bundle / "golden_inputs.pt", weights_only=False)
    assert len(golden_args) == len(golden_inputs) == 2
