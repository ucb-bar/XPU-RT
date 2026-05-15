"""Tests for runtime/bundle_runner.py — load + run a bundle from disk.

Builds a minimal bundle directory in a tmp path (payload.mlir +
manifest.json + exported_program.pt2 + golden_inputs.pt +
golden_outputs.pt), then round-trips it through
:func:`load_bundle` → :func:`run_bundle`, asserting the output
matches the stored golden.
"""

from __future__ import annotations

import io
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
import torch
from compgen.capture.torch_mlir_bridge import bridge_fx_graph
from compgen.runtime.bundle_runner import LoadedBundle, load_bundle, run_bundle
from xdsl.printer import Printer

from tests._fixtures.real_workloads import attention_mlp_tiny


def _write_full_bundle(bundle_dir: Path) -> Path:
    """Materialise a complete bundle (all 5 artefacts) from the
    attention_mlp_tiny fixture. Returns the bundle dir.

    This mirrors what Phase A item 6 (full-artefact emission) will
    wire into the bundle stage; here we do it by hand so the runner
    can be tested independently.
    """
    bundle_dir.mkdir(parents=True, exist_ok=True)
    fx = attention_mlp_tiny()
    bridged = bridge_fx_graph(fx.model, fx.example_inputs)
    assert bridged.module is not None

    # payload.mlir
    buf = io.StringIO()
    Printer(stream=buf).print_op(bridged.module)
    payload_text = buf.getvalue()
    (bundle_dir / "payload.mlir").write_text(payload_text)

    # manifest.json
    import hashlib

    manifest = {
        "version": "1.0",
        "target_profile": "test-cpu",
        "model_hash": hashlib.sha256(payload_text.encode()).hexdigest()[:16],
        "artifacts": {
            "payload": "payload.mlir",
            "exported_program": "exported_program.pt2",
            "golden_inputs": "golden_inputs.pt",
            "golden_outputs": "golden_outputs.pt",
        },
        "creation_timestamp": datetime.now(UTC).isoformat(),
    }
    (bundle_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    # exported_program.pt2
    torch.export.save(fx.exported, str(bundle_dir / "exported_program.pt2"))

    # golden_inputs.pt
    torch.save(list(fx.example_inputs), bundle_dir / "golden_inputs.pt")

    # golden_outputs.pt
    torch.save(fx.eager_output, bundle_dir / "golden_outputs.pt")

    return bundle_dir


def _write_minimal_bundle(bundle_dir: Path) -> Path:
    """Materialise only the two required artefacts
    (payload.mlir + manifest.json) — what BundleStage emits today."""
    bundle_dir.mkdir(parents=True, exist_ok=True)
    fx = attention_mlp_tiny()
    bridged = bridge_fx_graph(fx.model, fx.example_inputs)
    assert bridged.module is not None

    buf = io.StringIO()
    Printer(stream=buf).print_op(bridged.module)
    payload_text = buf.getvalue()
    (bundle_dir / "payload.mlir").write_text(payload_text)

    manifest = {
        "version": "1.0",
        "target_profile": "test-cpu",
        "model_hash": "deadbeef" * 2,
        "artifacts": {"payload": "payload.mlir"},
        "creation_timestamp": datetime.now(UTC).isoformat(),
    }
    (bundle_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return bundle_dir


# --- load_bundle ------------------------------------------------------------


def test_load_bundle_full_rehydrates_all_artefacts(tmp_path: Path) -> None:
    _write_full_bundle(tmp_path)

    bundle = load_bundle(tmp_path)
    assert isinstance(bundle, LoadedBundle)
    assert bundle.bundle_dir == tmp_path
    assert bundle.manifest["target_profile"] == "test-cpu"
    assert bundle.model_hash  # populated
    assert bundle.payload_module is not None
    assert (
        bundle.payload_text.startswith("builtin.module")
        or bundle.payload_text.startswith("module")
        or "func.func" in bundle.payload_text
    )
    assert bundle.exported_program is not None
    assert bundle.golden_inputs is not None
    assert isinstance(bundle.golden_inputs, tuple)
    assert len(bundle.golden_inputs) >= 1
    assert isinstance(bundle.golden_output, torch.Tensor)
    # Diagnostics record which files were present.
    assert "payload.mlir" in bundle.diagnostics["present"]
    assert "manifest.json" in bundle.diagnostics["present"]
    assert "exported_program.pt2" in bundle.diagnostics["present"]
    assert "golden_inputs.pt" in bundle.diagnostics["present"]
    assert "golden_outputs.pt" in bundle.diagnostics["present"]


def test_load_bundle_minimal_works_with_optional_artefacts_absent(tmp_path: Path) -> None:
    _write_minimal_bundle(tmp_path)

    bundle = load_bundle(tmp_path)
    assert bundle.payload_module is not None
    # Optional fields are None when the files aren't on disk.
    assert bundle.exported_program is None
    assert bundle.golden_inputs is None
    assert bundle.golden_output is None


def test_load_bundle_raises_if_manifest_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="manifest"):
        load_bundle(tmp_path)


def test_load_bundle_raises_if_payload_missing(tmp_path: Path) -> None:
    (tmp_path / "manifest.json").write_text(json.dumps({"version": "1.0"}))
    with pytest.raises(FileNotFoundError, match="payload"):
        load_bundle(tmp_path)


def test_load_bundle_raises_if_dir_missing(tmp_path: Path) -> None:
    missing = tmp_path / "nonexistent"
    with pytest.raises(FileNotFoundError, match="bundle directory"):
        load_bundle(missing)


# --- run_bundle -------------------------------------------------------------


def test_run_bundle_roundtrip_matches_golden(tmp_path: Path) -> None:
    """Full end-to-end: write a bundle, load it, run it, compare to
    the stored golden output. This is the flagship test for the
    recipe-library re-execution path."""
    _write_full_bundle(tmp_path)

    bundle = load_bundle(tmp_path)
    output = run_bundle(bundle)

    assert isinstance(output, torch.Tensor)
    assert bundle.golden_output is not None
    assert tuple(output.shape) == tuple(bundle.golden_output.shape)
    max_abs_diff = (output - bundle.golden_output).abs().max().item()
    assert max_abs_diff < 1e-5


def test_run_bundle_accepts_explicit_inputs(tmp_path: Path) -> None:
    """``run_bundle(bundle, inputs=...)`` overrides ``golden_inputs``."""
    _write_full_bundle(tmp_path)
    bundle = load_bundle(tmp_path)

    assert bundle.golden_inputs is not None
    # Re-run with the same inputs explicitly supplied; must match golden.
    output = run_bundle(bundle, inputs=bundle.golden_inputs)
    max_abs_diff = (output - bundle.golden_output).abs().max().item()
    assert max_abs_diff < 1e-5


def test_run_bundle_raises_without_exported_program(tmp_path: Path) -> None:
    _write_minimal_bundle(tmp_path)
    bundle = load_bundle(tmp_path)
    with pytest.raises(ValueError, match="exported_program"):
        run_bundle(bundle)


def test_run_bundle_raises_without_inputs(tmp_path: Path) -> None:
    """If the bundle has exported_program but no golden_inputs and the
    caller doesn't supply inputs, we fail loudly."""
    _write_full_bundle(tmp_path)
    # Remove golden_inputs.pt
    (tmp_path / "golden_inputs.pt").unlink()
    bundle = load_bundle(tmp_path)
    assert bundle.exported_program is not None
    assert bundle.golden_inputs is None

    with pytest.raises(ValueError, match="no inputs"):
        run_bundle(bundle)
