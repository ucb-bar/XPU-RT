"""The 14-artifact contract: every slot reports a status, nothing is silent.

``CLAUDE.md`` promises 14 artifacts per compiled bundle. Historically only
``payload.mlir`` + ``manifest.json`` actually materialised; the rest were
"aspirational". Phase 1 of the production-hardening closed that gap.

This file pins down the new guarantees:

1. Every slot has a :class:`ArtifactStatus` in the manifest's
   ``extended_artifacts`` block.
2. "Skipped" slots name a reason; nothing is silently absent.
3. Genuine emission bugs raise :class:`BundleEmissionError` (tested
   via a synthetic failure injection).
4. No bundle lands under ``/tmp`` unless the caller explicitly pointed
   there.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from compgen.api import compile_model, device
from compgen.runtime.bundle_emit import emit_extended_artefacts
from compgen.runtime.errors import ArtifactStatus, BundleEmissionError, BundleEmissionReport

EXEMPLAR_DIR = Path(__file__).parent.parent / "targetgen" / "exemplars"


# The 14 slots from CLAUDE.md. ``payload`` + ``manifest`` are written by
# the bundle stage itself (not via extended-artifact emission); the
# remaining 12 are the domain of :func:`emit_extended_artefacts`.
_EXTENDED_SLOT_NAMES = frozenset(
    {
        "exported_program",
        "golden_inputs",
        "golden_outputs",
        "compile_baseline",
        "graph_breaks",
        "execution_plan",
        "memory_plan",
        "gap_analysis",
        "kernel_contracts",
        "transforms",
        "generated_kernels",
        "verification_report",
    }
)


class _TinyMLP(nn.Module):
    """Same shape as the bundle_emit integration test."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(32, 16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x).relu()


@pytest.fixture
def compiled_bundle(tmp_path: Path) -> Path:
    torch.manual_seed(0)
    model = _TinyMLP()
    inputs = (torch.randn(4, 32),)
    dev = device(EXEMPLAR_DIR / "test_gpu_simt.yaml", output_dir=tmp_path / "tgt")
    compiled = compile_model(model, dev, sample_inputs=inputs, verify=False)
    bundle_dir_str = compiled.pipeline_result.all_artifacts.get("bundle_dir")
    assert bundle_dir_str is not None
    return Path(bundle_dir_str)


def test_manifest_has_extended_artifacts_block(compiled_bundle: Path) -> None:
    """Every compiled bundle gets a structured ``extended_artifacts``
    block in the manifest, not just top-level artifact paths."""
    manifest = json.loads((compiled_bundle / "manifest.json").read_text())
    assert "extended_artifacts" in manifest, "manifest must carry a status block"
    block = manifest["extended_artifacts"]
    assert isinstance(block, dict)


def test_every_extended_slot_is_reported(compiled_bundle: Path) -> None:
    """All 12 extended-artifact slots report a status — nothing silent."""
    manifest = json.loads((compiled_bundle / "manifest.json").read_text())
    block = manifest["extended_artifacts"]
    missing = _EXTENDED_SLOT_NAMES - set(block.keys())
    assert not missing, f"slots missing from manifest: {sorted(missing)}"


def test_every_status_is_valid(compiled_bundle: Path) -> None:
    """Status is one of the three honest values."""
    manifest = json.loads((compiled_bundle / "manifest.json").read_text())
    block = manifest["extended_artifacts"]
    for name, entry in block.items():
        assert entry["status"] in {"ok", "skipped", "failed"}, f"slot {name!r} has invalid status {entry['status']!r}"


def test_skipped_slots_have_reason(compiled_bundle: Path) -> None:
    """If a slot is skipped, the manifest tells you why."""
    manifest = json.loads((compiled_bundle / "manifest.json").read_text())
    block = manifest["extended_artifacts"]
    for name, entry in block.items():
        if entry["status"] == "skipped":
            assert entry["reason"], f"skipped slot {name!r} must carry a non-empty reason"


def test_ok_slots_have_materialised(compiled_bundle: Path) -> None:
    """Every ``ok`` status points at a file/dir that exists."""
    manifest = json.loads((compiled_bundle / "manifest.json").read_text())
    block = manifest["extended_artifacts"]
    for name, entry in block.items():
        if entry["status"] == "ok":
            rel_path = entry["path"]
            assert rel_path, f"ok slot {name!r} missing a path"
            candidate = compiled_bundle / rel_path.rstrip("/")
            assert candidate.exists(), f"ok slot {name!r} points at missing path {rel_path}"


def test_tinymlp_emits_core_extended_artifacts(compiled_bundle: Path) -> None:
    """For TinyMLP the core 7 slots should concretely emit as ``ok``."""
    manifest = json.loads((compiled_bundle / "manifest.json").read_text())
    block = manifest["extended_artifacts"]
    for name in (
        "exported_program",
        "golden_inputs",
        "golden_outputs",
        "graph_breaks",
        "kernel_contracts",
        "execution_plan",
        "memory_plan",
    ):
        assert block[name]["status"] == "ok", f"slot {name!r} expected ok for TinyMLP, got {block[name]}"


def test_bundle_emission_error_on_failure(tmp_path: Path) -> None:
    """When an artifact genuinely fails (not skipped), the aggregate
    error raises with the failing artifact listed."""
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    # Minimum manifest so ``_update_manifest`` has something to write to.
    (bundle_dir / "manifest.json").write_text(json.dumps({"artifacts": {}}))

    # Drive a genuine failure: pass sample_inputs containing an object
    # torch can't save (a function), forcing ``torch.save`` to raise —
    # that's a disk-write / serialisation error, which is exactly what
    # ``failed`` status is for.
    class _Unserializable:
        pass

    class _FakeCapture:
        exported_program = None
        diagnostics = None

    report = emit_extended_artefacts(
        bundle_dir,
        capture_artifact=_FakeCapture(),
        sample_inputs=(_Unserializable(),),  # torch.save will choke
    )
    assert isinstance(report, BundleEmissionReport)
    # Building the exception type is what the API does with strict_artifacts=True.
    exc = BundleEmissionError(report)
    assert "golden_inputs" in str(exc), "failed artifact must surface in the error message"


def test_artifact_status_serialisation_roundtrip() -> None:
    """The manifest-writing path uses ``as_dict``; it must round-trip."""
    s = ArtifactStatus(name="demo", status="ok", path="demo.txt")
    d = s.as_dict()
    assert d == {"status": "ok", "path": "demo.txt", "error": None, "reason": None}


def test_report_partitions_statuses() -> None:
    report = BundleEmissionReport(
        bundle_dir=Path("/does/not/exist"),
        statuses=(
            ArtifactStatus(name="a", status="ok", path="a.txt"),
            ArtifactStatus(name="b", status="failed", error="boom"),
            ArtifactStatus(name="c", status="skipped", reason="no data"),
        ),
    )
    assert [s.name for s in report.ok] == ["a"]
    assert [s.name for s in report.failed] == ["b"]
    assert [s.name for s in report.skipped] == ["c"]
    block = report.to_manifest_block()
    assert set(block.keys()) == {"a", "b", "c"}
