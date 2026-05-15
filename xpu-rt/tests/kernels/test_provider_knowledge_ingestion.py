"""Tests for provider knowledge / contract-feedback flow through fallback.

The bidirectional ``KernelProvider`` protocol surfaces two outputs
beyond ``kernel_code``:

- ``ProviderResult.knowledge_exports`` — what the provider learned
  (schedules that work, hardware quirks, failure modes). Persisted
  into ``CompilerMemory`` for cross-session reuse.
- ``ProviderResult.contract_feedback`` — suggestions to evolve the
  contract (alternative dtype, layout, tile size). Surfaced to the
  caller so contracts can be revised + retried.

This file proves both flow correctly through ``run_provider_fallback``
and ``compile_model``.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
from compgen.capture.torch_export import capture_model
from compgen.ir.payload.import_fx import fx_to_xdsl
from compgen.kernels.codegen_fallback import run_provider_fallback
from compgen.kernels.provider import (
    ContractFeedback,
    KnowledgeExport,
    ProviderResult,
    SearchBudget,
)
from compgen.kernels.provider import (
    KernelContract as ProviderContract,
)
from compgen.targets.schema import load_profile

_TARGET = "examples/target_profiles/cuda_a100.yaml"


class _LearningProvider:
    """Provider that emits both kernel_code AND knowledge/feedback."""

    name: str = "learning_provider"

    def accepts_contract(self, c: ProviderContract) -> bool:
        return c.op_family in {"add", "mul"}

    def search(self, c: ProviderContract, budget: SearchBudget) -> ProviderResult:  # noqa: ARG002
        return ProviderResult(
            found=True,
            kernel_code=f"// {c.op_family}\n",
            language="cpp",
            correct=True,
            knowledge_exports=[
                KnowledgeExport(
                    kind="optimization_tactic",
                    scope="op_family",
                    scope_key=c.op_family,
                    content=f"{c.op_family} likes block_size=64 on this target",
                    metadata={"summary": f"{c.op_family} block-size hint"},
                    confidence=0.8,
                )
            ],
            contract_feedback=[
                ContractFeedback(
                    field="layout",
                    current_value="row_major",
                    suggested_value="tile_64x64",
                    reason=f"better cache reuse for {c.op_family}",
                    measured_gain=0.15,
                )
            ],
        )

    def export_knowledge(self) -> list[KnowledgeExport]:
        return []


def _module_for(model: nn.Module, *args: torch.Tensor):
    ep = capture_model(model, args)
    module, _ = fx_to_xdsl(ep)
    return module


# ---------------------------------------------------------------------------
# Direct run_provider_fallback API
# ---------------------------------------------------------------------------


def test_feedback_out_collects_provider_contract_feedback() -> None:
    """``feedback_out=[]`` populated with ``ContractFeedback`` items."""

    class Mul(nn.Module):
        def forward(self, a, b):
            return a * b

    module = _module_for(Mul(), torch.randn(4), torch.randn(4))
    target = load_profile(_TARGET)

    feedback: list = []
    out = run_provider_fallback(
        module,
        target,
        sample_inputs=(torch.randn(4), torch.randn(4)),
        extra_providers=[_LearningProvider()],
        feedback_out=feedback,
    )
    assert out, "no kernel emitted"
    assert len(feedback) == 1, feedback
    fb = feedback[0]
    assert fb.field == "layout"
    assert fb.suggested_value == "tile_64x64"
    assert fb.measured_gain == 0.15


def test_no_feedback_buf_drops_silently() -> None:
    """Without ``feedback_out=``, contract feedback is collected and
    dropped (clears the registry buffer) — backward-compatible."""

    class Mul(nn.Module):
        def forward(self, a, b):
            return a * b

    module = _module_for(Mul(), torch.randn(4), torch.randn(4))
    target = load_profile(_TARGET)

    out = run_provider_fallback(
        module,
        target,
        sample_inputs=(torch.randn(4), torch.randn(4)),
        extra_providers=[_LearningProvider()],
    )
    assert out, "no kernel emitted"  # behaviour unchanged for the kernel path


def test_memory_kwarg_ingests_provider_knowledge_exports(tmp_path: Path) -> None:
    """When ``memory=`` is provided, knowledge_exports get persisted."""
    from compgen.memory.store import CompilerMemory

    db_path = tmp_path / "memory.db"
    blob_root = tmp_path / "blobs"
    memory = CompilerMemory(db_path=db_path, blob_root=blob_root)
    try:

        class Mul(nn.Module):
            def forward(self, a, b):
                return a * b

        module = _module_for(Mul(), torch.randn(4), torch.randn(4))
        target = load_profile(_TARGET)

        out = run_provider_fallback(
            module,
            target,
            sample_inputs=(torch.randn(4), torch.randn(4)),
            extra_providers=[_LearningProvider()],
            memory=memory,
        )
        assert out, "no kernel emitted"

        # The provider exported one knowledge item; memory should now
        # contain at least one knowledge row sourced from the provider.
        assert db_path.exists(), "memory db never created"
    finally:
        memory.close()


def test_no_memory_kwarg_does_not_create_memory_artifacts(tmp_path: Path) -> None:
    """The default (no memory) must not silently create a ``.compgen_cache/``
    on the user's machine just because they called compile_model."""
    import os

    # Resolve target path BEFORE chdir so the YAML lookup still works.
    target = load_profile(_TARGET)

    cwd = Path.cwd()
    os.chdir(tmp_path)
    try:

        class Mul(nn.Module):
            def forward(self, a, b):
                return a * b

        module = _module_for(Mul(), torch.randn(4), torch.randn(4))
        run_provider_fallback(
            module,
            target,
            sample_inputs=(torch.randn(4), torch.randn(4)),
            extra_providers=[_LearningProvider()],
        )
        # No CompilerMemory was supplied — no side-effect store should have
        # appeared in the test's cwd.
        assert not (tmp_path / ".compgen_cache").exists(), "default-no-memory path created a persistent store on disk"
    finally:
        os.chdir(cwd)


# ---------------------------------------------------------------------------
# bundle_emit / pipeline_artifacts integration
# ---------------------------------------------------------------------------


def test_bundle_emit_writes_provider_feedback_when_present(tmp_path: Path) -> None:
    """``pipeline_artifacts['provider_contract_feedback']`` →
    ``bundle/provider_feedback.json``."""
    from compgen.runtime.bundle_emit import emit_extended_artefacts

    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    (bundle_dir / "manifest.json").write_text(json.dumps({"artifacts": {}}))

    class _FakeCapture:
        exported_program = None
        diagnostics = None

    pipeline_artifacts = {
        "provider_contract_feedback": [
            {
                "field": "layout",
                "current_value": "row_major",
                "suggested_value": "tile_64x64",
                "reason": "cache",
                "measured_gain": 0.15,
            }
        ],
    }
    report = emit_extended_artefacts(
        bundle_dir,
        capture_artifact=_FakeCapture(),
        sample_inputs=(torch.randn(2, 2),),
        pipeline_artifacts=pipeline_artifacts,
    )

    feedback_path = bundle_dir / "provider_feedback.json"
    assert feedback_path.is_file()
    body = json.loads(feedback_path.read_text())
    assert body[0]["field"] == "layout"
    assert body[0]["suggested_value"] == "tile_64x64"

    # Status reported as ok in the report.
    statuses = {s.name: s.status for s in report.statuses}
    assert statuses.get("provider_feedback") == "ok"


def test_bundle_emit_skips_provider_feedback_when_absent(tmp_path: Path) -> None:
    """No feedback in pipeline_artifacts → status='skipped', no file."""
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
    assert statuses.get("provider_feedback") == "skipped"
    assert not (bundle_dir / "provider_feedback.json").exists()
