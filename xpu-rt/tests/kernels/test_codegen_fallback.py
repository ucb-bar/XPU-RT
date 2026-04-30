"""Tests for codegen provider fallback.

When the auto-generated codegen stage's native emitter declines all
candidates (Triton non-friendly target, vendor stub returns nothing),
``run_provider_fallback`` must walk registered ``KernelProvider``
implementations and surface the first ``found=True`` result in the
shape ``compgen.runtime.bundle_emit`` consumes.

The contract-driven path is exercised against the in-tree
``TritonTemplateProvider`` so we observe a real Provider rendering
real source from a real ``KernelContract`` (op_family / shapes /
dtypes), not a stub returning a hardcoded string.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from compgen.kernels.codegen_fallback import run_provider_fallback
from compgen.kernels.provider import (
    KernelContract as ProviderContract,
)
from compgen.kernels.provider import (
    KnowledgeExport,
    ProviderResult,
    SearchBudget,
)
from compgen.kernels.providers.triton_templates import TritonTemplateProvider
from compgen.stages.templates.codegen import CODEGEN_BACKEND_ATTR
from compgen.targets.schema import load_profile
from xdsl.dialects.builtin import StringAttr


class _AcceptingStubProvider:
    """Minimal KernelProvider that records the contracts it sees.

    Used to verify that ``run_provider_fallback`` passes a populated
    :class:`ProviderContract` (op_family, shapes, dtypes, target_name)
    to ``search`` rather than calling it with empty/None fields.
    """

    name: str = "stub_provider"

    def __init__(self) -> None:
        self.seen_contracts: list[ProviderContract] = []

    def accepts_contract(self, contract: ProviderContract) -> bool:  # noqa: ARG002
        return True

    def search(self, contract: ProviderContract, budget: SearchBudget) -> ProviderResult:  # noqa: ARG002
        self.seen_contracts.append(contract)
        return ProviderResult(
            found=True,
            # Echo a contract field into the kernel so downstream
            # assertions can prove the provider actually inspected it.
            kernel_code=f"// stub kernel for op_family={contract.op_family}\n",
            language="cpp",
            correct=True,
        )

    def export_knowledge(self) -> list[KnowledgeExport]:
        return []


class _RejectingStubProvider:
    name: str = "rejecting_stub"

    def accepts_contract(self, contract: ProviderContract) -> bool:  # noqa: ARG002
        return False

    def search(self, contract: ProviderContract, budget: SearchBudget) -> ProviderResult:  # noqa: ARG002
        return ProviderResult(found=False)

    def export_knowledge(self) -> list[KnowledgeExport]:
        return []


def _build_module() -> object:
    """Return an xDSL ModuleOp with at least one extractable kernel contract."""
    from compgen.capture.torch_export import capture_model
    from compgen.ir.payload.import_fx import fx_to_xdsl

    class TinyMLP(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(32, 16)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.linear(x)

    ep = capture_model(TinyMLP(), (torch.randn(4, 32),))
    module, _ = fx_to_xdsl(ep)
    return module


def test_run_provider_fallback_emits_kernel_when_provider_accepts() -> None:
    """First accepting provider's kernel_code lands in the result list,
    and the provider is called with a fully-populated contract."""
    module = _build_module()
    target = load_profile("examples/target_profiles/cuda_a100.yaml")
    provider = _AcceptingStubProvider()

    out = run_provider_fallback(
        module,
        target,
        sample_inputs=(torch.randn(4, 32),),
        extra_providers=[provider],
    )

    assert isinstance(out, list)
    assert len(out) >= 1, "expected at least one kernel from accepting provider"
    first = out[0]
    assert first["provider"] == "stub_provider"
    assert first["language"] == "cpp"
    assert first["extension"] == "cpp"
    assert first["op_name"]
    # REQ-026: the region_id is whatever the IR op was tagged with —
    # ``matmul_0`` for a matmul, ``transpose_0`` for a transpose, etc.
    # The bare ``region_<i>`` synthesised id only surfaces when the IR
    # op had no ``compgen.region_id`` tag, which doesn't happen
    # post-fx-import anymore. Just assert it's non-empty.
    assert first["region_id"], first

    # Contract-shape assertions: the provider must have been called with
    # a populated ProviderContract, not an empty placeholder.
    assert provider.seen_contracts, "provider.search was never called"
    seen = provider.seen_contracts[0]
    assert seen.op_family, "contract.op_family was empty — extraction broken"
    assert seen.target_name == target.name
    assert seen.dtypes, "contract.dtypes was empty — IR didn't supply any"
    # The op family must round-trip into the rendered source — proves the
    # provider read it (rather than emitting a fixed string regardless).
    assert f"op_family={seen.op_family}" in first["source"]


def test_run_provider_fallback_no_providers_returns_empty() -> None:
    """No registered providers → empty result, no error."""
    module = _build_module()
    target = load_profile("examples/target_profiles/cuda_a100.yaml")

    out = run_provider_fallback(
        module,
        target,
        sample_inputs=(torch.randn(4, 32),),
        extra_providers=[],
    )

    assert out == []


def test_run_provider_fallback_skips_rejecting_providers() -> None:
    """A rejecting provider yields no kernels; an accepting one does."""
    module = _build_module()
    target = load_profile("examples/target_profiles/cuda_a100.yaml")

    rejected_only = run_provider_fallback(
        module,
        target,
        sample_inputs=(torch.randn(4, 32),),
        extra_providers=[_RejectingStubProvider()],
    )
    assert rejected_only == []

    mixed = run_provider_fallback(
        module,
        target,
        sample_inputs=(torch.randn(4, 32),),
        extra_providers=[_RejectingStubProvider(), _AcceptingStubProvider()],
    )
    assert len(mixed) >= 1
    assert mixed[0]["provider"] == "stub_provider"


def test_run_provider_fallback_rewrites_codegen_backend_annotation() -> None:
    """IR ops tagged ``compgen.codegen_backend = "fallback"`` flip to
    the winning provider's name after fallback succeeds."""
    module = _build_module()
    target = load_profile("examples/target_profiles/cuda_a100.yaml")

    # Tag every op carrying ``compgen.region_id`` with the "fallback"
    # sentinel (REQ-026: the rewrite is region-id keyed, not
    # walk-position keyed — tagging an arbitrary op without a region
    # id is now a no-op since codegen-fallback only rewrites
    # ops that have contracts).
    tagged_count = 0
    for op in module.walk():
        if "compgen.region_id" in op.attributes and op.results:
            op.attributes[CODEGEN_BACKEND_ATTR] = StringAttr("fallback")
            tagged_count += 1
    assert tagged_count >= 1, "expected at least one region-tagged op in the fixture module"

    out = run_provider_fallback(
        module,
        target,
        sample_inputs=(torch.randn(4, 32),),
        extra_providers=[_AcceptingStubProvider()],
    )
    assert out, "fallback should have produced kernels"

    rewritten = [
        op
        for op in module.walk()
        if (
            (attr := op.attributes.get(CODEGEN_BACKEND_ATTR)) is not None
            and isinstance(attr, StringAttr)
            and attr.data == "stub_provider"
        )
    ]
    assert rewritten, "no op had its codegen_backend rewritten"
    # And no surviving 'fallback' tag.
    surviving = [
        op
        for op in module.walk()
        if (
            (attr := op.attributes.get(CODEGEN_BACKEND_ATTR)) is not None
            and isinstance(attr, StringAttr)
            and attr.data == "fallback"
        )
    ]
    assert surviving == [], "fallback tag should have been rewritten on every tagged op"


def test_run_provider_fallback_with_real_provider_renders_from_contract() -> None:
    """End-to-end with a real generative Provider, not a stub.

    Uses the in-tree :class:`TritonTemplateProvider`, which renders Triton
    source from the ``KernelContract`` (op_family / shapes / dtypes). The
    fixture's ``Linear(32, 16)`` lowers to ``linalg.matmul`` which the
    provider's template covers. We assert:

    1. The provider was actually selected (``"triton_templates"``).
    2. The rendered source mentions the contract's op family + tile sizes
       derived from the contract shapes — confirming the Provider read
       the contract rather than emitting a fixed string.
    """
    module = _build_module()
    target = load_profile("examples/target_profiles/cuda_a100.yaml")

    out = run_provider_fallback(
        module,
        target,
        sample_inputs=(torch.randn(4, 32),),
        extra_providers=[TritonTemplateProvider()],
    )

    assert out, "TritonTemplateProvider should accept matmul contracts from Linear(32,16)"
    matmul_entries = [k for k in out if "matmul" in k["op_name"].lower()]
    assert matmul_entries, f"no matmul kernel in {[k['op_name'] for k in out]}"

    first_matmul = matmul_entries[0]
    assert first_matmul["provider"] == "triton_templates"
    assert first_matmul["language"] == "triton"
    assert first_matmul["extension"] == "py"

    src = first_matmul["source"]
    assert "import triton" in src or "@triton.jit" in src or "tl.program_id" in src, (
        "rendered source doesn't look like Triton — provider probably skipped templating"
    )
    # The matmul template parameterizes BLOCK_M / BLOCK_N / BLOCK_K from
    # the contract's shapes. Their literal values aren't a stable contract
    # (the picker can rebalance them), but their *presence* as constants
    # is — that's how we know the contract drove rendering instead of a
    # fixed string.
    assert "BLOCK_M" in src and "BLOCK_N" in src and "BLOCK_K" in src, (
        "rendered source missing block-size constants → contract shapes not consumed"
    )


def test_bundle_emit_writes_provider_kernels_to_disk(tmp_path: Path) -> None:
    """End-to-end: kernel entries → bundle/generated_kernels/<provider>/<op>.<ext>."""
    from compgen.runtime.bundle_emit import emit_extended_artefacts

    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    (bundle_dir / "manifest.json").write_text(json.dumps({"artifacts": {}}))

    class _FakeCapture:
        exported_program = None
        diagnostics = None

    pipeline_artifacts = {
        "generated_kernels": [
            {
                "provider": "stub_provider",
                "op_name": "linalg.matmul",
                "source": "// stub kernel\n",
                "extension": "cpp",
            },
        ],
    }

    report = emit_extended_artefacts(
        bundle_dir,
        capture_artifact=_FakeCapture(),
        sample_inputs=(torch.randn(2, 2),),
        pipeline_artifacts=pipeline_artifacts,
    )

    gk_dir = bundle_dir / "generated_kernels"
    assert gk_dir.is_dir()
    written = sorted(p.relative_to(bundle_dir).as_posix() for p in gk_dir.rglob("*.cpp"))
    assert written, "no provider kernel was written to bundle/generated_kernels"
    assert any("stub_provider" in p for p in written)
    assert "ok" in {s.status for s in report.statuses if s.name == "generated_kernels"}
