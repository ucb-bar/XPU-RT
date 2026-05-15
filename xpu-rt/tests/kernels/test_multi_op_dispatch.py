"""Tests for multi-op graph dispatch through ``run_provider_fallback``.

A graph with multiple ops (e.g. ``relu(a + b)``) must produce one
contract per op and dispatch each independently. The fallback
emitter must write one kernel per accepted contract into
``bundle/generated_kernels/`` with the correct per-region
``op_family`` and shapes — no cross-contamination between regions.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from compgen.capture.torch_export import capture_model
from compgen.ir.payload.import_fx import fx_to_xdsl
from compgen.kernels.codegen_fallback import run_provider_fallback
from compgen.kernels.contracts import build_kernel_contracts, spec_to_provider_contract
from compgen.kernels.provider import (
    KernelContract as ProviderContract,
)
from compgen.kernels.provider import (
    KnowledgeExport,
    ProviderResult,
    SearchBudget,
)
from compgen.targets.schema import load_profile
from xdsl.dialects.builtin import ModuleOp

_TARGET = "examples/target_profiles/cuda_a100.yaml"


class _RecordingProvider:
    """Provider that accepts elementwise ops and tags each kernel
    with the op_family + shapes it saw, so tests can prove there
    was no cross-region contamination.
    """

    name: str = "recording_provider"

    _ACCEPTED = frozenset({"add", "sub", "mul", "div", "relu"})

    def __init__(self) -> None:
        self.calls: list[ProviderContract] = []

    def accepts_contract(self, contract: ProviderContract) -> bool:
        return contract.op_family in self._ACCEPTED

    def search(self, contract: ProviderContract, budget: SearchBudget) -> ProviderResult:  # noqa: ARG002
        self.calls.append(contract)
        return ProviderResult(
            found=True,
            kernel_code=(
                f"// op_family={contract.op_family}\n"
                f"// input_shapes={contract.input_shapes}\n"
                f"// output_shapes={contract.output_shapes}\n"
                f"// region_id={contract.region_id}\n"
            ),
            language="cpp",
            correct=True,
        )

    def export_knowledge(self) -> list[KnowledgeExport]:
        return []


def _module_for(model: nn.Module, *args: torch.Tensor) -> ModuleOp:
    ep = capture_model(model, args)
    module, _ = fx_to_xdsl(ep)
    return module


def test_relu_of_add_yields_two_distinct_contracts() -> None:
    """``torch.relu(a + b)`` extracts as two contracts with right shapes."""

    class ReluAdd(nn.Module):
        def forward(self, a, b):
            return torch.relu(a + b)

    module = _module_for(ReluAdd(), torch.randn(4), torch.randn(4))
    target = load_profile(_TARGET)
    specs = build_kernel_contracts(module, target, None)
    assert len(specs) == 2, [s.contract.op_name for s in specs]

    contracts = [spec_to_provider_contract(s, f"r{i}", target) for i, s in enumerate(specs)]
    families = {c.op_family for c in contracts}
    assert families == {"add", "relu"}, families

    # Each op has its own shape arity: add is binary (2 inputs), relu unary.
    add_c = next(c for c in contracts if c.op_family == "add")
    relu_c = next(c for c in contracts if c.op_family == "relu")
    assert add_c.input_shapes == ((4,), (4,))
    assert relu_c.input_shapes == ((4,),)
    assert add_c.output_shapes == ((4,),)
    assert relu_c.output_shapes == ((4,),)


def test_run_provider_fallback_emits_one_kernel_per_region() -> None:
    """Multi-op graph → multiple kernel entries, distinct region_ids."""

    class ReluAdd(nn.Module):
        def forward(self, a, b):
            return torch.relu(a + b)

    module = _module_for(ReluAdd(), torch.randn(4), torch.randn(4))
    target = load_profile(_TARGET)
    provider = _RecordingProvider()

    out = run_provider_fallback(
        module,
        target,
        sample_inputs=(torch.randn(4), torch.randn(4)),
        extra_providers=[provider],
    )

    assert len(out) == 2, [(k["op_name"], k["region_id"]) for k in out]
    region_ids = {k["region_id"] for k in out}
    assert len(region_ids) == 2, f"region_ids collided: {region_ids}"

    # The provider must have been called twice with two different
    # contracts — not the same one twice.
    assert len(provider.calls) == 2
    seen_families = {c.op_family for c in provider.calls}
    assert seen_families == {"add", "relu"}


def test_per_kernel_source_carries_correct_op_family_no_crosstalk() -> None:
    """Each kernel's source reflects its own contract, not its sibling's."""

    class ReluAdd(nn.Module):
        def forward(self, a, b):
            return torch.relu(a + b)

    module = _module_for(ReluAdd(), torch.randn(4), torch.randn(4))
    target = load_profile(_TARGET)

    out = run_provider_fallback(
        module,
        target,
        sample_inputs=(torch.randn(4), torch.randn(4)),
        extra_providers=[_RecordingProvider()],
    )

    by_family = {}
    for k in out:
        # The recording provider stamps op_family= into the source so
        # we can verify there's no cross-region contamination.
        if "op_family=add" in k["source"]:
            by_family["add"] = k
        elif "op_family=relu" in k["source"]:
            by_family["relu"] = k

    assert set(by_family) == {"add", "relu"}
    # add kernel must have binary input_shapes; relu kernel unary.
    assert "input_shapes=((4,), (4,))" in by_family["add"]["source"]
    assert "input_shapes=((4,),)" in by_family["relu"]["source"]
    # Cross-check: the relu kernel must NOT mention the add shape pair,
    # and vice versa.
    assert "((4,), (4,))" not in by_family["relu"]["source"]
    assert "op_family=add" not in by_family["relu"]["source"]
    assert "op_family=relu" not in by_family["add"]["source"]


def test_three_op_chain_produces_three_kernels() -> None:
    """A longer chain — relu(a + b) * c — exercises N>2 dispatch."""

    class ReluAddMul(nn.Module):
        def forward(self, a, b, c):
            return torch.relu(a + b) * c

    module = _module_for(ReluAddMul(), torch.randn(4), torch.randn(4), torch.randn(4))
    target = load_profile(_TARGET)

    out = run_provider_fallback(
        module,
        target,
        sample_inputs=(torch.randn(4), torch.randn(4), torch.randn(4)),
        extra_providers=[_RecordingProvider()],
    )

    assert len(out) == 3, [(k["op_name"], k["region_id"]) for k in out]
    families: list[str] = []
    for k in out:
        for fam in ("add", "mul", "relu"):
            if f"op_family={fam}" in k["source"]:
                families.append(fam)
                break
    assert sorted(families) == ["add", "mul", "relu"], families


def test_multi_provider_per_region_provenance() -> None:
    """Two providers split a multi-op graph: each region gets its own
    ``compgen.codegen_backend`` annotation reflecting the actual winner.

    Regression for the single-provider-name limitation in the original
    REQ-008 implementation, where every "fallback"-tagged op was rewritten
    to a single ``winning_provider_name``.
    """
    from compgen.stages.templates.codegen import CODEGEN_BACKEND_ATTR
    from xdsl.dialects.builtin import StringAttr

    class AddOnlyProvider:
        name = "add_only_provider"

        def accepts_contract(self, c: ProviderContract) -> bool:
            return c.op_family == "add"

        def search(self, c: ProviderContract, budget: SearchBudget) -> ProviderResult:  # noqa: ARG002
            return ProviderResult(found=True, kernel_code="// add\n", language="cpp", correct=True)

        def export_knowledge(self) -> list[KnowledgeExport]:
            return []

    class ReluOnlyProvider:
        name = "relu_only_provider"

        def accepts_contract(self, c: ProviderContract) -> bool:
            return c.op_family == "relu"

        def search(self, c: ProviderContract, budget: SearchBudget) -> ProviderResult:  # noqa: ARG002
            return ProviderResult(found=True, kernel_code="// relu\n", language="cpp", correct=True)

        def export_knowledge(self) -> list[KnowledgeExport]:
            return []

    class ReluAdd(nn.Module):
        def forward(self, a, b):
            return torch.relu(a + b)

    module = _module_for(ReluAdd(), torch.randn(4), torch.randn(4))
    target = load_profile(_TARGET)

    # Tag every kernel-eligible op with the "fallback" sentinel up front,
    # mirroring what CodegenStage.shared_passes does in the real pipeline.
    from xdsl.dialects.builtin import ModuleOp
    from xdsl.dialects.func import FuncOp, ReturnOp

    for op in module.walk():
        if isinstance(op, (ModuleOp, FuncOp, ReturnOp)):
            continue
        if op.results:
            op.attributes[CODEGEN_BACKEND_ATTR] = StringAttr("fallback")

    out = run_provider_fallback(
        module,
        target,
        sample_inputs=(torch.randn(4), torch.randn(4)),
        extra_providers=[AddOnlyProvider(), ReluOnlyProvider()],
    )

    assert len(out) == 2
    by_provider = {k["provider"]: k for k in out}
    assert set(by_provider) == {"add_only_provider", "relu_only_provider"}

    # The IR annotations must reflect each provider per region — not
    # collapsed to a single name.
    annotated: dict[str, int] = {}
    for op in module.walk():
        attr = op.attributes.get(CODEGEN_BACKEND_ATTR)
        if isinstance(attr, StringAttr):
            annotated[attr.data] = annotated.get(attr.data, 0) + 1

    assert annotated.get("add_only_provider") == 1, annotated
    assert annotated.get("relu_only_provider") == 1, annotated
    # No surviving "fallback" tag — both regions were claimed.
    assert annotated.get("fallback", 0) == 0, annotated


def test_partial_acceptance_emits_only_for_supported_ops() -> None:
    """Provider that rejects relu emits a kernel only for the add region."""

    class AddOnlyProvider:
        name = "add_only"

        def accepts_contract(self, c: ProviderContract) -> bool:
            return c.op_family == "add"

        def search(self, c: ProviderContract, budget: SearchBudget) -> ProviderResult:  # noqa: ARG002
            return ProviderResult(found=True, kernel_code="// add\n", language="cpp", correct=True)

        def export_knowledge(self) -> list[KnowledgeExport]:
            return []

    class ReluAdd(nn.Module):
        def forward(self, a, b):
            return torch.relu(a + b)

    module = _module_for(ReluAdd(), torch.randn(4), torch.randn(4))
    target = load_profile(_TARGET)

    out = run_provider_fallback(
        module,
        target,
        sample_inputs=(torch.randn(4), torch.randn(4)),
        extra_providers=[AddOnlyProvider()],
    )

    # Only the add region was accepted; relu was skipped (no provider).
    assert len(out) == 1
    assert "// add" in out[0]["source"]
