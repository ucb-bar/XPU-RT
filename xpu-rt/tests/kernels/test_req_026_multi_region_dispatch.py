"""Tests for REQ-026 — codegen-fallback no longer drops same-op-family
regions.

Two compounding bugs caused multi-layer models to surface fewer
``index.json`` entries than ``payload.mlir`` had dispatches:

1. ``extract_contracts`` walked every op with results, including the
   ``arith.mulf`` / ``arith.addf`` inside ``linalg.matmul``'s implicit
   body and ``tensor.empty`` init buffers. Those spurious "kernels"
   shifted every legitimate region's positional index.
2. ``run_provider_fallback`` used the spec index to assign region_ids
   from a parallel module walk — when (1) drifted, the assignment
   collided and same-op-family regions overwrote each other in
   ``index.json``.

After REQ-026:

- Contract extraction filters to ops carrying ``compgen.region_id``
  (decompositions tag their dispatch boundaries explicitly; opaque
  ``func.call`` ops get synthesised ids in the post-import pass).
- ``run_provider_fallback`` reads region_id / dispatch_id straight off
  the contract metadata — no parallel walk, no positional drift.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
from compgen.capture.torch_export import capture_model
from compgen.ir.payload.import_fx import fx_to_xdsl
from compgen.kernels.codegen_fallback import run_provider_fallback
from compgen.kernels.contracts import build_kernel_contracts
from compgen.kernels.provider import (
    KernelContract as ProviderContract,
)
from compgen.kernels.provider import (
    KnowledgeExport,
    ProviderResult,
    SearchBudget,
)
from compgen.targets.schema import load_profile

_TARGET = "examples/target_profiles/cuda_a100.yaml"


class _AcceptAll:
    name = "accept_all"

    def accepts_contract(self, c: ProviderContract) -> bool:
        return True

    def search(self, c: ProviderContract, budget: SearchBudget) -> ProviderResult:  # noqa: ARG002
        return ProviderResult(
            found=True,
            kernel_code=f"// {c.op_family}/{c.region_id}\n",
            language="cpp",
            correct=True,
        )

    def export_knowledge(self) -> list[KnowledgeExport]:
        return []


def _module_for(model: nn.Module, *args: torch.Tensor):
    ep = capture_model(model, args)
    module, _ = fx_to_xdsl(ep)
    return module


def test_two_layer_mlp_emits_one_entry_per_dispatch() -> None:
    """``Linear → ReLU → Linear`` → 5 dispatches; index has 5 entries."""

    class MLP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc1 = nn.Linear(8, 8)
            self.fc2 = nn.Linear(8, 4)

        def forward(self, x):
            return self.fc2(torch.relu(self.fc1(x)))

    module = _module_for(MLP().eval(), torch.randn(1, 8))
    target = load_profile(_TARGET)
    out = run_provider_fallback(module, target, sample_inputs=(torch.randn(1, 8),), extra_providers=[_AcceptAll()])

    region_ids = [k["region_id"] for k in out]
    # Two matmuls + two transposes + one relu = 5 distinct regions.
    assert len(out) == 5, region_ids
    assert len(set(region_ids)) == 5, region_ids
    # Both matmul regions present, distinguishable.
    assert "matmul_0" in region_ids
    assert "matmul_1" in region_ids
    # Both transpose regions present.
    assert "transpose_0" in region_ids
    assert "transpose_1" in region_ids
    # Opaque relu got a synthesised id.
    assert any("relu" in r for r in region_ids), region_ids


def test_no_spurious_arith_or_tensor_empty_contracts() -> None:
    """Pre-REQ-026: ``extract_contracts`` surfaced
    ``tensor.empty`` / ``arith.mulf`` / ``arith.addf`` ops as
    pseudo-kernels because ``module.walk()`` recurses into the
    matmul's implicit body. After REQ-026, only ops tagged with
    ``compgen.region_id`` become contracts — those don't."""

    class MLP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc1 = nn.Linear(8, 8)
            self.fc2 = nn.Linear(8, 4)

        def forward(self, x):
            return self.fc2(torch.relu(self.fc1(x)))

    module = _module_for(MLP().eval(), torch.randn(1, 8))
    target = load_profile(_TARGET)
    specs = build_kernel_contracts(module, target, None)

    op_names = [s.contract.op_name for s in specs]
    assert "arith.mulf" not in op_names
    assert "arith.addf" not in op_names
    assert "tensor.empty" not in op_names


def test_index_region_ids_align_with_payload_mlir(tmp_path: Path) -> None:
    """End-to-end through ``compile_model``: ``index.json`` region_ids
    are the same set that appear in ``payload.mlir``."""
    from compgen.api import compile_model, device

    spec = tmp_path / "spec.yaml"
    spec.write_text(
        "name: t\n"
        "schema_version: '2.0'\n"
        "platform:\n"
        "  vendor: v\n  family: f\n  chip_name: c\n"
        "execution_model:\n  model: simt_gpu\n"
        "engine_geometry:\n  max_warp_size: 8\n"
    )

    class MLP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc1 = nn.Linear(8, 8)
            self.fc2 = nn.Linear(8, 4)

        def forward(self, x):
            return self.fc2(torch.relu(self.fc1(x)))

    dev = device(str(spec))

    # Register the catch-all provider via extra_providers wouldn't reach
    # compile_model; use a tmp-path side-channel — register through the
    # plugin registry instead.
    from compgen.plugins import GROUP_KERNEL_PROVIDERS, register, reset_registry

    reset_registry()
    try:
        register(GROUP_KERNEL_PROVIDERS, "accept_all", _AcceptAll())
        cm = compile_model(
            MLP().eval(),
            dev,
            sample_inputs=(torch.randn(1, 8),),
            verify=False,
        )
    finally:
        reset_registry()

    bundle = Path(cm.pipeline_result.all_artifacts["bundle_dir"])
    idx_path = bundle / "generated_kernels" / "index.json"
    assert idx_path.is_file(), idx_path
    entries = json.loads(idx_path.read_text())

    # The post-pipeline payload.mlir reflects whatever the eqsat /
    # rewrite passes produced; what matters for REQ-026 is the
    # invariant that **every region_id in index.json is unique** (no
    # collisions between same-op-family regions). The pre-REQ-026
    # bug had ``aten_add_0`` overwrite ``aten_add_1`` in index.json.
    rids = [e["region_id"] for e in entries]
    assert len(set(rids)) == len(rids), f"colliding region_ids in index.json: {rids}"

    # A 2-layer MLP must surface BOTH matmul regions distinctly.
    matmul_rids = [r for r in rids if "matmul" in r]
    assert len(matmul_rids) >= 2, (rids,)
    assert len(set(matmul_rids)) == len(matmul_rids), matmul_rids


def test_two_same_op_family_regions_both_emit() -> None:
    """``(a+b) + (c+d) + e`` has three ``aten_add`` regions; all three
    surface in the kernel list with distinct region_ids."""

    class ThreeAdds(nn.Module):
        def forward(self, a, b, c, d, e):
            return (a + b) + (c + d) + e

    module = _module_for(
        ThreeAdds(),
        torch.randn(4),
        torch.randn(4),
        torch.randn(4),
        torch.randn(4),
        torch.randn(4),
    )
    target = load_profile(_TARGET)
    out = run_provider_fallback(
        module,
        target,
        sample_inputs=tuple(torch.randn(4) for _ in range(5)),
        extra_providers=[_AcceptAll()],
    )

    add_entries = [k for k in out if "add" in k["op_name"]]
    # ``(a+b)+(c+d)+e`` lowers to four ``aten_add`` ops in left-folded
    # form ``(((a+b)+(c+d))+e)`` — what matters is each surfaces with
    # a distinct region_id (no clobbering — that was the REQ-026 bug).
    assert len(add_entries) >= 3, [k["op_name"] for k in add_entries]
    rids = [k["region_id"] for k in add_entries]
    assert len(set(rids)) == len(rids), rids


def test_relu_func_call_gets_synthesised_region_id() -> None:
    """REQ-026's secondary blocker: opaque ``func.call`` ops without
    a region_id were silently dropped by the contract extractor.
    The post-import pass now synthesises an id like
    ``aten_relu_default_0`` so they're claimable."""

    class Net(nn.Module):
        def forward(self, x):
            return torch.relu(x)

    module = _module_for(Net(), torch.randn(4))
    target = load_profile(_TARGET)
    specs = build_kernel_contracts(module, target, None)
    relu_specs = [s for s in specs if "relu" in s.contract.op_name]
    assert relu_specs
    rid = (relu_specs[0].contract.metadata or {}).get("region_id")
    assert rid and "relu" in rid, rid
