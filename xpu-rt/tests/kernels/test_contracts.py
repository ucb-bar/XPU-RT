"""Tests for kernels/contracts.py -- kernel contract definitions."""

from __future__ import annotations

import torch
from compgen.ir.payload.contracts import KernelContract, extract_contracts
from compgen.kernels.contracts import KernelSearchPlan, KernelSpec, build_kernel_contracts
from compgen.targets.schema import load_profile


def test_kernel_spec_defaults() -> None:
    contract = KernelContract(op_name="matmul")
    spec = KernelSpec(contract=contract)
    assert spec.contract.op_name == "matmul"
    assert spec.input_shapes == []
    assert spec.output_shapes == []
    assert spec.reference_code == ""
    assert spec.perf_target_us is None
    assert spec.priority == 0


def test_kernel_search_plan_defaults() -> None:
    contract = KernelContract(op_name="relu")
    spec = KernelSpec(contract=contract)
    plan = KernelSearchPlan(spec=spec, strategy="autocomp")
    assert plan.strategy == "autocomp"
    assert plan.search_budget == 50
    assert plan.backends == ["triton"]
    assert plan.constraints == {}


def test_extract_contracts_from_matmul() -> None:
    """extract_contracts should produce contracts from a real FX-imported module."""
    from compgen.capture.torch_export import capture_model
    from compgen.ir.payload.import_fx import fx_to_xdsl

    class SimpleMLP(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(32, 16)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.linear(x)

    ep = capture_model(SimpleMLP(), (torch.randn(4, 32),))
    module, _ = fx_to_xdsl(ep)
    contracts = extract_contracts(module)
    assert len(contracts) > 0
    # All contracts should have op_name
    for c in contracts:
        assert c.op_name


def test_build_kernel_contracts_from_matmul() -> None:
    """build_kernel_contracts should produce sorted KernelSpecs."""
    from compgen.capture.torch_export import capture_model
    from compgen.ir.payload.import_fx import fx_to_xdsl

    class SimpleMLP(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(32, 16)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.linear(x)

    ep = capture_model(SimpleMLP(), (torch.randn(4, 32),))
    module, _ = fx_to_xdsl(ep)
    target = load_profile("examples/target_profiles/cuda_a100.yaml")
    specs = build_kernel_contracts(module, target)
    assert len(specs) > 0

    # Sorted by priority (highest first)
    priorities = [s.priority for s in specs]
    assert priorities == sorted(priorities, reverse=True)


def test_build_kernel_contracts_from_arith() -> None:
    """REQ-026: only ops carrying ``compgen.region_id`` surface as
    contracts. Ops without that annotation (default for raw arith /
    tensor.empty / structured-op body) are deliberately filtered to
    avoid drowning the dispatch list in non-kernel pseudo-ops.

    A synthetic ``arith.addi`` with the annotation surfaces; without
    it, it doesn't.
    """
    from xdsl.dialects import arith, func
    from xdsl.dialects.builtin import IndexType, ModuleOp, StringAttr
    from xdsl.ir import Block, Region

    idx = IndexType()

    # Untagged arith op — does NOT surface as a contract (REQ-026).
    block_a = Block(arg_types=[idx, idx])
    a, b = block_a.args
    add_a = arith.AddiOp(a, b)
    block_a.add_op(add_a)
    block_a.add_op(func.ReturnOp(add_a.result))
    untagged = ModuleOp([func.FuncOp("test", ([idx, idx], [idx]), Region([block_a]))])
    target = load_profile("examples/target_profiles/cuda_a100.yaml")
    assert build_kernel_contracts(untagged, target) == []

    # Tagged with ``compgen.region_id`` → surfaces.
    block_b = Block(arg_types=[idx, idx])
    a2, b2 = block_b.args
    add_b = arith.AddiOp(a2, b2)
    add_b.attributes["compgen.region_id"] = StringAttr("addi_0")
    block_b.add_op(add_b)
    block_b.add_op(func.ReturnOp(add_b.result))
    tagged = ModuleOp([func.FuncOp("test2", ([idx, idx], [idx]), Region([block_b]))])
    specs = build_kernel_contracts(tagged, target)
    assert len(specs) == 1
    assert specs[0].contract.op_name == "arith.addi"
    assert (specs[0].contract.metadata or {}).get("region_id") == "addi_0"
