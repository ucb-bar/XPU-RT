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
    """build_kernel_contracts on simple arith ops."""
    from xdsl.dialects import arith, func
    from xdsl.dialects.builtin import IndexType, ModuleOp
    from xdsl.ir import Block, Region

    idx = IndexType()
    block = Block(arg_types=[idx, idx])
    a, b = block.args
    add = arith.AddiOp(a, b)
    block.add_op(add)
    block.add_op(func.ReturnOp(add.result))
    module = ModuleOp([func.FuncOp("test", ([idx, idx], [idx]), Region([block]))])

    target = load_profile("examples/target_profiles/cuda_a100.yaml")
    specs = build_kernel_contracts(module, target)
    assert len(specs) >= 1
    assert specs[0].contract.op_name == "arith.addi"
