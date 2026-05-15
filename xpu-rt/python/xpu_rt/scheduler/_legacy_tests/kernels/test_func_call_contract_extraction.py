"""Tests for ``func.call`` contract extraction.

When a payload IR has elementwise / unary ops wrapped behind
``func.call @aten_<op>(%a, %b)``, the contract built from the call
must reflect the *callee* — not the literal ``"call"`` token — and
must carry the operand/result tensor shapes.

Without this, every kernel provider downstream sees
``op_family='call'`` with empty shapes and refuses to accept,
defeating the codegen-fallback path.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from compgen.capture.torch_export import capture_model
from compgen.ir.payload.contracts import extract_contracts
from compgen.ir.payload.import_fx import fx_to_xdsl
from compgen.kernels.contracts import build_kernel_contracts, spec_to_provider_contract
from compgen.targets.schema import load_profile

_TARGET = "examples/target_profiles/cuda_a100.yaml"


def _module_for(model: nn.Module, *args: torch.Tensor):
    ep = capture_model(model, args)
    module, _ = fx_to_xdsl(ep)
    return module


# ---------------------------------------------------------------------------
# REQ-022 — dtype passthrough
# ---------------------------------------------------------------------------


def test_f16_add_surfaces_as_f16_contract() -> None:
    class Add(nn.Module):
        def forward(self, a, b):
            return a + b

    module = _module_for(
        Add(),
        torch.randn(4, dtype=torch.float16),
        torch.randn(4, dtype=torch.float16),
    )
    target = load_profile(_TARGET)
    specs = build_kernel_contracts(module, target, None)
    assert specs
    pc = spec_to_provider_contract(specs[0], "r0", target)
    # MLIR-canonical name; not the legacy PyTorch-style "float16".
    assert pc.dtypes == ("f16",), pc.dtypes


def test_bf16_add_surfaces_as_bf16_contract() -> None:
    class Add(nn.Module):
        def forward(self, a, b):
            return a + b

    module = _module_for(
        Add(),
        torch.zeros(4, dtype=torch.bfloat16),
        torch.zeros(4, dtype=torch.bfloat16),
    )
    target = load_profile(_TARGET)
    specs = build_kernel_contracts(module, target, None)
    assert specs
    pc = spec_to_provider_contract(specs[0], "r0", target)
    assert pc.dtypes == ("bf16",), pc.dtypes


def test_default_dtype_is_canonical_mlir_f32_not_legacy_float32() -> None:
    """Empty supported_dtypes → MLIR-canonical ``("f32",)``, not ``("float32",)``."""
    from compgen.ir.payload.contracts import (
        CostEstimate,
    )
    from compgen.ir.payload.contracts import (
        KernelContract as IRContract,
    )
    from compgen.kernels.contracts import KernelSpec, spec_to_provider_contract

    ir = IRContract(op_name="aten_add", supported_dtypes=set(), cost=CostEstimate(flops=4))
    spec = KernelSpec(contract=ir)
    target = load_profile(_TARGET)
    pc = spec_to_provider_contract(spec, "r0", target)
    assert pc.dtypes == ("f32",), pc.dtypes


def test_extract_contracts_resolves_call_to_callee_for_mul() -> None:
    class Mul(nn.Module):
        def forward(self, a, b):
            return a * b

    module = _module_for(Mul(), torch.randn(4), torch.randn(4))
    contracts = extract_contracts(module)
    call_contracts = [c for c in contracts if c.op_name.startswith("aten_")]
    assert call_contracts, f"no aten_* contract surfaced; got {[c.op_name for c in contracts]}"
    c = call_contracts[0]
    assert c.op_name == "aten_mul"
    assert c.metadata.get("input_shapes") == [(4,), (4,)]
    assert c.metadata.get("output_shapes") == [(4,)]


def test_provider_contract_has_clean_op_family_and_shapes_for_mul() -> None:
    class Mul(nn.Module):
        def forward(self, a, b):
            return a * b

    module = _module_for(Mul(), torch.randn(4), torch.randn(4))
    target = load_profile(_TARGET)
    specs = build_kernel_contracts(module, target, None)
    assert specs, "build_kernel_contracts produced nothing"

    pc = spec_to_provider_contract(specs[0], "r0", target)
    assert pc.op_family == "mul", f"got {pc.op_family!r}"
    assert pc.input_shapes == ((4,), (4,))
    assert pc.output_shapes == ((4,),)
    assert pc.dtypes, "dtypes should be populated from the IR"


def test_provider_contract_strips_aten_prefix_for_relu() -> None:
    class Relu(nn.Module):
        def forward(self, a):
            return torch.relu(a)

    module = _module_for(Relu(), torch.randn(4))
    target = load_profile(_TARGET)
    specs = build_kernel_contracts(module, target, None)
    assert specs, "build_kernel_contracts produced nothing"

    pc = spec_to_provider_contract(specs[0], "r0", target)
    assert pc.op_family == "relu", f"got {pc.op_family!r}"
    assert pc.input_shapes == ((4,),)
    assert pc.output_shapes == ((4,),)


def test_provider_contract_preserves_shapes_through_size_8() -> None:
    """Shapes survive end-to-end at non-default sizes (no hardcoded 4)."""

    class Add(nn.Module):
        def forward(self, a, b):
            return a + b

    module = _module_for(Add(), torch.randn(8), torch.randn(8))
    target = load_profile(_TARGET)
    specs = build_kernel_contracts(module, target, None)
    pc = spec_to_provider_contract(specs[0], "r0", target)
    assert pc.input_shapes == ((8,), (8,))
    assert pc.output_shapes == ((8,),)
    assert pc.op_family == "add"
