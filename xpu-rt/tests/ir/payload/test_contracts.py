"""Tests for kernel contracts."""

from __future__ import annotations

import pytest
from compgen.ir.payload.contracts import CostEstimate, KernelContract, LayoutKind, LayoutRequirement


def test_layout_kind_values() -> None:
    assert LayoutKind.ROW_MAJOR.value == "row_major"
    assert LayoutKind.COLUMN_MAJOR.value == "column_major"
    assert LayoutKind.CUSTOM_STRIDES.value == "custom_strides"
    assert LayoutKind.ANY.value == "any"


def test_layout_requirement_defaults() -> None:
    req = LayoutRequirement()
    assert req.kind == LayoutKind.ANY
    assert req.strides is None
    assert req.alignment == 1


def test_cost_estimate_defaults() -> None:
    cost = CostEstimate()
    assert cost.flops == 0
    assert cost.bytes_read == 0
    assert cost.bytes_written == 0
    assert cost.latency_us is None


def test_kernel_contract_construction() -> None:
    layout = LayoutRequirement(kind=LayoutKind.ROW_MAJOR, alignment=64)
    cost = CostEstimate(flops=1024, bytes_read=512, bytes_written=256)
    contract = KernelContract(
        op_name="linalg.matmul",
        input_layouts=[layout, layout],
        output_layouts=[layout],
        cost=cost,
        fusable=False,
    )
    assert contract.op_name == "linalg.matmul"
    assert len(contract.input_layouts) == 2
    assert contract.cost.flops == 1024
    assert contract.fusable is False
    assert "float32" in contract.supported_dtypes


@pytest.mark.skip(reason="scaffold only -- implementation pending")
def test_extract_contracts() -> None:
    """extract_contracts should walk an xDSL module and emit KernelContracts."""


@pytest.mark.skip(reason="scaffold only -- implementation pending")
def test_kernel_contract_yaml_serialization() -> None:
    """KernelContract should be serializable to YAML for LLM context."""
