"""Tests for the P6 AutocompCostBudget addition to KernelContract."""

from __future__ import annotations

import pytest

from compgen.ir.payload.contracts import (
    AutocompCostBudget,
    CostEstimate,
    KernelContract,
    LayoutKind,
    LayoutRequirement,
)


def test_legacy_contract_has_none_budget() -> None:
    k = KernelContract(op_name="linalg.matmul")
    assert k.autocomp_cost_budget is None


def test_budget_defaults() -> None:
    b = AutocompCostBudget(max_wall_seconds=60.0, max_candidates=20)
    assert b.early_stop_if_matches_library is True
    assert b.budget_source == "derived_from_target_model"


def test_contract_with_budget() -> None:
    b = AutocompCostBudget(
        max_wall_seconds=120.0,
        max_candidates=30,
        early_stop_if_matches_library=False,
        budget_source="explicit",
    )
    k = KernelContract(op_name="linalg.matmul", autocomp_cost_budget=b)
    assert k.autocomp_cost_budget is b
    assert k.autocomp_cost_budget.max_wall_seconds == 120.0
    assert k.autocomp_cost_budget.budget_source == "explicit"


def test_budget_is_frozen() -> None:
    b = AutocompCostBudget(max_wall_seconds=60.0, max_candidates=20)
    with pytest.raises(Exception):  # dataclass(frozen=True) → FrozenInstanceError
        b.max_wall_seconds = 9999.0   # type: ignore[misc]


def test_contract_without_budget_backward_compatible() -> None:
    # Original construction path (all pre-v2 fields) must still work.
    k = KernelContract(
        op_name="linalg.matmul",
        input_layouts=[LayoutRequirement(kind=LayoutKind.ROW_MAJOR)],
        output_layouts=[LayoutRequirement(kind=LayoutKind.ROW_MAJOR)],
        supported_dtypes={"float32", "bfloat16"},
        cost=CostEstimate(flops=1_000_000, bytes_read=4096),
        fusable=False,
    )
    assert k.autocomp_cost_budget is None
    assert k.fusable is False
    assert "bfloat16" in k.supported_dtypes
