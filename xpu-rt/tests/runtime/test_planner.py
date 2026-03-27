"""Tests for runtime/planner.py -- execution plan generation."""

from __future__ import annotations

import pytest
from compgen.runtime.planner import (
    CopyOp,
    ExecutionPlan,
    MemoryPlan,
    PlacementDecision,
)


def test_placement_decision_construction() -> None:
    pd = PlacementDecision(op_name="matmul_0", device_index=1, reason="gpu has fused kernel")
    assert pd.op_name == "matmul_0"
    assert pd.device_index == 1
    assert pd.reason == "gpu has fused kernel"


def test_placement_decision_default_reason() -> None:
    pd = PlacementDecision(op_name="relu_0", device_index=0)
    assert pd.reason == ""


def test_copy_op_defaults() -> None:
    c = CopyOp(tensor_name="x", src_device=0, dst_device=1)
    assert c.size_bytes == 0
    assert c.async_ is True


def test_memory_plan_defaults() -> None:
    mp = MemoryPlan(device_index=0)
    assert mp.peak_bytes == 0
    assert mp.allocations == []


def test_execution_plan_defaults() -> None:
    plan = ExecutionPlan()
    assert plan.placements == []
    assert plan.copies == []
    assert plan.execution_order == []
    assert plan.memory_plans == []
    assert plan.estimated_latency_us is None
    assert plan.metadata == {}


@pytest.mark.skip(reason="scaffold only -- implementation pending")
def test_execution_planner_plan() -> None:
    """ExecutionPlanner.plan should produce a valid ExecutionPlan."""


@pytest.mark.skip(reason="scaffold only -- implementation pending")
def test_plan_execution_convenience() -> None:
    """plan_execution should work with default settings."""
