"""Tests for runtime/planner.py -- execution plan generation."""

from __future__ import annotations

from xdsl.builder import Builder, ImplicitBuilder
from xdsl.dialects.arith import ConstantOp
from xdsl.dialects.builtin import FloatAttr, Float32Type, ModuleOp, TensorType, f32
from xdsl.dialects.func import FuncOp, ReturnOp
from xdsl.ir import Block, Region

from compgen.runtime.planner import (
    CopyOp,
    ExecutionPlan,
    ExecutionPlanner,
    MemoryPlan,
    PlacementDecision,
    plan_execution,
)
from compgen.targets.schema import DeviceSpec, TargetProfile


def _make_trivial_module() -> ModuleOp:
    """Build a minimal xDSL ModuleOp with one constant op."""
    tensor_type = TensorType(f32, [2, 2])

    @Builder.implicit_region
    def body() -> None:
        @Builder.implicit_region((tensor_type,))
        def func_body(args: tuple) -> None:  # type: ignore[type-arg]
            ReturnOp(args[0])

        FuncOp("main", ([tensor_type], [tensor_type]), func_body)

    return ModuleOp(body)


def _make_single_device_profile() -> TargetProfile:
    return TargetProfile(
        name="test-single",
        devices=[DeviceSpec(device_type="gpu", name="test-gpu-0")],
    )


def _make_multi_device_profile() -> TargetProfile:
    return TargetProfile(
        name="test-multi",
        devices=[
            DeviceSpec(device_type="gpu", name="test-gpu-0"),
            DeviceSpec(device_type="gpu", name="test-gpu-1"),
        ],
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


def test_execution_planner_plan() -> None:
    """ExecutionPlanner.plan should produce a valid ExecutionPlan for a single device."""
    module = _make_trivial_module()
    profile = _make_single_device_profile()
    planner = ExecutionPlanner(target=profile)
    plan = planner.plan(module)

    assert isinstance(plan, ExecutionPlan)
    # Single device: everything on device 0, no copies
    for p in plan.placements:
        assert p.device_index == 0
    assert plan.copies == []
    assert plan.estimated_latency_us is not None


def test_plan_execution_convenience() -> None:
    """plan_execution should work with default settings."""
    module = _make_trivial_module()
    profile = _make_single_device_profile()
    plan = plan_execution(module, profile)

    assert isinstance(plan, ExecutionPlan)
    assert len(plan.execution_order) >= 0
    assert isinstance(plan.memory_plans, list)
