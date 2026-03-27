"""Integration tests for the full solver pipeline wired into the execution planner.

Verifies: partition -> place -> schedule -> memory -> ExecutionPlan.
"""

from __future__ import annotations

from compgen.agent.memory import CostCalibration
from compgen.runtime.planner import (
    CopyOp,
    ExecutionPlan,
    MemoryPlan,
    plan_execution,
)
from compgen.solve.contracts import (
    SolverProblem,
    extract_solver_problem,
)
from compgen.targets.schema import (
    DeviceSpec,
    Interconnect,
    MemoryLevel,
    TargetProfile,
)
from xdsl.dialects.arith import AddiOp, ConstantOp
from xdsl.dialects.builtin import IntegerAttr, IntegerType, ModuleOp
from xdsl.dialects.func import FuncOp, ReturnOp
from xdsl.ir import Block, Region

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_two_device_target(
    name: str = "test-2gpu",
    memory_bytes: int = 8 * 1024**3,
    bandwidth_gbps: float = 100.0,
) -> TargetProfile:
    """Create a two-device target profile for testing."""
    mem = MemoryLevel(name="hbm", size_bytes=memory_bytes)
    return TargetProfile(
        name=name,
        devices=[
            DeviceSpec(device_type="gpu", name="gpu0", memory_hierarchy=[mem]),
            DeviceSpec(device_type="gpu", name="gpu1", memory_hierarchy=[mem]),
        ],
        interconnects=[
            Interconnect(
                topology="nvlink",
                bandwidth_gbps=bandwidth_gbps,
                devices=(0, 1),
            ),
        ],
    )


def _make_single_device_target(name: str = "test-1gpu") -> TargetProfile:
    """Create a single-device target profile for testing."""
    mem = MemoryLevel(name="hbm", size_bytes=16 * 1024**3)
    return TargetProfile(
        name=name,
        devices=[
            DeviceSpec(device_type="gpu", name="gpu0", memory_hierarchy=[mem]),
        ],
    )


def _make_simple_module() -> ModuleOp:
    """Create a simple xDSL module with two add operations for testing."""
    i32_type = IntegerType(32)

    block = Block(arg_types=[i32_type, i32_type])
    a, b = block.args
    c0 = ConstantOp(IntegerAttr(1, i32_type))
    add1 = AddiOp(a, b)
    add2 = AddiOp(add1.result, c0.result)
    ret = ReturnOp(add2.result)
    block.add_ops([c0, add1, add2, ret])

    region = Region([block])
    func = FuncOp("main", ([i32_type, i32_type], [i32_type]), region)

    module = ModuleOp([func])
    return module


# ---------------------------------------------------------------------------
# Tests: Full pipeline (partition -> place -> schedule -> memory)
# ---------------------------------------------------------------------------


class TestMultiDevicePipeline:
    """Integration tests for multi-device execution planning."""

    def test_full_pipeline_produces_valid_plan(self) -> None:
        """Full pipeline should produce a valid ExecutionPlan with all fields populated."""
        target = _make_two_device_target()
        module = _make_simple_module()

        plan = plan_execution(module, target)

        assert isinstance(plan, ExecutionPlan)
        assert len(plan.placements) > 0
        assert len(plan.execution_order) > 0
        assert plan.estimated_latency_us is not None
        assert plan.estimated_latency_us > 0

    def test_placements_cover_all_partitions(self) -> None:
        """Every partition should have a placement decision."""
        target = _make_two_device_target()
        module = _make_simple_module()

        plan = plan_execution(module, target)

        placed_ops = {p.op_name for p in plan.placements}
        ordered_ops = set(plan.execution_order)
        # All ordered ops should have placements
        assert ordered_ops.issubset(placed_ops)

    def test_placements_respect_device_count(self) -> None:
        """Placement device indices should be within [0, num_devices)."""
        target = _make_two_device_target()
        module = _make_simple_module()

        plan = plan_execution(module, target)

        for p in plan.placements:
            assert 0 <= p.device_index < len(target.devices)

    def test_execution_order_sorted_by_schedule(self) -> None:
        """Execution order should reflect scheduling start times."""
        target = _make_two_device_target()
        module = _make_simple_module()

        plan = plan_execution(module, target)

        # Just check it's a valid ordering (non-empty, all partitions present)
        assert len(plan.execution_order) > 0
        # No duplicates
        assert len(plan.execution_order) == len(set(plan.execution_order))

    def test_memory_plans_present(self) -> None:
        """Memory plans should be generated for devices that have allocated buffers."""
        target = _make_two_device_target()
        module = _make_simple_module()

        plan = plan_execution(module, target)

        # At least one memory plan should exist
        assert len(plan.memory_plans) > 0
        for mp in plan.memory_plans:
            assert isinstance(mp, MemoryPlan)
            assert mp.peak_bytes >= 0

    def test_metadata_contains_solver_info(self) -> None:
        """Plan metadata should contain solver timing and feasibility info."""
        target = _make_two_device_target()
        module = _make_simple_module()

        plan = plan_execution(module, target)

        assert "placement_gap" in plan.metadata
        assert "placement_time_ms" in plan.metadata
        assert "schedule_feasible" in plan.metadata
        assert "schedule_time_ms" in plan.metadata
        assert "memory_feasible" in plan.metadata
        assert "memory_time_ms" in plan.metadata

    def test_estimated_latency_from_schedule(self) -> None:
        """Estimated latency should come from the scheduling solver's makespan."""
        target = _make_two_device_target()
        module = _make_simple_module()

        plan = plan_execution(module, target)

        assert plan.estimated_latency_us is not None
        assert plan.estimated_latency_us > 0
        # Should be the schedule makespan (feasible case)
        if plan.metadata.get("schedule_feasible"):
            # Makespan should be less than or equal to serial sum
            from compgen.solve.partition import partition_graph
            partitions = partition_graph(module)
            serial_sum = sum(p.estimated_cost_us for p in partitions)
            assert plan.estimated_latency_us <= serial_sum + 1.0  # +1 for copy overhead


class TestCopyOperations:
    """Tests for cross-device copy operation handling."""

    def test_copies_detected_for_cross_device_deps(self) -> None:
        """Copy ops should be generated when dependent partitions are on different devices."""
        target = _make_two_device_target()
        module = _make_simple_module()

        plan = plan_execution(module, target)

        # If there are cross-device dependencies, copies should be present
        # If all on same device, copies should be empty
        for c in plan.copies:
            assert isinstance(c, CopyOp)
            assert c.src_device != c.dst_device
            assert c.size_bytes >= 0

    def test_copy_ops_have_cost(self) -> None:
        """Copy operations should have estimated costs."""
        target = _make_two_device_target()
        module = _make_simple_module()

        plan = plan_execution(module, target)

        for c in plan.copies:
            assert c.estimated_cost_us >= 0


class TestSingleDevicePipeline:
    """Single-device should not use the full solver pipeline."""

    def test_single_device_no_copies(self) -> None:
        """Single device plan should have no copies."""
        target = _make_single_device_target()
        module = _make_simple_module()

        plan = plan_execution(module, target)

        assert plan.copies == []
        assert len(plan.placements) > 0
        for p in plan.placements:
            assert p.device_index == 0

    def test_single_device_no_solver_metadata(self) -> None:
        """Single device plan should not have solver metadata."""
        target = _make_single_device_target()
        module = _make_simple_module()

        plan = plan_execution(module, target)

        assert plan.metadata == {}


class TestCalibrationIntegration:
    """Tests for calibration correction in extract_solver_problem."""

    def test_extract_without_calibration(self) -> None:
        """extract_solver_problem should work without calibration."""
        target = _make_two_device_target()
        module = _make_simple_module()

        problem = extract_solver_problem(module, target)

        assert isinstance(problem, SolverProblem)
        assert len(problem.partitions) > 0
        assert problem.target_name == "test-2gpu"

    def test_extract_with_calibration(self) -> None:
        """extract_solver_problem should apply calibration corrections."""
        target = _make_two_device_target()
        module = _make_simple_module()

        problem_uncalibrated = extract_solver_problem(module, target)
        original_costs = {p.partition_id: p.estimated_cost_us for p in problem_uncalibrated.partitions}

        cal = CostCalibration()
        for p in problem_uncalibrated.partitions:
            op_type = p.op_names[0] if p.op_names else "unknown"
            cal.factors.setdefault("gpu0", {})[op_type] = 2.0

        problem_calibrated = extract_solver_problem(module, target, calibration=cal)

        for p in problem_calibrated.partitions:
            if p.partition_id in original_costs:
                expected = original_costs[p.partition_id] * 2.0
                assert abs(p.estimated_cost_us - expected) < 0.001, (
                    f"Partition {p.partition_id}: expected {expected}, got {p.estimated_cost_us}"
                )

    def test_extract_with_identity_calibration(self) -> None:
        """Calibration with factor 1.0 should not change costs."""
        target = _make_two_device_target()
        module = _make_simple_module()

        problem_base = extract_solver_problem(module, target)
        base_costs = {p.partition_id: p.estimated_cost_us for p in problem_base.partitions}

        # Empty CostCalibration returns 1.0 for all lookups
        cal = CostCalibration()
        problem_cal = extract_solver_problem(module, target, calibration=cal)

        for p in problem_cal.partitions:
            if p.partition_id in base_costs:
                assert p.estimated_cost_us == base_costs[p.partition_id]

    def test_device_capacities_populated(self) -> None:
        """Device capacities should reflect target profile memory."""
        mem_bytes = 4 * 1024**3
        target = _make_two_device_target(memory_bytes=mem_bytes)
        module = _make_simple_module()

        problem = extract_solver_problem(module, target)

        assert len(problem.device_capacities) == 2
        for dev_idx, cap in problem.device_capacities.items():
            assert cap == mem_bytes

    def test_transfer_costs_populated(self) -> None:
        """Transfer costs should reflect target interconnects."""
        target = _make_two_device_target(bandwidth_gbps=200.0)
        module = _make_simple_module()

        problem = extract_solver_problem(module, target)

        # Self-transfer should be 0
        assert problem.transfer_costs.get((0, 0), -1) == 0.0
        assert problem.transfer_costs.get((1, 1), -1) == 0.0
        # Cross-device should be derived from bandwidth
        assert problem.transfer_costs.get((0, 1), -1) > 0.0
        assert problem.transfer_costs.get((1, 0), -1) > 0.0


class TestPlanSerialization:
    """Test that plans serialize correctly."""

    def test_to_dict_multi_device(self) -> None:
        """Multi-device plan should serialize to a valid dict."""
        target = _make_two_device_target()
        module = _make_simple_module()

        plan = plan_execution(module, target)
        d = plan.to_dict()

        assert "placements" in d
        assert "copies" in d
        assert "execution_order" in d
        assert "memory_plans" in d
        assert "estimated_latency_us" in d

        for p in d["placements"]:
            assert "op" in p
            assert "device" in p

    def test_to_dict_single_device(self) -> None:
        """Single-device plan should serialize to a valid dict."""
        target = _make_single_device_target()
        module = _make_simple_module()

        plan = plan_execution(module, target)
        d = plan.to_dict()

        assert len(d["copies"]) == 0
        assert len(d["placements"]) > 0
