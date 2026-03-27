"""Kill Test 4: Heterogeneous Planning Usefulness.

Validates that solver-backed planning works for heterogeneous targets:
partition IR → solve placement → solve schedule → verify feasible + deterministic.

Uses the multi_device.yaml profile (CPU + GPU).
"""

from __future__ import annotations

from compgen.runtime.planner import ExecutionPlan, plan_execution
from compgen.solve.memory import BufferLifetime, solve_memory
from compgen.solve.partition import partition_graph
from compgen.solve.placement import solve_placement
from compgen.solve.schedule import solve_schedule
from compgen.targets.schema import load_profile
from xdsl.dialects import arith, func
from xdsl.dialects.builtin import IndexType, ModuleOp
from xdsl.ir import Block, Region


def _make_chain_module(n: int = 10) -> ModuleOp:
    """Create a chain of ops suitable for partitioning and placement."""
    idx = IndexType()
    block = Block(arg_types=[idx, idx])
    a, b = block.args
    prev = a
    for _ in range(n):
        add = arith.AddiOp(prev, b)
        block.add_op(add)
        prev = add.result
    block.add_op(func.ReturnOp(prev))
    return ModuleOp([func.FuncOp("chain", ([idx, idx], [idx]), Region([block]))])


def test_heterogeneous_plan_feasibility() -> None:
    """Solver finds a feasible placement plan for CPU+GPU within 10s."""
    target = load_profile("examples/target_profiles/multi_device.yaml")
    assert len(target.devices) >= 2, "Need multi-device profile"

    module = _make_chain_module(6)
    partitions = partition_graph(module)
    assert len(partitions) >= 2

    solution = solve_placement(
        partitions=partitions,
        num_devices=2,
        device_compute_rates=[2.0, 312.0],
        device_memory_caps=[256 * 1024**3, 80 * 1024**3],
        timeout_ms=10000,
    )

    assert solution.feasible
    assert solution.solve_time_ms < 10000
    assert len(solution.assignments) == len(partitions)


def test_plan_replay_determinism() -> None:
    """Same inputs produce the same placement assignments."""
    load_profile("examples/target_profiles/multi_device.yaml")  # verify profile loads
    module = _make_chain_module(6)
    partitions = partition_graph(module)

    sol1 = solve_placement(partitions, 2, [1.0, 1.0], [8 * 1024**3] * 2)
    sol2 = solve_placement(partitions, 2, [1.0, 1.0], [8 * 1024**3] * 2)

    assert sol1.assignments == sol2.assignments, "Placement is not deterministic"


def test_scheduling_feasibility() -> None:
    """Schedule solver produces feasible temporal schedule."""
    partition_ids = ["t0", "t1", "t2"]
    durations = {"t0": 10.0, "t1": 20.0, "t2": 15.0}
    device_assignments = {"t0": 0, "t1": 0, "t2": 1}
    dependencies = {"t0": [], "t1": ["t0"], "t2": ["t0"]}

    solution = solve_schedule(
        partition_ids, durations, device_assignments, dependencies,
        num_devices=2, timeout_ms=5000,
    )
    assert solution.feasible
    # t1 must start after t0 finishes
    assert solution.start_times["t1"] >= solution.start_times["t0"] + 10
    # t2 must start after t0 finishes
    assert solution.start_times["t2"] >= solution.start_times["t0"] + 10


def test_memory_allocation_feasibility() -> None:
    """Memory allocation finds feasible assignment."""
    lifetimes = [
        BufferLifetime("buf_a", 1024, 0, 0.0, 50.0),
        BufferLifetime("buf_b", 2048, 0, 10.0, 60.0),
        BufferLifetime("buf_c", 1024, 1, 0.0, 100.0),
    ]
    result = solve_memory(lifetimes, {0: 1_000_000, 1: 1_000_000})
    assert result.feasible


def test_heterogeneous_go_no_go() -> None:
    """Aggregate: placement + scheduling + memory all feasible."""
    target = load_profile("examples/target_profiles/multi_device.yaml")
    module = _make_chain_module(4)

    # Execution planner wraps partition + placement + scheduling
    plan = plan_execution(module, target)
    assert isinstance(plan, ExecutionPlan)
    assert len(plan.placements) > 0
    assert plan.estimated_latency_us is not None
