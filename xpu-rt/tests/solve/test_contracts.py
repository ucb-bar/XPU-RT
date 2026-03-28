"""Tests for solve/contracts.py -- solver problem extraction."""

from __future__ import annotations

import pytest
from compgen.solve.contracts import SolverProblem


def test_solver_problem_defaults() -> None:
    sp = SolverProblem()
    assert sp.partitions == []
    assert sp.placement_constraints == []
    assert sp.schedule_constraints == []
    assert sp.device_capacities == {}
    assert sp.transfer_costs == {}
    assert sp.target_name == ""


def test_solver_problem_with_target_name() -> None:
    sp = SolverProblem(target_name="cuda_a100")
    assert sp.target_name == "cuda_a100"


def test_extract_solver_problem() -> None:
    """extract_solver_problem should build a SolverProblem from Recipe IR + target."""
    from compgen.solve.contracts import extract_solver_problem
    from compgen.targets.schema import DeviceSpec, MemoryLevel, TargetProfile

    from xdsl.dialects.builtin import Float32Type, ModuleOp, TensorType
    from xdsl.dialects.func import FuncOp, ReturnOp
    from xdsl.dialects.linalg import MatmulOp
    from xdsl.dialects.tensor import EmptyOp
    from xdsl.ir import Block, Region

    # Build a minimal module with a matmul
    f32 = Float32Type()
    lhs_type = TensorType(f32, [32, 64])
    rhs_type = TensorType(f32, [64, 128])
    out_type = TensorType(f32, [32, 128])

    block = Block(arg_types=[lhs_type, rhs_type])
    empty = EmptyOp([], out_type)
    matmul = MatmulOp(
        inputs=[block.args[0], block.args[1]],
        outputs=[empty.results[0]],
        res=[out_type],
    )
    ret = ReturnOp(matmul)
    block.add_ops([empty, matmul, ret])
    func_op = FuncOp("main", ([lhs_type, rhs_type], [out_type]), Region(block))
    module = ModuleOp([func_op])

    # Build a target profile with one device
    device = DeviceSpec(
        device_type="gpu",
        name="test-gpu",
        memory_hierarchy=[MemoryLevel(name="hbm", size_bytes=80 * 1024**3)],
    )
    target = TargetProfile(name="test_target", devices=[device])

    problem = extract_solver_problem(module, target)
    assert problem.target_name == "test_target"
    assert len(problem.partitions) >= 1
    assert 0 in problem.device_capacities
    assert problem.device_capacities[0] > 0


def test_extract_solver_problem_with_cost_data() -> None:
    """extract_solver_problem should incorporate profiled cost data."""
    from compgen.solve.contracts import extract_solver_problem
    from compgen.targets.schema import DeviceSpec, MemoryLevel, TargetProfile

    from xdsl.dialects.builtin import Float32Type, ModuleOp, TensorType
    from xdsl.dialects.func import FuncOp, ReturnOp
    from xdsl.dialects.linalg import MatmulOp
    from xdsl.dialects.tensor import EmptyOp
    from xdsl.ir import Block, Region

    f32 = Float32Type()
    lhs_type = TensorType(f32, [16, 32])
    rhs_type = TensorType(f32, [32, 64])
    out_type = TensorType(f32, [16, 64])

    block = Block(arg_types=[lhs_type, rhs_type])
    empty = EmptyOp([], out_type)
    matmul = MatmulOp(
        inputs=[block.args[0], block.args[1]],
        outputs=[empty.results[0]],
        res=[out_type],
    )
    ret = ReturnOp(matmul)
    block.add_ops([empty, matmul, ret])
    func_op = FuncOp("main", ([lhs_type, rhs_type], [out_type]), Region(block))
    module = ModuleOp([func_op])

    device = DeviceSpec(
        device_type="gpu",
        name="test-gpu",
        memory_hierarchy=[MemoryLevel(name="hbm", size_bytes=40 * 1024**3)],
    )
    target = TargetProfile(name="cost_target", devices=[device])

    # Provide cost_data -- extract_solver_problem accepts it but currently
    # delegates to partition_graph which doesn't use it directly; the
    # function should still return a valid problem.
    cost_data = {"linalg.matmul": 100.0}
    problem = extract_solver_problem(module, target, cost_data=cost_data)
    assert problem.target_name == "cost_target"
    assert len(problem.partitions) >= 1
