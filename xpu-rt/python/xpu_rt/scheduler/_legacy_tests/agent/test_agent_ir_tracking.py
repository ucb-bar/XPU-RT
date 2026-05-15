"""Integration tests for paired Recipe + Agent IR tracking."""

from __future__ import annotations

from compgen.agent.env import AssignDeviceAction, CompilerEnv
from compgen.ir.agent.lower import lower_agent
from compgen.ir.agent.validate import validate_agent_module
from compgen.targets.schema import load_profile
from xdsl.dialects import func, linalg, tensor
from xdsl.dialects.builtin import Float32Type, ModuleOp, TensorType
from xdsl.ir import Block, Region


def _make_matmul_module() -> ModuleOp:
    f32 = Float32Type()
    lhs_type = TensorType(f32, [4, 8])
    rhs_type = TensorType(f32, [8, 16])
    out_type = TensorType(f32, [4, 16])

    block = Block(arg_types=[lhs_type, rhs_type])
    lhs, rhs = block.args
    empty = tensor.EmptyOp([], out_type)
    block.add_op(empty)
    matmul = linalg.MatmulOp(inputs=[lhs, rhs], outputs=[empty.results[0]], res=[out_type])
    block.add_op(matmul)
    block.add_op(func.ReturnOp(matmul.results[0]))
    return ModuleOp([func.FuncOp("matmul_test", ([lhs_type, rhs_type], [out_type]), Region([block]))])


def test_enable_recipe_tracking_also_seeds_agent_ir() -> None:
    env = CompilerEnv()
    target = load_profile("examples/target_profiles/cuda_a100.yaml")
    env.reset(_make_matmul_module(), target, budget=5)

    env.enable_recipe_tracking()

    assert env.recipe is not None
    assert env.agent_ir is not None

    validation = validate_agent_module(env.agent_ir, recipe_module=env.recipe)
    assert validation.valid, validation.errors

    lowered = lower_agent(env.agent_ir)
    assert len(lowered.frontier_states) > 0


def test_applied_action_appends_agent_request() -> None:
    env = CompilerEnv()
    target = load_profile("examples/target_profiles/cuda_a100.yaml")
    obs = env.reset(_make_matmul_module(), target, budget=5)
    env.enable_recipe_tracking()

    assert obs.regions
    region_id = obs.regions[0].region_id
    result = env.step(AssignDeviceAction(region_id=region_id, device_index=0))

    assert result.info.action_applied
    assert env.agent_ir is not None

    validation = validate_agent_module(env.agent_ir, recipe_module=env.recipe)
    assert validation.valid, validation.errors

    lowered = lower_agent(env.agent_ir)
    assert any(job["request_type"] == "agent.request_backend_plan" for job in lowered.request_jobs)
