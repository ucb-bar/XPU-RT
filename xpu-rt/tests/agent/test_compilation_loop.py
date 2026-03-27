"""Tests for the agentic compilation loop."""

from __future__ import annotations

from compgen.agent.compilation_loop import AgenticCompilationLoop, CompilationResult
from compgen.agent.env import CompilerEnv
from compgen.llm.mock_client import MockLLMClient
from compgen.targets.schema import load_profile
from xdsl.dialects import arith, func
from xdsl.dialects.builtin import IndexType, ModuleOp
from xdsl.ir import Block, Region


def _make_module() -> ModuleOp:
    idx = IndexType()
    block = Block(arg_types=[idx, idx, idx])
    a, b, c = block.args
    add = arith.AddiOp(a, b)
    block.add_op(add)
    mul = arith.MuliOp(add.result, c)
    block.add_op(mul)
    block.add_op(func.ReturnOp(mul.result))
    return ModuleOp([func.FuncOp("test", ([idx, idx, idx], [idx]), Region([block]))])


def _make_mock_client() -> MockLLMClient:
    """Create a mock LLM that returns valid analysis + refinement responses."""
    client = MockLLMClient(strict=False)
    # Analysis response
    analysis = '[{"action_type": "eqsat", "target": "all", "reason": "algebraic", "expected_improvement": 5.0}]'
    client.add_response("optimization", analysis)
    # Refinement response
    refine = '{"action_type": "eqsat", "target_region": "", "parameters": {}, "reasoning": "try more"}'
    client.add_response("iteratively", refine)
    return client


def test_compilation_loop_runs() -> None:
    """The agentic loop runs without error on a simple module."""
    env = CompilerEnv()
    target = load_profile("examples/target_profiles/cuda_a100.yaml")
    module = _make_module()
    env.reset(module, target, budget=20)

    client = _make_mock_client()
    loop = AgenticCompilationLoop(llm_client=client, env=env, budget=5)
    result = loop.run(target)

    assert isinstance(result, CompilationResult)
    assert result.iterations_run >= 0
    assert result.initial_cost_us >= 0


def test_compilation_loop_tracks_history() -> None:
    """The loop records iteration history."""
    env = CompilerEnv()
    target = load_profile("examples/target_profiles/cuda_a100.yaml")
    module = _make_module()
    env.reset(module, target, budget=20)

    client = _make_mock_client()
    loop = AgenticCompilationLoop(llm_client=client, env=env, budget=3)
    result = loop.run(target)

    assert len(result.history) == result.iterations_run
    for record in result.history:
        assert record.action_type


def test_compilation_loop_stops_on_noop() -> None:
    """The loop stops when LLM suggests noop."""
    env = CompilerEnv()
    target = load_profile("examples/target_profiles/cuda_a100.yaml")
    module = _make_module()
    env.reset(module, target, budget=20)

    client = MockLLMClient(strict=False)
    # Return empty analysis → no proposals → falls to refinement → returns noop
    client.add_response("optimization", "[]")
    client.add_response("iteratively", '{"action_type": "noop"}')

    loop = AgenticCompilationLoop(llm_client=client, env=env, budget=10)
    result = loop.run(target)
    # Should stop early (not run all 10 iterations)
    assert result.iterations_run < 10


def test_compilation_loop_with_real_eqsat() -> None:
    """The loop can drive real eqsat optimization."""
    env = CompilerEnv()
    target = load_profile("examples/target_profiles/cuda_a100.yaml")
    module = _make_module()
    env.reset(module, target, budget=20)

    client = MockLLMClient(strict=False)
    resp = '[{"action_type": "eqsat", "target": "all", "reason": "commute", "expected_improvement": 2.0}]'
    client.add_response("optimization", resp)

    loop = AgenticCompilationLoop(llm_client=client, env=env, budget=2)
    result = loop.run(target)

    # At least 1 iteration should have applied eqsat
    eqsat_iters = [r for r in result.history if r.action_type == "eqsat"]
    assert len(eqsat_iters) >= 1
