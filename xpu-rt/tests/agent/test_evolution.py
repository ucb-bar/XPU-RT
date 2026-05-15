"""Tests for evolutionary optimization."""

from __future__ import annotations

from compgen.agent.env import CompilerEnv
from compgen.agent.evolution import EvolutionaryOptimizer, EvolutionResult, Strategy
from compgen.llm.mock_client import MockLLMClient
from compgen.targets.schema import load_profile
from xdsl.dialects import arith, func
from xdsl.dialects.builtin import IndexType, ModuleOp
from xdsl.ir import Block, Region


def _make_module() -> ModuleOp:
    idx = IndexType()
    block = Block(arg_types=[idx, idx])
    a, b = block.args
    add = arith.AddiOp(a, b)
    block.add_op(add)
    mul = arith.MuliOp(add.result, b)
    block.add_op(mul)
    block.add_op(func.ReturnOp(mul.result))
    return ModuleOp([func.FuncOp("test", ([idx, idx], [idx]), Region([block]))])


def _make_mock_client() -> MockLLMClient:
    client = MockLLMClient(strict=False)
    # Initial population
    pop = (
        '[{"name": "s1", "actions": ["eqsat"], "description": "basic"}, '
        '{"name": "s2", "actions": ["eqsat", "eqsat"], "description": "double"}]'
    )
    client.add_response("optimization strategies", pop)
    # Mutation
    client.add_response("Refine", '[{"name": "s1_v2", "actions": ["eqsat"], "description": "refined"}]')
    return client


def test_evolution_runs() -> None:
    """Evolutionary optimizer runs without error."""
    env = CompilerEnv()
    target = load_profile("examples/target_profiles/cuda_a100.yaml")
    env.reset(_make_module(), target, budget=50)

    client = _make_mock_client()
    optimizer = EvolutionaryOptimizer(llm_client=client, env=env, population_size=2, generations=2)
    result = optimizer.evolve(target)

    assert isinstance(result, EvolutionResult)
    assert result.generations_run >= 1
    assert result.candidates_evaluated >= 2


def test_evolution_tracks_history() -> None:
    """Evolution records per-generation results."""
    env = CompilerEnv()
    target = load_profile("examples/target_profiles/cuda_a100.yaml")
    env.reset(_make_module(), target, budget=50)

    client = _make_mock_client()
    optimizer = EvolutionaryOptimizer(llm_client=client, env=env, population_size=2, generations=2)
    result = optimizer.evolve(target)

    assert len(result.history) == result.generations_run
    for gen in result.history:
        assert len(gen) >= 1


def test_evolution_selects_best() -> None:
    """Evolution returns the best strategy across generations."""
    env = CompilerEnv()
    target = load_profile("examples/target_profiles/cuda_a100.yaml")
    env.reset(_make_module(), target, budget=50)

    client = _make_mock_client()
    optimizer = EvolutionaryOptimizer(llm_client=client, env=env, population_size=2, generations=2)
    result = optimizer.evolve(target)

    assert result.best_strategy is not None
    assert result.best_strategy.name


def test_strategy_dataclass() -> None:
    s = Strategy(name="test", action_types=["eqsat", "tile"], description="test strategy")
    assert s.name == "test"
    assert len(s.action_types) == 2
