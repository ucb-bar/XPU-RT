"""Tests for agent memory and cost model calibration."""

from __future__ import annotations

import sys
from pathlib import Path

from compgen.agent.env import BenchmarkAction, CalibrateAction, CompilerEnv, DiscoverOpsAction
from compgen.agent.memory import AgentMemory, CostCalibration
from compgen.capture.torch_export import capture_model
from compgen.ir.payload.import_fx import fx_to_xdsl
from compgen.targets.schema import load_profile

EXAMPLES = Path(__file__).parent.parent.parent / "examples"


def _get_full():
    sys.path.insert(0, str(EXAMPLES / "models"))
    from simple_mlp import SimpleMLP, get_sample_inputs

    model = SimpleMLP()
    inputs = get_sample_inputs()
    ep = capture_model(model, inputs)
    module, _ = fx_to_xdsl(ep)
    return module, ep, model, inputs


def _get_target():
    return load_profile(EXAMPLES / "target_profiles" / "cuda_a100.yaml")


# ---- CostCalibration unit tests ----


def test_cost_calibration_defaults() -> None:
    cal = CostCalibration()
    assert cal.get_factor("gpu", "matmul") == 1.0


def test_cost_calibration_update() -> None:
    cal = CostCalibration()
    cal.update("gpu", "matmul", estimated_us=10.0, measured_us=100.0)
    # Factor should move toward 10.0 (100/10)
    factor = cal.get_factor("gpu", "matmul")
    assert factor > 1.0  # should be pulled toward 10.0


def test_cost_calibration_ema() -> None:
    """Multiple updates should use exponential moving average."""
    cal = CostCalibration()
    cal.update("gpu", "matmul", estimated_us=10.0, measured_us=100.0)
    f1 = cal.get_factor("gpu", "matmul")
    cal.update("gpu", "matmul", estimated_us=10.0, measured_us=100.0)
    f2 = cal.get_factor("gpu", "matmul")
    # Second update should push factor further toward 10.0
    assert f2 > f1


# ---- AgentMemory persistence ----


def test_agent_memory_save_load(tmp_path: Path) -> None:
    mem = AgentMemory()
    mem.cost_calibration.update("gpu", "matmul", 10.0, 100.0)
    mem.record_strategy("mlp", "a100", "linear_chain", ["generalize", "dce"], 0.5, 0.3, True)
    mem.record_pass_ordering("linear_chain", ["generalize", "canonicalize", "dce"])
    mem.session_count = 5

    path = tmp_path / "memory.json"
    mem.save(path)

    loaded = AgentMemory.load(path)
    assert loaded.session_count == 5
    assert loaded.cost_calibration.get_factor("gpu", "matmul") > 1.0
    assert len(loaded.strategy_history) == 1
    assert loaded.best_pass_ordering("linear_chain") == ["generalize", "canonicalize", "dce"]


# ---- CalibrateAction in env ----


def test_calibrate_after_benchmark() -> None:
    """CalibrateAction should update cost model from benchmark results."""
    module, ep, model, inputs = _get_full()
    env = CompilerEnv()
    env.reset(module, _get_target(), pytorch_model=model, sample_inputs=inputs, exported_program=ep, budget=10)

    # First benchmark
    r1 = env.step(BenchmarkAction(device="cpu", mode="eager", num_iterations=5))
    assert r1.info.action_applied

    # Then calibrate
    r2 = env.step(CalibrateAction())
    assert r2.info.action_applied, f"Calibrate failed: {r2.info.error}"
    assert any("CALIBRATE" in d for d in r2.info.diagnostics)
    assert any("correction" in d for d in r2.info.diagnostics)


def test_calibrate_without_benchmark_fails() -> None:
    """CalibrateAction without prior benchmark should fail."""
    module, ep, model, inputs = _get_full()
    env = CompilerEnv()
    env.reset(module, _get_target(), pytorch_model=model, sample_inputs=inputs, budget=5)

    result = env.step(CalibrateAction())
    assert not result.info.action_applied
    assert "benchmark" in result.info.error.lower()


# ---- DiscoverOpsAction ----


def test_discover_ops_simple_mlp() -> None:
    """SimpleMLP should have some unknown ops (gelu not in decomposition table by default)."""
    module, ep, model, inputs = _get_full()
    env = CompilerEnv()
    env.reset(module, _get_target(), exported_program=ep, budget=5)

    result = env.step(DiscoverOpsAction())
    assert result.info.action_applied
    assert any("DISCOVER" in d for d in result.info.diagnostics)
