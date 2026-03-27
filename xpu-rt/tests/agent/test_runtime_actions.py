"""Tests for new runtime action types in agent/env.py."""

from __future__ import annotations

from xdsl.dialects.builtin import ModuleOp

from compgen.agent.env import (
    CompilerEnv,
    ConfigureDispatchAction,
    ConfigureProfilingAction,
    GenerateRuntimeHooksAction,
)
from compgen.targets.schema import DeviceSpec, TargetProfile


def _make_env() -> CompilerEnv:
    """Create a CompilerEnv with a minimal module and target."""
    module = ModuleOp([])
    target = TargetProfile(
        name="test-target",
        devices=[DeviceSpec(device_type="gpu", name="TestGPU")],
    )
    env = CompilerEnv()
    env.reset(module, target, budget=100)
    return env


class TestConfigureProfilingAction:
    def test_basic(self) -> None:
        env = _make_env()
        action = ConfigureProfilingAction(
            instrumentation_level="op_level",
            counters=["cycles", "instructions"],
            analysis_focus="latency",
        )
        result = env.step(action)
        assert result.info.action_applied is True
        assert "profiling" in env.runtime_artifacts
        assert env.runtime_artifacts["profiling"]["level"] == "op_level"

    def test_tile_level(self) -> None:
        env = _make_env()
        action = ConfigureProfilingAction(
            instrumentation_level="tile_level",
            counters=["cycles", "cache_misses", "dram_reads"],
        )
        result = env.step(action)
        assert result.info.action_applied is True
        assert len(env.runtime_artifacts["profiling"]["counters"]) == 3

    def test_with_custom_hooks(self) -> None:
        env = _make_env()
        action = ConfigureProfilingAction(
            instrumentation_level="full",
            custom_hooks={
                "pre_dispatch": "my_trace_begin();",
                "post_dispatch": "my_trace_end();",
            },
        )
        result = env.step(action)
        assert result.info.action_applied is True
        assert "pre_dispatch" in env.generated_hooks
        assert env.generated_hooks["pre_dispatch"] == "my_trace_begin();"


class TestConfigureDispatchAction:
    def test_basic(self) -> None:
        env = _make_env()
        action = ConfigureDispatchAction(strategy="pipeline")
        result = env.step(action)
        assert result.info.action_applied is True
        assert "dispatch" in env.runtime_artifacts
        assert env.runtime_artifacts["dispatch"]["strategy"] == "pipeline"

    def test_with_transport_overrides(self) -> None:
        env = _make_env()
        action = ConfigureDispatchAction(
            strategy="wavefront",
            transport_overrides={"host->npu": "zephyr_ipc"},
            thread_config={"dispatch": 3},
            double_buffer=True,
        )
        result = env.step(action)
        assert result.info.action_applied is True
        assert env.runtime_artifacts["dispatch"]["double_buffer"] is True

    def test_invalid_strategy(self) -> None:
        env = _make_env()
        action = ConfigureDispatchAction(strategy="invalid_strategy_name")
        result = env.step(action)
        assert result.info.action_applied is False
        assert "Unknown" in result.info.error


class TestGenerateRuntimeHooksAction:
    def test_basic(self) -> None:
        env = _make_env()
        action = GenerateRuntimeHooksAction(
            hook_type="profiling",
            hook_code={
                "pre_dispatch": 'CG_TRACE_BEGIN("dispatch", name);',
                "post_dispatch": "CG_TRACE_END();",
            },
        )
        result = env.step(action)
        assert result.info.action_applied is True
        assert "hooks" in env.runtime_artifacts
        assert "pre_dispatch" in env.runtime_artifacts["hooks"]

    def test_empty_hook_code_rejected(self) -> None:
        env = _make_env()
        action = GenerateRuntimeHooksAction(hook_code={})
        result = env.step(action)
        assert result.info.action_applied is False

    def test_empty_code_string_rejected(self) -> None:
        env = _make_env()
        action = GenerateRuntimeHooksAction(
            hook_code={"pre_dispatch": "  "},
        )
        result = env.step(action)
        assert result.info.action_applied is False

    def test_hooks_accumulate(self) -> None:
        env = _make_env()
        env.step(GenerateRuntimeHooksAction(
            hook_code={"pre_dispatch": "hook1();"},
        ))
        env.step(GenerateRuntimeHooksAction(
            hook_code={"post_dispatch": "hook2();"},
        ))
        assert len(env.generated_hooks) == 2
        assert "pre_dispatch" in env.generated_hooks
        assert "post_dispatch" in env.generated_hooks
