"""Tests for runtime/dispatch_strategy.py -- dispatch strategies."""

from __future__ import annotations

import pytest

from compgen.runtime.dispatch_strategy import (
    BulkSyncStrategy,
    DispatchOp,
    DispatchWave,
    PipelineStrategy,
    StrategyKind,
    StreamingStrategy,
    WavefrontStrategy,
    create_strategy,
    register_strategy,
)


# ---- Shared test data ----


def _linear_dag() -> tuple[list[str], dict[str, int], dict[str, list[str]], dict[str, float]]:
    """A → B → C, all on device 0."""
    ops = ["A", "B", "C"]
    placements = {"A": 0, "B": 0, "C": 0}
    deps = {"A": [], "B": ["A"], "C": ["B"]}
    latencies = {"A": 10.0, "B": 20.0, "C": 5.0}
    return ops, placements, deps, latencies


def _diamond_dag() -> tuple[list[str], dict[str, int], dict[str, list[str]], dict[str, float]]:
    """Diamond: A → (B, C) → D, B on device 0, C on device 1."""
    ops = ["A", "B", "C", "D"]
    placements = {"A": 0, "B": 0, "C": 1, "D": 0}
    deps = {"A": [], "B": ["A"], "C": ["A"], "D": ["B", "C"]}
    latencies = {"A": 5.0, "B": 15.0, "C": 10.0, "D": 8.0}
    return ops, placements, deps, latencies


def _with_copies() -> tuple[list[str], dict[str, int], dict[str, list[str]], dict[str, float]]:
    """A → copy_A_to_B → B, cross-device."""
    ops = ["A", "copy_A_to_B", "B"]
    placements = {"A": 0, "copy_A_to_B": 1, "B": 1}
    deps = {"A": [], "copy_A_to_B": ["A"], "B": ["copy_A_to_B"]}
    latencies = {"A": 10.0, "copy_A_to_B": 3.0, "B": 20.0}
    return ops, placements, deps, latencies


# ---- DispatchOp / DispatchWave ----


class TestDispatchTypes:
    def test_dispatch_op_defaults(self) -> None:
        op = DispatchOp(op_name="matmul_0", device_index=0)
        assert op.node_name == ""
        assert op.is_copy is False
        assert op.estimated_latency_us == 0.0

    def test_dispatch_wave(self) -> None:
        wave = DispatchWave(
            wave_id=0,
            ops=[DispatchOp(op_name="A", device_index=0)],
            sync_after=True,
        )
        assert len(wave.ops) == 1
        assert wave.sync_after is True


# ---- BulkSyncStrategy ----


class TestBulkSyncStrategy:
    def test_kind(self) -> None:
        s = BulkSyncStrategy()
        assert s.kind == StrategyKind.BULK_SYNC
        assert s.name == "bulk_sync"

    def test_linear(self) -> None:
        ops, placements, deps, latencies = _linear_dag()
        waves = BulkSyncStrategy().plan_waves(ops, placements, deps, latencies)
        # A(level 0), B(level 1), C(level 2) → 3 waves
        assert len(waves) == 3
        assert waves[0].ops[0].op_name == "A"
        assert waves[1].ops[0].op_name == "B"
        assert waves[2].ops[0].op_name == "C"
        # All sync after
        assert all(w.sync_after for w in waves)

    def test_diamond(self) -> None:
        ops, placements, deps, latencies = _diamond_dag()
        waves = BulkSyncStrategy().plan_waves(ops, placements, deps, latencies)
        # A(level 0), B+C(level 1), D(level 2)
        assert len(waves) == 3
        level1_ops = {op.op_name for op in waves[1].ops}
        assert level1_ops == {"B", "C"}

    def test_node_names(self) -> None:
        ops, placements, deps, latencies = _diamond_dag()
        node_map = {0: "host", 1: "npu"}
        waves = BulkSyncStrategy().plan_waves(
            ops, placements, deps, latencies, node_for_device=node_map,
        )
        # C is on device 1 → node "npu"
        c_op = [op for w in waves for op in w.ops if op.op_name == "C"][0]
        assert c_op.node_name == "npu"


# ---- PipelineStrategy ----


class TestPipelineStrategy:
    def test_kind(self) -> None:
        s = PipelineStrategy()
        assert s.kind == StrategyKind.PIPELINE

    def test_with_copies(self) -> None:
        ops, placements, deps, latencies = _with_copies()
        waves = PipelineStrategy().plan_waves(ops, placements, deps, latencies)
        # Should separate copy from compute
        assert len(waves) >= 2
        # Find the copy wave
        copy_waves = [w for w in waves if any(op.is_copy for op in w.ops)]
        assert len(copy_waves) >= 1
        # Copy wave should not sync (async overlap)
        for cw in copy_waves:
            assert cw.sync_after is False

    def test_summary(self) -> None:
        s = PipelineStrategy(num_stages=4)
        summary = s.summary()
        assert summary["kind"] == "pipeline"
        assert summary["num_stages"] == 4


# ---- WavefrontStrategy ----


class TestWavefrontStrategy:
    def test_kind(self) -> None:
        s = WavefrontStrategy()
        assert s.kind == StrategyKind.WAVEFRONT

    def test_diamond(self) -> None:
        ops, placements, deps, latencies = _diamond_dag()
        waves = WavefrontStrategy().plan_waves(ops, placements, deps, latencies)
        # Wave 0: A, Wave 1: B+C, Wave 2: D
        assert len(waves) == 3
        assert waves[0].ops[0].op_name == "A"
        wave1_names = {op.op_name for op in waves[1].ops}
        assert wave1_names == {"B", "C"}
        # Wave 1 crosses devices → sync_after
        assert waves[1].sync_after is True

    def test_single_device_no_sync(self) -> None:
        ops, placements, deps, latencies = _linear_dag()
        waves = WavefrontStrategy().plan_waves(ops, placements, deps, latencies)
        # All on same device → no cross-device sync needed
        for w in waves:
            assert w.sync_after is False

    def test_wavefront_width_metadata(self) -> None:
        ops, placements, deps, latencies = _diamond_dag()
        waves = WavefrontStrategy().plan_waves(ops, placements, deps, latencies)
        assert waves[1].metadata["wavefront_width"] == 2


# ---- StreamingStrategy ----


class TestStreamingStrategy:
    def test_kind(self) -> None:
        s = StreamingStrategy()
        assert s.kind == StrategyKind.STREAMING

    def test_multi_device(self) -> None:
        ops = ["A0", "A1", "B0", "B1"]
        placements = {"A0": 0, "A1": 0, "B0": 1, "B1": 1}
        deps: dict[str, list[str]] = {"A0": [], "A1": ["A0"], "B0": [], "B1": ["B0"]}
        latencies = {"A0": 5.0, "A1": 5.0, "B0": 5.0, "B1": 5.0}
        waves = StreamingStrategy().plan_waves(ops, placements, deps, latencies)
        # Each wave should have one op per device
        for w in waves[:-1]:
            assert w.sync_after is False  # no sync in streaming
        # Last wave syncs
        assert waves[-1].sync_after is True

    def test_double_buffer_metadata(self) -> None:
        s = StreamingStrategy(double_buffer=True)
        ops = ["A"]
        waves = s.plan_waves(ops, {"A": 0}, {"A": []}, {"A": 1.0})
        assert waves[0].metadata["double_buffer"] is True

    def test_summary(self) -> None:
        s = StreamingStrategy(double_buffer=False)
        assert s.summary()["double_buffer"] is False


# ---- Factory ----


class TestStrategyFactory:
    def test_create_all(self) -> None:
        for name in ("bulk_sync", "pipeline", "wavefront", "streaming"):
            s = create_strategy(name)
            assert s.kind.value == name

    def test_create_unknown(self) -> None:
        with pytest.raises(ValueError, match="Unknown dispatch strategy"):
            create_strategy("magic")

    def test_create_with_kwargs(self) -> None:
        s = create_strategy("pipeline", num_stages=8)
        assert isinstance(s, PipelineStrategy)

    def test_register_custom(self) -> None:
        class MyStrategy(BulkSyncStrategy):
            @property
            def kind(self) -> StrategyKind:
                return StrategyKind.BULK_SYNC

            @property
            def name(self) -> str:
                return "my_custom"

        register_strategy("my_custom", MyStrategy)
        s = create_strategy("my_custom")
        assert s.name == "my_custom"
