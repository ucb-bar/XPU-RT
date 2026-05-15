"""Tests for runtime/heterogeneous_executor.py."""

from __future__ import annotations

from compgen.runtime.dispatch_strategy import (
    PipelineStrategy,
    WavefrontStrategy,
)
from compgen.runtime.heterogeneous_executor import (
    ExecutionResult,
    ExecutionStatus,
    HeterogeneousExecutor,
    OpResult,
)
from compgen.runtime.planner import ExecutionPlan, PlacementDecision
from compgen.runtime.topology import (
    RuntimeDevice,
    RuntimeLink,
    RuntimeNode,
    RuntimeTopology,
)
from compgen.targetgen.hardware_spec import DeploymentTopology


def _simple_topology() -> RuntimeTopology:
    """Single host node with one device."""
    return RuntimeTopology(
        deployment=DeploymentTopology.SINGLE_DEVICE,
        nodes=[
            RuntimeNode(
                name="host",
                devices=[RuntimeDevice(device_index=0, device_type="gpu")],
                role="host",
            ),
        ],
    )


def _soc_topology() -> RuntimeTopology:
    """Heterogeneous SoC: host + npu."""
    return RuntimeTopology(
        deployment=DeploymentTopology.MULTI_DOMAIN_SOC,
        nodes=[
            RuntimeNode(
                name="host",
                devices=[RuntimeDevice(device_index=0, device_type="cpu")],
                role="host",
            ),
            RuntimeNode(
                name="npu",
                devices=[RuntimeDevice(device_index=1, device_type="npu")],
                role="accelerator",
                runtime_env="zephyr_rtos",
            ),
        ],
        links=[
            RuntimeLink(
                src_node="host",
                dst_node="npu",
                transport="local",
                bandwidth_gbps=10.0,
            ),
        ],
    )


def _simple_plan() -> ExecutionPlan:
    """A simple 3-op execution plan."""
    return ExecutionPlan(
        placements=[
            PlacementDecision(op_name="op_0", device_index=0),
            PlacementDecision(op_name="op_1", device_index=0),
            PlacementDecision(op_name="op_2", device_index=0),
        ],
        execution_order=["op_0", "op_1", "op_2"],
        estimated_latency_us=30.0,
    )


class TestHeterogeneousExecutor:
    def test_simple_execution(self) -> None:
        topo = _simple_topology()
        executor = HeterogeneousExecutor(topology=topo)
        plan = _simple_plan()
        result = executor.execute(plan)

        assert result.status == ExecutionStatus.SUCCESS
        assert len(result.op_results) == 3
        assert result.waves_executed >= 1
        assert result.metadata["strategy"] == "bulk_sync"
        executor.shutdown()

    def test_soc_execution(self) -> None:
        topo = _soc_topology()
        executor = HeterogeneousExecutor(topology=topo)
        plan = ExecutionPlan(
            placements=[
                PlacementDecision(op_name="preprocess", device_index=0),
                PlacementDecision(op_name="matmul", device_index=1),
                PlacementDecision(op_name="postprocess", device_index=0),
            ],
            execution_order=["preprocess", "matmul", "postprocess"],
            estimated_latency_us=100.0,
        )
        result = executor.execute(plan)
        assert result.status == ExecutionStatus.SUCCESS
        assert result.metadata["topology"] == "multi_domain_soc"
        executor.shutdown()

    def test_with_pipeline_strategy(self) -> None:
        topo = _simple_topology()
        executor = HeterogeneousExecutor(
            topology=topo,
            strategy=PipelineStrategy(),
        )
        plan = _simple_plan()
        result = executor.execute(plan)
        assert result.status == ExecutionStatus.SUCCESS
        assert result.metadata["strategy"] == "pipeline"
        executor.shutdown()

    def test_with_wavefront_strategy(self) -> None:
        topo = _soc_topology()
        executor = HeterogeneousExecutor(
            topology=topo,
            strategy=WavefrontStrategy(),
        )
        plan = _simple_plan()
        result = executor.execute(plan)
        assert result.status == ExecutionStatus.SUCCESS
        assert result.metadata["strategy"] == "wavefront"
        executor.shutdown()

    def test_initialize_shutdown(self) -> None:
        topo = _soc_topology()
        executor = HeterogeneousExecutor(topology=topo)
        executor.initialize()
        assert executor._initialized is True
        executor.shutdown()
        assert executor._initialized is False

    def test_op_result_defaults(self) -> None:
        r = OpResult(op_name="test")
        assert r.status == ExecutionStatus.SUCCESS
        assert r.latency_us == 0.0

    def test_execution_result_defaults(self) -> None:
        r = ExecutionResult()
        assert r.status == ExecutionStatus.NOT_STARTED
        assert r.waves_executed == 0
