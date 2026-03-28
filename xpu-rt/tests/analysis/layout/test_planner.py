"""Tests for the LayoutPlanner."""

from __future__ import annotations

import pytest
from compgen.analysis.layout.planner import LayoutPlanner, LayoutPlan
from compgen.agent.analyzer import NetworkAnalysis, PatternCluster, DataFlowEdge
from compgen.targets.schema import TargetProfile, DeviceSpec, MemoryLevel, ComputeUnit


def _mock_target() -> TargetProfile:
    return TargetProfile(
        name="test_gpu",
        devices=[DeviceSpec(
            device_type="gpu", name="TestGPU", vendor="test",
            compute_units=[ComputeUnit(name="tensor_core", count=1, peak_tflops=100.0)],
            memory_hierarchy=[MemoryLevel(name="hbm", size_bytes=1024**3)],
            supported_ops=["matmul"], features=["tensor_core"],
            kernel_backends=["triton"],
        )],
    )


def _mock_target_no_tc() -> TargetProfile:
    """Target without tensor cores."""
    return TargetProfile(
        name="test_gpu_no_tc",
        devices=[DeviceSpec(
            device_type="gpu", name="BasicGPU", vendor="test",
            compute_units=[ComputeUnit(name="cuda_core", count=1, peak_tflops=10.0)],
            memory_hierarchy=[MemoryLevel(name="hbm", size_bytes=1024**3)],
            supported_ops=["matmul"], features=[],
            kernel_backends=["triton"],
        )],
    )


def _mock_analysis() -> NetworkAnalysis:
    cluster = PatternCluster(
        cluster_id="c0",
        pattern_type="linear_chain",
        node_names=("linear_0", "gelu_0", "linear_1"),
        total_flops=1000000,
        total_bytes=100000,
        arithmetic_intensity=10.0,
        estimated_latency_per_device={"TestGPU": 100.0},
        best_device="TestGPU",
        is_bottleneck=True,
        kernel_opportunity="fused_mlp",
        input_shapes={"linear_0": (1, 128)},
        output_shapes={"linear_1": (1, 64)},
    )
    return NetworkAnalysis(
        model_name="test_mlp",
        total_params=10000,
        total_flops=1000000,
        total_bytes=100000,
        clusters=[cluster],
        unclustered_ops=[],
        data_flow=[],
        bottleneck_clusters=["c0"],
        optimization_opportunities=["fuse MLP layers"],
    )


def _mock_analysis_matmul() -> NetworkAnalysis:
    """Analysis with matmul pattern type."""
    cluster = PatternCluster(
        cluster_id="m0",
        pattern_type="matmul_chain",
        node_names=("lhs", "rhs", "output"),
        total_flops=2000000,
        total_bytes=200000,
        arithmetic_intensity=10.0,
        estimated_latency_per_device={"TestGPU": 50.0},
        best_device="TestGPU",
        is_bottleneck=True,
        kernel_opportunity="tiled_matmul",
        input_shapes={"lhs": (64, 128)},
        output_shapes={"output": (64, 256)},
    )
    return NetworkAnalysis(
        model_name="test_matmul",
        total_params=0,
        total_flops=2000000,
        total_bytes=200000,
        clusters=[cluster],
        unclustered_ops=[],
        data_flow=[],
        bottleneck_clusters=["m0"],
        optimization_opportunities=[],
    )


class TestLayoutPlanner:
    def test_plan_returns_dict(self) -> None:
        planner = LayoutPlanner()
        plans = planner.plan(_mock_analysis(), _mock_target())
        assert isinstance(plans, dict)

    def test_plan_has_entries(self) -> None:
        planner = LayoutPlanner()
        plans = planner.plan(_mock_analysis(), _mock_target())
        assert len(plans) > 0

    def test_plan_values_are_layout_plans(self) -> None:
        planner = LayoutPlanner()
        plans = planner.plan(_mock_analysis(), _mock_target())
        for plan in plans.values():
            assert isinstance(plan, LayoutPlan)
            assert plan.preferred_output_layout in (
                "row_major", "tiled", "blocked", "col_major", "rowmajor",
            )

    def test_plan_keys_match_cluster_ids(self) -> None:
        planner = LayoutPlanner()
        analysis = _mock_analysis()
        plans = planner.plan(analysis, _mock_target())
        cluster_ids = {c.cluster_id for c in analysis.clusters}
        assert set(plans.keys()) == cluster_ids

    def test_matmul_cluster_gets_tiled(self) -> None:
        planner = LayoutPlanner()
        plans = planner.plan(_mock_analysis_matmul(), _mock_target())
        plan = plans["m0"]
        assert plan.preferred_output_layout == "tiled"
        assert plan.tile_encoding is not None

    def test_matmul_prepack_candidate(self) -> None:
        planner = LayoutPlanner()
        plans = planner.plan(_mock_analysis_matmul(), _mock_target())
        plan = plans["m0"]
        # RHS (second operand) should be marked for prepacking
        assert len(plan.prepack_candidates) > 0

    def test_no_tensor_core_target(self) -> None:
        planner = LayoutPlanner()
        plans = planner.plan(_mock_analysis_matmul(), _mock_target_no_tc())
        plan = plans["m0"]
        # Should still produce a plan, just with default tile encoding
        assert isinstance(plan, LayoutPlan)
        assert plan.tile_encoding is not None
