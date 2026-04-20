"""Tests for PrepackPlanner."""

from __future__ import annotations

from compgen.agent.analyzer import DataFlowEdge, NetworkAnalysis, PatternCluster
from compgen.analysis.layout.prepack import PrepackCandidate, PrepackPlanner


def _mock_analysis_with_weights() -> NetworkAnalysis:
    """Analysis containing constant weight operands."""
    cluster = PatternCluster(
        cluster_id="c0",
        pattern_type="linear_chain",
        node_names=("p_weight_linear", "bias_linear", "activation_0"),
        total_flops=1000000,
        total_bytes=300000,
        arithmetic_intensity=3.33,
        estimated_latency_per_device={"TestGPU": 100.0},
        best_device="TestGPU",
        is_bottleneck=True,
        kernel_opportunity="fused_linear",
        input_shapes={"p_weight_linear": (256, 128)},
        output_shapes={"activation_0": (1, 256)},
    )
    return NetworkAnalysis(
        model_name="test_model",
        total_params=1000,
        total_flops=1000000,
        total_bytes=300000,
        clusters=[cluster],
        unclustered_ops=[],
        data_flow=[],
        bottleneck_clusters=["c0"],
        optimization_opportunities=[],
    )


def _mock_analysis_reuse() -> NetworkAnalysis:
    """Analysis with cross-region operand reuse."""
    c0 = PatternCluster(
        cluster_id="c0",
        pattern_type="linear_chain",
        node_names=("shared_embed", "linear_0"),
        total_flops=500000,
        total_bytes=100000,
        arithmetic_intensity=5.0,
        estimated_latency_per_device={"TestGPU": 50.0},
        best_device="TestGPU",
        is_bottleneck=False,
        kernel_opportunity="fused_linear",
        input_shapes={"shared_embed": (1, 128)},
        output_shapes={"linear_0": (1, 64)},
    )
    c1 = PatternCluster(
        cluster_id="c1",
        pattern_type="linear_chain",
        node_names=("shared_embed", "linear_1"),
        total_flops=500000,
        total_bytes=100000,
        arithmetic_intensity=5.0,
        estimated_latency_per_device={"TestGPU": 50.0},
        best_device="TestGPU",
        is_bottleneck=False,
        kernel_opportunity="fused_linear",
        input_shapes={"shared_embed": (1, 128)},
        output_shapes={"linear_1": (1, 64)},
    )
    return NetworkAnalysis(
        model_name="test_reuse",
        total_params=500,
        total_flops=1000000,
        total_bytes=200000,
        clusters=[c0, c1],
        unclustered_ops=[],
        data_flow=[DataFlowEdge(src="c0", dst="c1", tensor_bytes=50000)],
        bottleneck_clusters=[],
        optimization_opportunities=[],
    )


def _mock_analysis_no_prepack() -> NetworkAnalysis:
    """Analysis with no constant operands and no reuse."""
    cluster = PatternCluster(
        cluster_id="c0",
        pattern_type="elementwise",
        node_names=("relu_0", "add_0"),
        total_flops=10000,
        total_bytes=10000,
        arithmetic_intensity=1.0,
        estimated_latency_per_device={"TestGPU": 10.0},
        best_device="TestGPU",
        is_bottleneck=False,
        kernel_opportunity="",
        input_shapes={"relu_0": (1, 64)},
        output_shapes={"add_0": (1, 64)},
    )
    return NetworkAnalysis(
        model_name="test_no_prepack",
        total_params=0,
        total_flops=10000,
        total_bytes=10000,
        clusters=[cluster],
        unclustered_ops=[],
        data_flow=[],
        bottleneck_clusters=[],
        optimization_opportunities=[],
    )


class TestPrepackPlanner:
    def test_identifies_constant_weights(self) -> None:
        planner = PrepackPlanner()
        candidates = planner.identify_prepack_opportunities(_mock_analysis_with_weights())
        assert len(candidates) > 0
        # Should find weight and bias operands
        names = {c.operand_name for c in candidates}
        assert "p_weight_linear" in names or "bias_linear" in names

    def test_candidates_are_prepack_candidate_type(self) -> None:
        planner = PrepackPlanner()
        candidates = planner.identify_prepack_opportunities(_mock_analysis_with_weights())
        for c in candidates:
            assert isinstance(c, PrepackCandidate)

    def test_sorted_by_benefit(self) -> None:
        planner = PrepackPlanner()
        candidates = planner.identify_prepack_opportunities(_mock_analysis_with_weights())
        if len(candidates) >= 2:
            for i in range(len(candidates) - 1):
                assert candidates[i].estimated_benefit_us >= candidates[i + 1].estimated_benefit_us

    def test_cross_region_reuse_detected(self) -> None:
        planner = PrepackPlanner()
        candidates = planner.identify_prepack_opportunities(_mock_analysis_reuse())
        # shared_embed appears in both clusters, should be detected
        reuse_names = {c.operand_name for c in candidates if c.reuse_count > 1}
        assert "shared_embed" in reuse_names

    def test_no_candidates_for_elementwise_only(self) -> None:
        planner = PrepackPlanner()
        candidates = planner.identify_prepack_opportunities(_mock_analysis_no_prepack())
        assert len(candidates) == 0

    def test_constant_is_flagged(self) -> None:
        planner = PrepackPlanner()
        candidates = planner.identify_prepack_opportunities(_mock_analysis_with_weights())
        const_candidates = [c for c in candidates if c.is_constant]
        assert len(const_candidates) > 0
