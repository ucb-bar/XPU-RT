"""Tests for TransposeProfitabilityAnalyzer."""

from __future__ import annotations

import pytest
from compgen.analysis.layout.transpose import (
    TransposeClassification,
    TransposeProfitabilityAnalyzer,
)
from compgen.agent.analyzer import NetworkAnalysis, PatternCluster, DataFlowEdge
from compgen.ir.payload.contracts import KernelContract


def _mock_transpose_analysis(
    consumer_type: str = "elementwise",
    extra_clusters: list[PatternCluster] | None = None,
    extra_edges: list[DataFlowEdge] | None = None,
) -> NetworkAnalysis:
    """Build an analysis with a transpose cluster and configurable consumers."""
    transpose_cluster = PatternCluster(
        cluster_id="t0",
        pattern_type="transpose_op",
        node_names=("transpose_0",),
        total_flops=0,
        total_bytes=50000,
        arithmetic_intensity=0.0,
        estimated_latency_per_device={"TestGPU": 5.0},
        best_device="TestGPU",
        is_bottleneck=False,
        kernel_opportunity="",
        input_shapes={"transpose_0": (64, 128)},
        output_shapes={"transpose_0": (128, 64)},
    )
    consumer_cluster = PatternCluster(
        cluster_id="c0",
        pattern_type=consumer_type,
        node_names=("consumer_0",),
        total_flops=100000,
        total_bytes=50000,
        arithmetic_intensity=2.0,
        estimated_latency_per_device={"TestGPU": 20.0},
        best_device="TestGPU",
        is_bottleneck=False,
        kernel_opportunity="",
        input_shapes={"consumer_0": (128, 64)},
        output_shapes={"consumer_0": (128, 64)},
    )
    clusters = [transpose_cluster, consumer_cluster]
    edges = [DataFlowEdge(src="t0", dst="c0", tensor_bytes=50000)]
    if extra_clusters:
        clusters.extend(extra_clusters)
    if extra_edges:
        edges.extend(extra_edges)
    return NetworkAnalysis(
        model_name="test_transpose",
        total_params=0,
        total_flops=100000,
        total_bytes=100000,
        clusters=clusters,
        unclustered_ops=[],
        data_flow=edges,
        bottleneck_clusters=[],
        optimization_opportunities=[],
    )


class TestTransposeProfitabilityAnalyzer:
    def test_classify_returns_dict(self) -> None:
        analyzer = TransposeProfitabilityAnalyzer()
        analysis = _mock_transpose_analysis()
        result = analyzer.classify_transposes(analysis, [])
        assert isinstance(result, dict)

    def test_transpose_nodes_classified(self) -> None:
        analyzer = TransposeProfitabilityAnalyzer()
        analysis = _mock_transpose_analysis()
        result = analyzer.classify_transposes(analysis, [])
        assert "transpose_0" in result
        assert isinstance(result["transpose_0"], TransposeClassification)

    def test_propagatable_through_elementwise(self) -> None:
        analyzer = TransposeProfitabilityAnalyzer()
        analysis = _mock_transpose_analysis(consumer_type="elementwise_relu")
        result = analyzer.classify_transposes(analysis, [])
        assert result["transpose_0"] == TransposeClassification.PROPAGATABLE

    def test_boundary_for_unknown_consumer(self) -> None:
        analyzer = TransposeProfitabilityAnalyzer()
        analysis = _mock_transpose_analysis(consumer_type="custom_unknown_op")
        result = analyzer.classify_transposes(analysis, [])
        assert result["transpose_0"] == TransposeClassification.BOUNDARY

    def test_absorbable_with_contract(self) -> None:
        analyzer = TransposeProfitabilityAnalyzer()
        analysis = _mock_transpose_analysis(consumer_type="matmul_fused")
        contract = KernelContract(
            op_name="matmul_fused",
            metadata={"can_absorb_transpose": True},
        )
        result = analyzer.classify_transposes(analysis, [contract])
        assert result["transpose_0"] == TransposeClassification.ABSORBABLE

    def test_eliminable_double_transpose(self) -> None:
        second_transpose = PatternCluster(
            cluster_id="t1",
            pattern_type="transpose_op",
            node_names=("transpose_1",),
            total_flops=0,
            total_bytes=50000,
            arithmetic_intensity=0.0,
            estimated_latency_per_device={"TestGPU": 5.0},
            best_device="TestGPU",
            is_bottleneck=False,
            kernel_opportunity="",
            input_shapes={"transpose_1": (128, 64)},
            output_shapes={"transpose_1": (64, 128)},
        )
        edge = DataFlowEdge(src="t0", dst="t1", tensor_bytes=50000)
        analysis = _mock_transpose_analysis(
            extra_clusters=[second_transpose],
            extra_edges=[edge],
        )
        analyzer = TransposeProfitabilityAnalyzer()
        result = analyzer.classify_transposes(analysis, [])
        assert result["transpose_0"] == TransposeClassification.ELIMINABLE

    def test_no_transposes_gives_empty(self) -> None:
        cluster = PatternCluster(
            cluster_id="c0",
            pattern_type="linear_chain",
            node_names=("linear_0",),
            total_flops=100000,
            total_bytes=50000,
            arithmetic_intensity=2.0,
            estimated_latency_per_device={"TestGPU": 20.0},
            best_device="TestGPU",
            is_bottleneck=False,
            kernel_opportunity="fused_linear",
            input_shapes={"linear_0": (1, 128)},
            output_shapes={"linear_0": (1, 64)},
        )
        analysis = NetworkAnalysis(
            model_name="no_transpose",
            total_params=0,
            total_flops=100000,
            total_bytes=50000,
            clusters=[cluster],
            unclustered_ops=[],
            data_flow=[],
            bottleneck_clusters=[],
            optimization_opportunities=[],
        )
        analyzer = TransposeProfitabilityAnalyzer()
        result = analyzer.classify_transposes(analysis, [])
        assert len(result) == 0
