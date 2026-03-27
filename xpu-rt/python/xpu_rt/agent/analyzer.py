"""Network analyzer — pattern-level intelligence for the agent.

Analyzes a torch dynamo FX graph to produce structured network analysis:
pattern clusters, data flow, bottlenecks, and optimization opportunities.

Works directly on FX graphs (not xDSL IR) because FX has the clearest
computation structure with explicit data flow via node.users.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from compgen.agent.patterns import FXNodeInfo, extract_fx_nodes, match_patterns
from compgen.targets.schema import TargetProfile


@dataclass(frozen=True)
class PatternCluster:
    """A group of ops forming a recognized computation pattern.

    This is what the agent reasons about — not individual ops, but
    meaningful computation units with kernel opportunities.
    """

    cluster_id: str
    pattern_type: str                # "linear_chain", "gqa_attention", etc.
    node_names: tuple[str, ...]      # FX node names in this cluster
    total_flops: int
    total_bytes: int
    arithmetic_intensity: float      # flops / bytes
    estimated_latency_per_device: dict[str, float]  # device_name → us
    best_device: str
    is_bottleneck: bool
    kernel_opportunity: str          # what kernel could replace this
    input_shapes: dict[str, tuple[int, ...]]   # node_name → shape for cluster inputs
    output_shapes: dict[str, tuple[int, ...]]  # node_name → shape for cluster outputs


@dataclass(frozen=True)
class DataFlowEdge:
    """Data flow between clusters (or between a cluster and input/output)."""

    src: str                         # cluster_id or "input"
    dst: str                         # cluster_id or "output"
    tensor_bytes: int


@dataclass(frozen=True)
class NetworkAnalysis:
    """Complete analysis of a model's computation graph.

    This is what the agent receives when it runs AnalyzeAction.
    """

    model_name: str
    total_params: int
    total_flops: int
    total_bytes: int
    clusters: list[PatternCluster]
    unclustered_ops: list[str]       # FX nodes not in any cluster
    data_flow: list[DataFlowEdge]
    bottleneck_clusters: list[str]   # cluster_ids sorted by latency (worst first)
    optimization_opportunities: list[str]  # natural-language descriptions


class NetworkAnalyzer:
    """Analyzes FX graphs for the agent."""

    def analyze(
        self,
        exported_program: Any,
        target: TargetProfile,
        model_name: str = "unknown",
    ) -> NetworkAnalysis:
        """Analyze a torch.export ExportedProgram.

        Steps:
            1. Extract FX node info
            2. Match patterns
            3. Build clusters with cost estimates
            4. Build data flow graph
            5. Identify bottlenecks
            6. Generate optimization opportunities
        """
        # Step 1: Extract FX nodes
        fx_nodes = extract_fx_nodes(exported_program)
        node_by_name: dict[str, FXNodeInfo] = {n.name: n for n in fx_nodes}

        # Step 2: Match patterns
        matched = match_patterns(fx_nodes)

        # Step 3: Build clusters
        clusters: list[PatternCluster] = []
        clustered_nodes: set[str] = set()

        for match in matched:
            cluster_nodes = [node_by_name[n] for n in match.node_names if n in node_by_name]
            if not cluster_nodes:
                continue

            total_flops = sum(n.flops for n in cluster_nodes)
            total_bytes = sum(n.bytes_total for n in cluster_nodes)
            ai = total_flops / total_bytes if total_bytes > 0 else 0.0

            # Estimate per-device latency
            latency_per_device: dict[str, float] = {}
            best_device = ""
            best_latency = float("inf")

            for dev in target.devices:
                peak_flops = 0.0
                peak_bw = 0.0
                for cu in dev.compute_units:
                    if cu.peak_tflops:
                        peak_flops = max(peak_flops, cu.peak_tflops * 1e12)
                for ml in dev.memory_hierarchy:
                    if ml.bandwidth_gbps:
                        peak_bw = max(peak_bw, ml.bandwidth_gbps * 1e9)

                compute_time = total_flops / peak_flops if peak_flops > 0 else float("inf")
                memory_time = total_bytes / peak_bw if peak_bw > 0 else float("inf")
                latency_us = max(compute_time, memory_time) * 1e6

                latency_per_device[dev.name] = latency_us
                if latency_us < best_latency:
                    best_latency = latency_us
                    best_device = dev.name

            # Input/output shapes
            input_shapes: dict[str, tuple[int, ...]] = {}
            output_shapes: dict[str, tuple[int, ...]] = {}
            first_node = cluster_nodes[0]
            last_node = cluster_nodes[-1]
            if first_node.shape:
                input_shapes[first_node.name] = first_node.shape
            if last_node.shape:
                output_shapes[last_node.name] = last_node.shape

            clusters.append(PatternCluster(
                cluster_id=match.cluster_id,
                pattern_type=match.pattern_name,
                node_names=match.node_names,
                total_flops=total_flops,
                total_bytes=total_bytes,
                arithmetic_intensity=ai,
                estimated_latency_per_device=latency_per_device,
                best_device=best_device,
                is_bottleneck=False,  # set below
                kernel_opportunity=match.kernel_opportunity,
                input_shapes=input_shapes,
                output_shapes=output_shapes,
            ))

            clustered_nodes.update(match.node_names)

        # Unclustered ops
        unclustered = [n.name for n in fx_nodes if n.name not in clustered_nodes]

        # Step 4: Data flow between clusters
        data_flow = self._build_data_flow(clusters, node_by_name, fx_nodes)

        # Step 5: Bottleneck identification (sort by total estimated latency)
        if clusters:
            sorted_clusters = sorted(
                clusters,
                key=lambda c: min(c.estimated_latency_per_device.values()) if c.estimated_latency_per_device else 0,
                reverse=True,
            )
            bottleneck_ids = [c.cluster_id for c in sorted_clusters]

            # Mark top clusters as bottlenecks (top 30%)
            n_bottlenecks = max(1, len(clusters) // 3)
            bottleneck_set = set(bottleneck_ids[:n_bottlenecks])

            clusters = [
                PatternCluster(
                    cluster_id=c.cluster_id, pattern_type=c.pattern_type,
                    node_names=c.node_names, total_flops=c.total_flops,
                    total_bytes=c.total_bytes, arithmetic_intensity=c.arithmetic_intensity,
                    estimated_latency_per_device=c.estimated_latency_per_device,
                    best_device=c.best_device,
                    is_bottleneck=c.cluster_id in bottleneck_set,
                    kernel_opportunity=c.kernel_opportunity,
                    input_shapes=c.input_shapes, output_shapes=c.output_shapes,
                )
                for c in clusters
            ]
        else:
            bottleneck_ids = []

        # Step 6: Optimization opportunities
        opportunities = self._generate_opportunities(clusters, unclustered, target)

        # Total stats
        total_flops = sum(n.flops for n in fx_nodes)
        total_bytes = sum(n.bytes_total for n in fx_nodes)
        total_params = sum(
            1 for node in exported_program.graph.nodes
            if node.op == "placeholder" and node.name.startswith("p_")
        )

        return NetworkAnalysis(
            model_name=model_name,
            total_params=total_params,
            total_flops=total_flops,
            total_bytes=total_bytes,
            clusters=clusters,
            unclustered_ops=unclustered,
            data_flow=data_flow,
            bottleneck_clusters=bottleneck_ids,
            optimization_opportunities=opportunities,
        )

    def _build_data_flow(
        self,
        clusters: list[PatternCluster],
        node_by_name: dict[str, FXNodeInfo],
        all_nodes: list[FXNodeInfo],
    ) -> list[DataFlowEdge]:
        """Build data flow edges between clusters."""
        # Map node_name → cluster_id
        node_to_cluster: dict[str, str] = {}
        for c in clusters:
            for n in c.node_names:
                node_to_cluster[n] = c.cluster_id

        edges: list[DataFlowEdge] = []
        seen_edges: set[tuple[str, str]] = set()

        for node in all_nodes:
            src_cluster = node_to_cluster.get(node.name, "unclustered")
            for user_name in node.user_names:
                dst_cluster = node_to_cluster.get(user_name, "unclustered")
                if src_cluster != dst_cluster:
                    edge_key = (src_cluster, dst_cluster)
                    if edge_key not in seen_edges:
                        bytes_transferred = node.bytes_total
                        edges.append(DataFlowEdge(
                            src=src_cluster, dst=dst_cluster,
                            tensor_bytes=bytes_transferred,
                        ))
                        seen_edges.add(edge_key)

        return edges

    def _generate_opportunities(
        self,
        clusters: list[PatternCluster],
        unclustered: list[str],
        target: TargetProfile,
    ) -> list[str]:
        """Generate natural-language optimization opportunity descriptions."""
        opps: list[str] = []

        for c in clusters:
            if c.kernel_opportunity:
                opps.append(
                    f"Cluster '{c.cluster_id}' ({c.pattern_type}): "
                    f"kernel opportunity '{c.kernel_opportunity}' — "
                    f"{c.total_flops:,} FLOPs, "
                    f"{'compute' if c.arithmetic_intensity > 10 else 'memory'}-bound"
                )

            if c.is_bottleneck:
                opps.append(f"BOTTLENECK: '{c.cluster_id}' is a top latency contributor")

        if unclustered:
            opps.append(f"{len(unclustered)} ops not in any recognized pattern — may need custom handling")

        if len(target.devices) > 1:
            opps.append(
                f"Multi-device target ({len(target.devices)} devices) — "
                f"heterogeneous placement opportunity"
            )

        return opps


__all__ = [
    "DataFlowEdge",
    "NetworkAnalysis",
    "NetworkAnalyzer",
    "PatternCluster",
]
