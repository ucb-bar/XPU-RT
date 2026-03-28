"""Network analyzer — pattern-level intelligence for the agent.

Analyzes a torch dynamo FX graph to produce structured network analysis:
pattern clusters, data flow, bottlenecks, and optimization opportunities.

Works directly on FX graphs (not xDSL IR) because FX has the clearest
computation structure with explicit data flow via node.users.
"""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
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
    dossier: "GraphAnalysisDossier | None" = None


@dataclass(frozen=True)
class RegionDossier:
    """Deterministic per-region analysis for the agent and compiler."""

    region_id: str
    kind: str
    node_names: tuple[str, ...]
    repeated_count: int
    flops: int
    bytes: int
    arithmetic_intensity: float
    dynamic_shapes: bool
    producers: tuple[str, ...]
    consumers: tuple[str, ...]
    parallelizable_with: tuple[str, ...]
    layout_candidates: tuple[str, ...]
    backend_viability: tuple[str, ...]
    best_device: str
    local_memory_fit: dict[str, bool] = field(default_factory=dict)


@dataclass(frozen=True)
class GraphAnalysisDossier:
    """Richer deterministic graph-analysis view for planning and synthesis."""

    model_name: str
    op_histogram: dict[str, int]
    repeated_patterns: dict[str, int]
    total_regions: int
    total_flops: int
    total_bytes: int
    critical_path: tuple[str, ...]
    independent_region_sets: tuple[tuple[str, ...], ...]
    dynamic_shape_regions: tuple[str, ...]
    unsupported_targets: tuple[str, ...]
    regions: tuple[RegionDossier, ...]


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
        total_params = self._count_parameter_placeholders(exported_program)
        dossier = self._build_dossier(
            fx_nodes=fx_nodes,
            clusters=clusters,
            data_flow=data_flow,
            target=target,
            model_name=model_name,
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
            dossier=dossier,
        )

    def _count_parameter_placeholders(self, exported_program: Any) -> int:
        graphs = getattr(exported_program, "graphs", ()) or ()
        if graphs:
            return sum(
                1
                for graph_module in graphs
                for node in graph_module.graph.nodes
                if node.op == "placeholder" and node.name.startswith("p_")
            )
        graph = getattr(exported_program, "graph", None)
        if graph is None:
            return 0
        return sum(
            1 for node in graph.nodes
            if node.op == "placeholder" and node.name.startswith("p_")
        )

    def _build_dossier(
        self,
        *,
        fx_nodes: list[FXNodeInfo],
        clusters: list[PatternCluster],
        data_flow: list[DataFlowEdge],
        target: TargetProfile,
        model_name: str,
    ) -> GraphAnalysisDossier:
        """Build the richer deterministic graph-analysis dossier."""

        node_by_name = {node.name: node for node in fx_nodes}
        op_histogram = Counter(node.target for node in fx_nodes)
        cluster_counter = Counter(cluster.pattern_type for cluster in clusters)
        node_to_region: dict[str, str] = {}
        region_bytes: dict[str, int] = {}
        region_flops: dict[str, int] = {}
        region_kind: dict[str, str] = {}
        region_best_device: dict[str, str] = {}
        region_node_names: dict[str, tuple[str, ...]] = {}

        for cluster in clusters:
            region_bytes[cluster.cluster_id] = cluster.total_bytes
            region_flops[cluster.cluster_id] = cluster.total_flops
            region_kind[cluster.cluster_id] = cluster.pattern_type
            region_best_device[cluster.cluster_id] = cluster.best_device
            region_node_names[cluster.cluster_id] = cluster.node_names
            for node_name in cluster.node_names:
                node_to_region[node_name] = cluster.cluster_id

        for node in fx_nodes:
            if node.name in node_to_region:
                continue
            node_to_region[node.name] = node.name
            region_bytes[node.name] = node.bytes_total
            region_flops[node.name] = node.flops
            region_kind[node.name] = node.target
            region_best_device[node.name] = target.devices[0].name if target.devices else ""
            region_node_names[node.name] = (node.name,)
            cluster_counter[node.target] += 1

        adjacency: dict[str, list[str]] = defaultdict(list)
        predecessors: dict[str, list[str]] = defaultdict(list)
        edge_set: set[tuple[str, str]] = set()
        for node in fx_nodes:
            src_region = node_to_region.get(node.name)
            if not src_region:
                continue
            for user_name in node.user_names:
                dst_region = node_to_region.get(user_name)
                if not dst_region or src_region == dst_region:
                    continue
                key = (src_region, dst_region)
                if key in edge_set:
                    continue
                edge_set.add(key)
                adjacency[src_region].append(dst_region)
                predecessors[dst_region].append(src_region)

        indegree: dict[str, int] = {
            region_id: len(set(predecessors.get(region_id, ())))
            for region_id in region_kind
        }

        topo_queue = deque(sorted(region_id for region_id, value in indegree.items() if value == 0))
        topo_order: list[str] = []
        depth: dict[str, int] = {}
        while topo_queue:
            region_id = topo_queue.popleft()
            topo_order.append(region_id)
            parent_depth = max((depth.get(parent, -1) for parent in predecessors.get(region_id, ())), default=-1)
            depth[region_id] = parent_depth + 1
            for child in adjacency.get(region_id, ()):
                indegree[child] -= 1
                if indegree[child] == 0:
                    topo_queue.append(child)

        if not topo_order:
            topo_order = [node.name for node in fx_nodes]
            depth = {node.name: 0 for node in fx_nodes}

        cluster_by_id = {cluster.cluster_id: cluster for cluster in clusters}
        best_latency = {}
        for region_id in region_kind:
            cluster = cluster_by_id.get(region_id)
            if cluster is not None and cluster.estimated_latency_per_device:
                best_latency[region_id] = min(cluster.estimated_latency_per_device.values())
            else:
                best_latency[region_id] = max(region_bytes.get(region_id, 0), 1) / 1e3
        longest_cost: dict[str, float] = {}
        longest_path: dict[str, tuple[str, ...]] = {}
        for region_id in topo_order:
            parent_ids = tuple(sorted(set(predecessors.get(region_id, ()))))
            if not parent_ids:
                longest_cost[region_id] = best_latency.get(region_id, 0.0)
                longest_path[region_id] = (region_id,)
                continue
            parent = max(parent_ids, key=lambda rid: longest_cost.get(rid, 0.0))
            longest_cost[region_id] = longest_cost.get(parent, 0.0) + best_latency.get(region_id, 0.0)
            longest_path[region_id] = longest_path.get(parent, ()) + (region_id,)
        critical_path = max(longest_path.values(), key=len, default=())

        same_depth: dict[int, list[str]] = defaultdict(list)
        for region_id, value in depth.items():
            same_depth[value].append(region_id)
        independent_region_sets = tuple(
            tuple(sorted(region_ids))
            for region_ids in same_depth.values()
            if len(region_ids) > 1
        )

        unsupported_targets: set[str] = set()
        from compgen.ir.payload.decompositions import DECOMPOSITION_TABLE

        for node in fx_nodes:
            if node.target not in DECOMPOSITION_TABLE:
                unsupported_targets.add(node.target)

        regions: list[RegionDossier] = []
        for region_id in topo_order:
            related_depth = depth.get(region_id, 0)
            parallelizable = tuple(
                rid for rid in same_depth.get(related_depth, [])
                if rid != region_id
            )
            node_names = region_node_names.get(region_id, ())
            dynamic_shapes = any(
                not all(isinstance(dim, int) and dim >= 0 for dim in (node_by_name[name].shape or ()))
                for name in node_names
                if name in node_by_name
            )
            regions.append(RegionDossier(
                region_id=region_id,
                kind=region_kind.get(region_id, region_id),
                node_names=node_names,
                repeated_count=cluster_counter[region_kind.get(region_id, region_id)],
                flops=region_flops.get(region_id, 0),
                bytes=region_bytes.get(region_id, 0),
                arithmetic_intensity=(
                    region_flops.get(region_id, 0) / region_bytes.get(region_id, 1)
                    if region_bytes.get(region_id, 0) > 0 else 0.0
                ),
                dynamic_shapes=dynamic_shapes,
                producers=tuple(sorted(set(predecessors.get(region_id, ())))),
                consumers=tuple(sorted(set(adjacency.get(region_id, ())))),
                parallelizable_with=parallelizable,
                layout_candidates=self._layout_candidates(region_kind.get(region_id, region_id)),
                backend_viability=self._backend_viability_for_kind(region_kind.get(region_id, region_id), target),
                best_device=region_best_device.get(region_id, ""),
                local_memory_fit={
                    device.name: self._local_memory_fit_for_bytes(region_bytes.get(region_id, 0), device)
                    for device in target.devices
                },
            ))

        dynamic_shape_regions = tuple(
            region.region_id for region in regions if region.dynamic_shapes
        )

        return GraphAnalysisDossier(
            model_name=model_name,
            op_histogram=dict(op_histogram),
            repeated_patterns=dict(cluster_counter),
            total_regions=len(regions),
            total_flops=sum(node.flops for node in fx_nodes),
            total_bytes=sum(node.bytes_total for node in fx_nodes),
            critical_path=tuple(critical_path),
            independent_region_sets=independent_region_sets,
            dynamic_shape_regions=dynamic_shape_regions,
            unsupported_targets=tuple(sorted(unsupported_targets)),
            regions=tuple(regions),
        )

    def _layout_candidates(self, pattern_type: str) -> tuple[str, ...]:
        if "attention" in pattern_type:
            return ("qkv_packed", "blocked_head_major")
        if "linear" in pattern_type or "mlp" in pattern_type:
            return ("rowmajor_epilogue", "blocked_64x64x32")
        return ("contiguous",)

    def _backend_viability_for_kind(self, pattern_type: str, target: TargetProfile) -> tuple[str, ...]:
        viable: list[str] = ["library", "fallback"]
        device_names = " ".join(device.name.lower() for device in target.devices)
        if any(token in device_names for token in ("cuda", "gpu")):
            viable.insert(0, "triton")
        if any(token in device_names for token in ("npu", "accel", "soc")):
            viable.insert(0, "accel_native")
        if pattern_type.startswith("linear") or "mlp" in pattern_type or "aten.addmm" in pattern_type:
            viable.append("exo")
        return tuple(dict.fromkeys(viable))

    def _backend_viability(self, cluster: PatternCluster, target: TargetProfile) -> tuple[str, ...]:
        return self._backend_viability_for_kind(cluster.pattern_type, target)

    def _local_memory_fit_for_bytes(self, required_bytes: int, device: Any) -> bool:
        available = max((getattr(level, "size_bytes", 0) for level in device.memory_hierarchy), default=0)
        return required_bytes <= available

    def _local_memory_fit(self, cluster: PatternCluster, device: Any) -> bool:
        return self._local_memory_fit_for_bytes(cluster.total_bytes, device)

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
    "GraphAnalysisDossier",
    "NetworkAnalysis",
    "NetworkAnalyzer",
    "PatternCluster",
    "RegionDossier",
]
