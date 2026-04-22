"""LLM-facing graph digest + focused-chunk views.

This module produces the two data structures the LLM reads instead of
the raw IR:

* :class:`GraphDigest` — a shape-free overall summary (pattern histogram,
  dim spectrum, dtype spectrum, FLOP distribution, bottlenecks, fusion
  opportunity count, region index).
* :class:`ChunkView` — a focused subgraph view carrying both
  oracle-enumerated :class:`DecisionKnobs` (safer, bounded candidates
  from ``fusion_oracle`` / ``granularity_oracle`` / ``tile_oracle``) and
  an open-ended :class:`DoFDescription` (free-form design space so the
  LLM can propose novel strategies that the infrastructure verifies).

Heavy reuse:

* :class:`compgen.agent.analyzer.NetworkAnalysis` + :class:`GraphAnalysisDossier`
  — pattern clusters, region dossiers, critical path, bottlenecks.
* :func:`compgen.kernels.envelope_bridge.envelope_from_target_profile`
  — deterministic :class:`HardwareEnvelope`.
* :mod:`compgen.kernels.tile_oracle`, :mod:`granularity_oracle`,
  :mod:`fusion_oracle` — decision-knob enumeration.

The digest is shape-free by default so prompts stay small.
``to_prompt_summary()`` caps at a few KB.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from xdsl.dialects.builtin import ModuleOp, TensorType

from compgen.agent.analyzer import GraphAnalysisDossier, NetworkAnalysis
from compgen.analysis.dim_semantics import (
    DimRole,
    annotate_dim_roles,
    dim_roles_for_op,
)
from compgen.kernels.contract_v3 import (
    HardwareEnvelope,
    KernelContractV3,
    MemoryTier,
)
from compgen.kernels.contract_v3_references import (
    reference_matmul_contract,
    reference_pointwise_add_contract,
    reference_silu_contract,
    reference_softmax_contract,
)
from compgen.kernels.envelope_bridge import envelope_from_target_profile
from compgen.kernels.fusion_oracle import FusionDecision, should_fuse
from compgen.kernels.granularity_oracle import recommend_granularity
from compgen.kernels.tile_oracle import TileRecommendation, recommend_tile
from compgen.targets.schema import TargetProfile

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FlopDistribution:
    total: int = 0
    top5: tuple[tuple[str, int], ...] = ()
    source: str = "analyzer"  # "analyzer" | "ir_walk_fallback"


@dataclass(frozen=True)
class ByteDistribution:
    total: int = 0
    top5: tuple[tuple[str, int], ...] = ()


@dataclass(frozen=True)
class DimSpectrum:
    """Dim-role + rank distribution, shape-free."""

    rank_histogram: dict[int, int] = field(default_factory=dict)
    parallel_dims: int = 0
    reduce_dims: int = 0
    batch_dims: int = 0
    broadcast_dims: int = 0


@dataclass(frozen=True)
class GraphDigest:
    """Overall, shape-free digest of a Payload IR module.

    Intended to be the first thing the LLM sees before it even asks for
    a focused chunk.
    """

    model_name: str
    target_name: str
    pattern_histogram: dict[str, int] = field(default_factory=dict)
    dim_spectrum: DimSpectrum = field(default_factory=DimSpectrum)
    dtype_spectrum: dict[str, int] = field(default_factory=dict)
    quant_spectrum: dict[str, int] = field(default_factory=dict)
    flop_distribution: FlopDistribution = field(default_factory=FlopDistribution)
    byte_distribution: ByteDistribution = field(default_factory=ByteDistribution)
    memory_footprint_bytes: int = 0
    critical_path: tuple[str, ...] = ()
    fusion_opportunity_count: int = 0
    pattern_size_histogram: dict[int, int] = field(default_factory=dict)
    bottleneck_ops: tuple[str, ...] = ()
    region_index: tuple[str, ...] = ()

    def to_prompt_summary(self, *, max_bytes: int = 2048) -> str:
        """Compact textual summary ≤ ``max_bytes`` for inclusion in prompts."""
        lines = [
            f"# graph_digest model={self.model_name} target={self.target_name}",
            f"patterns={dict(sorted(self.pattern_histogram.items(), key=lambda kv: -kv[1])[:8])}",
            f"ranks={dict(sorted(self.dim_spectrum.rank_histogram.items()))}",
            f"dim_roles=parallel:{self.dim_spectrum.parallel_dims} "
            f"reduce:{self.dim_spectrum.reduce_dims} "
            f"batch:{self.dim_spectrum.batch_dims} "
            f"broadcast:{self.dim_spectrum.broadcast_dims}",
            f"dtypes={self.dtype_spectrum}",
            f"quant={self.quant_spectrum}",
            f"flops_total={self.flop_distribution.total}",
            f"bytes_total={self.byte_distribution.total}",
            f"memory_footprint_bytes={self.memory_footprint_bytes}",
            f"critical_path={list(self.critical_path)[:8]}",
            f"fusion_opps={self.fusion_opportunity_count}",
            f"pattern_sizes={self.pattern_size_histogram}",
            f"bottleneck_ops={list(self.bottleneck_ops)[:8]}",
            f"regions={list(self.region_index)[:16]}",
        ]
        out = "\n".join(lines)
        if len(out) > max_bytes:
            out = out[: max_bytes - 16] + "\n… (truncated)"
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "target_name": self.target_name,
            "pattern_histogram": dict(self.pattern_histogram),
            "dim_spectrum": {
                "rank_histogram": dict(self.dim_spectrum.rank_histogram),
                "parallel_dims": self.dim_spectrum.parallel_dims,
                "reduce_dims": self.dim_spectrum.reduce_dims,
                "batch_dims": self.dim_spectrum.batch_dims,
                "broadcast_dims": self.dim_spectrum.broadcast_dims,
            },
            "dtype_spectrum": dict(self.dtype_spectrum),
            "quant_spectrum": dict(self.quant_spectrum),
            "flop_distribution": {
                "total": self.flop_distribution.total,
                "top5": list(self.flop_distribution.top5),
                "source": self.flop_distribution.source,
            },
            "byte_distribution": {
                "total": self.byte_distribution.total,
                "top5": list(self.byte_distribution.top5),
            },
            "memory_footprint_bytes": self.memory_footprint_bytes,
            "critical_path": list(self.critical_path),
            "fusion_opportunity_count": self.fusion_opportunity_count,
            "pattern_size_histogram": dict(self.pattern_size_histogram),
            "bottleneck_ops": list(self.bottleneck_ops),
            "region_index": list(self.region_index),
        }


@dataclass(frozen=True)
class DecisionKnobs:
    """Candidates surfaced for an LLM to consider — **non-binding**.

    Every candidate carries a ``source`` (which oracle or cost-model
    produced it) and an optional ``oracle_advisory`` flag marking the
    one entry the oracle would pick IF the agent didn't override.

    This is intentionally shaped like the :class:`DecisionSite`
    candidate list so the same LLM can reason over both. Field names
    say "oracle_advisory" rather than "recommended" so no reader
    assumes the oracle's pick is authoritative.

    * ``granularity_options``: every enum value, one marked advisory.
    * ``tile_options``: multi-dtype × multi-shape sweep.
    * ``memory_tier_options``: filtered by envelope viability.
    * ``fusion_options``: adjacent-op pairs with ``should_fuse``
      verdicts. A verdict of ``FUSE`` is an oracle suggestion, not
      a command.
    * ``alternatives``: extra agent-supplied or history-supplied
      candidates not enumerated by the oracle.
    """

    granularity_options: tuple[dict[str, Any], ...] = ()
    tile_options: tuple[dict[str, Any], ...] = ()
    memory_tier_options: tuple[str, ...] = ()
    fusion_options: tuple[dict[str, Any], ...] = ()
    alternatives: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True)
class DoFDescription:
    """Open-ended design-space description.

    Presented alongside :class:`DecisionKnobs` so the LLM is free to
    invent candidates beyond the oracles' recommendations. The
    infrastructure still gates novel picks through verification.
    """

    axes: tuple[str, ...] = ()
    memory_tiers: tuple[str, ...] = ()
    archetypes: tuple[str, ...] = ()
    fusion_boundaries: tuple[str, ...] = ()
    heuristic_hints: tuple[str, ...] = ()


@dataclass(frozen=True)
class ChunkView:
    """A focused subgraph view.

    Carries both a chunk-local :class:`DecisionKnobs` (safe, bounded)
    and a :class:`DoFDescription` (free-form). Concrete shapes are
    omitted by default to bound prompt size.
    """

    region_id: str
    pattern_type: str
    ops: tuple[dict[str, Any], ...] = ()
    edges: tuple[dict[str, Any], ...] = ()
    symbolic_shapes: tuple[tuple[str | None, ...], ...] = ()
    concrete_shapes: tuple[tuple[int, ...], ...] = ()
    dim_roles: tuple[str, ...] = ()
    dtypes: tuple[str, ...] = ()
    quant_attrs: dict[str, Any] = field(default_factory=dict)
    envelope_facts: dict[str, Any] = field(default_factory=dict)
    decision_knobs: DecisionKnobs = field(default_factory=DecisionKnobs)
    dof_description: DoFDescription = field(default_factory=DoFDescription)

    def to_dict(self) -> dict[str, Any]:
        return {
            "region_id": self.region_id,
            "pattern_type": self.pattern_type,
            "ops": list(self.ops),
            "edges": list(self.edges),
            "symbolic_shapes": [list(s) for s in self.symbolic_shapes],
            "concrete_shapes": [list(s) for s in self.concrete_shapes],
            "dim_roles": list(self.dim_roles),
            "dtypes": list(self.dtypes),
            "quant_attrs": dict(self.quant_attrs),
            "envelope_facts": dict(self.envelope_facts),
            "decision_knobs": {
                "advisory_nature": "non-binding; agent is the decider",
                "granularity_options": list(self.decision_knobs.granularity_options),
                "tile_options": list(self.decision_knobs.tile_options),
                "memory_tier_options": list(self.decision_knobs.memory_tier_options),
                "fusion_options": list(self.decision_knobs.fusion_options),
                "alternatives": list(self.decision_knobs.alternatives),
            },
            "dof_description": {
                "axes": list(self.dof_description.axes),
                "memory_tiers": list(self.dof_description.memory_tiers),
                "archetypes": list(self.dof_description.archetypes),
                "fusion_boundaries": list(self.dof_description.fusion_boundaries),
                "heuristic_hints": list(self.dof_description.heuristic_hints),
            },
        }


# ---------------------------------------------------------------------------
# IR walkers
# ---------------------------------------------------------------------------


_STRUCTURAL_OP_NAMES = {"builtin.module", "func.func", "func.return", "builtin.unrealized_conversion_cast"}


def _is_structural(op_name: str) -> bool:
    return op_name in _STRUCTURAL_OP_NAMES


_MATMUL_OP_NAMES = {
    "linalg.matmul",
    "linalg.batch_matmul",
    "linalg.quantized_matmul",
    "aten.mm",
    "aten.bmm",
    "aten.addmm",
    "aten.matmul",
}


def _flops_from_ir(module: ModuleOp) -> int:
    """IR-walking FLOP estimator used when ``NetworkAnalysis.total_flops``
    is zero (the FX-side accumulator mis-fires on some patterns).

    Heuristic per op:

    * **matmul-like**: 2·M·N·K using the last two result dims for the
      output shape and the inner shared dim from the input that isn't the
      output's (first) operand.
    * **pointwise** (arith/elementwise linalg): one FLOP per output
      element — rough, but non-zero where the graph has real compute.
    * All other ops count as 0.
    """
    total = 0
    for op in module.walk():
        if _is_structural(op.name):
            continue
        name = op.name.lower()
        is_matmul = any(n in name for n in _MATMUL_OP_NAMES) or "matmul" in name or "bmm" in name
        out_shape: tuple[int, ...] = ()
        for res in op.results:
            if isinstance(res.type, TensorType):
                out_shape = tuple(d for d in res.type.get_shape() if d > 0)
                break
        if not out_shape:
            continue
        out_elems = 1
        for d in out_shape:
            out_elems *= d
        if is_matmul:
            # For matmul, the inner K dim lives on the second operand.
            k_dim = 0
            for operand in op.operands:
                t = getattr(operand, "type", None)
                if not isinstance(t, TensorType):
                    continue
                dims = [d for d in t.get_shape() if d > 0]
                if len(dims) >= 2:
                    k_dim = dims[-2] if k_dim == 0 else k_dim
                    break
            if k_dim == 0:
                k_dim = out_shape[-1]  # conservative fallback
            total += 2 * out_elems * k_dim
        else:
            total += out_elems  # 1 FLOP per output element (pointwise proxy)
    return total


def _dtype_name(t: Any) -> str:
    try:
        elem = t.get_element_type() if hasattr(t, "get_element_type") else t
        name = type(elem).__name__.lower()
    except Exception:  # noqa: BLE001
        return "unknown"
    # Map xdsl element-type class names to short dtype tags
    mapping = {
        "bfloat16type": "bf16",
        "float16type": "f16",
        "float32type": "f32",
        "float64type": "f64",
        "integertype": "int",
        "float8e4m3fntype": "fp8_e4m3",
        "float8e5m2type": "fp8_e5m2",
    }
    return mapping.get(name, name)


def _walk_tensor_results(module: ModuleOp) -> list[tuple[Any, TensorType]]:
    results: list[tuple[Any, TensorType]] = []
    for op in module.walk():
        if _is_structural(op.name):
            continue
        for res in op.results:
            if isinstance(res.type, TensorType):
                results.append((op, res.type))
    return results


# ---------------------------------------------------------------------------
# Digester
# ---------------------------------------------------------------------------


class GraphDigester:
    """Build :class:`GraphDigest` from analyzer output + IR."""

    def __init__(
        self,
        analysis: NetworkAnalysis,
        module: ModuleOp | None = None,
        *,
        target_name: str = "",
    ) -> None:
        self.analysis = analysis
        self.module = module
        self.target_name = target_name or (
            analysis.dossier.model_name if analysis.dossier is not None else ""
        )

    def digest(self) -> GraphDigest:
        dossier: GraphAnalysisDossier | None = self.analysis.dossier
        pattern_hist = Counter(c.pattern_type for c in self.analysis.clusters)
        pattern_size_hist = Counter(len(c.node_names) for c in self.analysis.clusters)

        # Ensure dim-role attributes are present on the module before
        # we count them (idempotent — annotates any op without the attr).
        if self.module is not None:
            try:
                annotate_dim_roles(self.module)
            except Exception:  # noqa: BLE001
                # Annotation is best-effort; fall through to a zero count.
                pass

        # Dim spectrum — rank + dtype + quant from tensor result types;
        # dim-role counts from the ``compgen.dim_role`` attribute that
        # ``dim_semantics`` just stamped on every op.
        rank_hist: Counter[int] = Counter()
        dtype_spectrum: Counter[str] = Counter()
        quant_spectrum: Counter[str] = Counter()
        role_counts: Counter[DimRole] = Counter()
        memory_footprint = 0
        if self.module is not None:
            for op, ttype in _walk_tensor_results(self.module):
                shape = list(ttype.get_shape())
                rank_hist[len(shape)] += 1
                dtype = _dtype_name(ttype)
                dtype_spectrum[dtype] += 1
                if dtype.startswith("fp8") or dtype.startswith("int"):
                    quant_spectrum[dtype] += 1
                elem_bits = getattr(ttype.get_element_type(), "bitwidth", 32)
                elem_bytes = max(1, (elem_bits + 7) // 8)
                n = 1
                for d in shape:
                    if d > 0:
                        n *= d
                memory_footprint += n * elem_bytes
                for role in dim_roles_for_op(op):
                    role_counts[role] += 1

        dim_spectrum = DimSpectrum(
            rank_histogram=dict(rank_hist),
            parallel_dims=role_counts.get(DimRole.PARALLEL, 0),
            reduce_dims=role_counts.get(DimRole.REDUCE, 0),
            batch_dims=role_counts.get(DimRole.BATCH, 0),
            broadcast_dims=role_counts.get(DimRole.BROADCAST, 0),
        )

        # FLOP / byte distribution (top 5 clusters by flops / bytes)
        flop_entries = sorted(
            ((c.cluster_id, c.total_flops) for c in self.analysis.clusters),
            key=lambda kv: -kv[1],
        )[:5]
        byte_entries = sorted(
            ((c.cluster_id, c.total_bytes) for c in self.analysis.clusters),
            key=lambda kv: -kv[1],
        )[:5]

        critical_path: tuple[str, ...] = ()
        region_index: tuple[str, ...] = ()
        if dossier is not None:
            critical_path = tuple(dossier.critical_path)
            region_index = tuple(r.region_id for r in dossier.regions)

        fusion_opps = sum(
            1
            for opp in self.analysis.optimization_opportunities
            if "fus" in opp.lower()
        )

        # Fallback: if the FX-level analyzer returned zero FLOPs (happens
        # on some captures where shape meta is missing), walk the IR and
        # estimate. Gap #8.
        total_flops = self.analysis.total_flops
        flop_source = "analyzer"
        if total_flops == 0 and self.module is not None:
            total_flops = _flops_from_ir(self.module)
            flop_source = "ir_walk_fallback"

        return GraphDigest(
            model_name=self.analysis.model_name,
            target_name=self.target_name,
            pattern_histogram=dict(pattern_hist),
            dim_spectrum=dim_spectrum,
            dtype_spectrum=dict(dtype_spectrum),
            quant_spectrum=dict(quant_spectrum),
            flop_distribution=FlopDistribution(
                total=total_flops,
                top5=tuple(flop_entries),
                source=flop_source,
            ),
            byte_distribution=ByteDistribution(
                total=self.analysis.total_bytes,
                top5=tuple(byte_entries),
            ),
            memory_footprint_bytes=memory_footprint,
            critical_path=critical_path,
            fusion_opportunity_count=fusion_opps,
            pattern_size_histogram=dict(pattern_size_hist),
            bottleneck_ops=tuple(self.analysis.bottleneck_clusters[:8]),
            region_index=region_index,
        )


# ---------------------------------------------------------------------------
# Chunk extractor + knob enumerator
# ---------------------------------------------------------------------------


@dataclass
class ChunkSelector:
    region_id: str = ""
    pattern_type: str = ""
    cluster_id: str = ""
    node_names: tuple[str, ...] = ()


def _selector_from_dict(selector: dict[str, Any]) -> ChunkSelector:
    return ChunkSelector(
        region_id=str(selector.get("region_id", "")),
        pattern_type=str(selector.get("pattern_type", "")),
        cluster_id=str(selector.get("cluster_id", "")),
        node_names=tuple(selector.get("node_names", []) or ()),
    )


class ChunkExtractor:
    """Extract a :class:`ChunkView` for a selector against analysis + IR."""

    def __init__(
        self,
        analysis: NetworkAnalysis,
        target: TargetProfile,
        module: ModuleOp | None = None,
    ) -> None:
        self.analysis = analysis
        self.target = target
        self.module = module
        try:
            self.envelope: HardwareEnvelope | None = envelope_from_target_profile(target)
        except Exception:  # noqa: BLE001
            self.envelope = None
        # Stamp dim-role attributes once so ``_cluster_dim_roles`` can
        # read them off ops directly. Idempotent + safe on pre-annotated modules.
        if module is not None:
            try:
                annotate_dim_roles(module)
            except Exception:  # noqa: BLE001
                pass

    def extract(self, selector: dict[str, Any], *, include_concrete_shapes: bool = False) -> ChunkView:
        sel = _selector_from_dict(selector)
        cluster = self._resolve_cluster(sel)
        if cluster is None:
            cluster = self._synthesize_cluster_from_dossier(sel)
        if cluster is None:
            # Still nothing to focus on — return a chunk with empty ops
            # but keep envelope facts + DoF so the LLM sees the design
            # space even when the graph has no recognised patterns.
            synth = _SyntheticCluster(cluster_id="<none>", pattern_type="unknown")
            return ChunkView(
                region_id=sel.region_id or "<none>",
                pattern_type="unknown",
                envelope_facts=self._envelope_facts(),
                decision_knobs=self._enumerate_knobs(synth),
                dof_description=self._describe_dof(synth),
            )

        ops_payload = tuple({"name": n} for n in cluster.node_names)
        edges_payload = tuple(
            {"src": a, "dst": b, "operand_idx": 0}
            for a, b in zip(cluster.node_names, cluster.node_names[1:])
        )
        symbolic_shapes = tuple(tuple(None for _ in s) for s in cluster.input_shapes.values())
        concrete_shapes: tuple[tuple[int, ...], ...] = ()
        if include_concrete_shapes:
            concrete_shapes = tuple(tuple(s) for s in cluster.input_shapes.values())
        dtypes = self._cluster_dtypes(cluster)
        dim_roles_tuple = self._cluster_dim_roles(cluster)
        envelope_facts = self._envelope_facts()
        knobs = self._enumerate_knobs(cluster)
        dof = self._describe_dof(cluster)
        return ChunkView(
            region_id=cluster.cluster_id,
            pattern_type=cluster.pattern_type,
            ops=ops_payload,
            edges=edges_payload,
            symbolic_shapes=symbolic_shapes,
            concrete_shapes=concrete_shapes,
            dim_roles=dim_roles_tuple,
            dtypes=dtypes,
            quant_attrs=self._quant_attrs(dtypes),
            envelope_facts=envelope_facts,
            decision_knobs=knobs,
            dof_description=dof,
        )

    def _resolve_cluster(self, sel: ChunkSelector) -> Any | None:
        for c in self.analysis.clusters:
            if sel.cluster_id and c.cluster_id == sel.cluster_id:
                return c
            if sel.region_id and c.cluster_id == sel.region_id:
                return c
            if sel.pattern_type and c.pattern_type == sel.pattern_type:
                return c
            if sel.node_names and set(sel.node_names).issubset(set(c.node_names)):
                return c
        if self.analysis.clusters:
            return self.analysis.clusters[0]
        return None

    def _synthesize_cluster_from_dossier(self, sel: ChunkSelector) -> Any | None:
        """Fall back to dossier regions when no pattern cluster matches.

        Models without recognised patterns (e.g. simple MLPs that decompose
        to permute/addmm) still need a usable chunk view. We build a
        :class:`_SyntheticCluster` from the first matching region dossier
        so :meth:`extract` can populate knobs + DoF.
        """
        dossier = self.analysis.dossier
        if dossier is None or not dossier.regions:
            return None
        chosen = None
        for region in dossier.regions:
            if sel.region_id and region.region_id == sel.region_id:
                chosen = region
                break
            if sel.pattern_type and sel.pattern_type in region.kind:
                chosen = region
                break
        if chosen is None:
            chosen = dossier.regions[0]
        return _SyntheticCluster(
            cluster_id=chosen.region_id,
            pattern_type=chosen.kind,
            node_names=tuple(chosen.node_names),
        )

    def _cluster_ops(self, cluster: Any) -> list[Any]:
        """Resolve xDSL ops that correspond to the cluster's FX node names.

        FX nodes use names like ``rmsnorm_0`` / ``mm_3`` while xDSL ops
        are named ``aten.rms_norm`` / ``linalg.matmul``. We match by
        converting FX names to a normalised key (``mm`` → ``matmul``,
        ``rmsnorm`` → ``rms_norm`` / ``rmsnorm``) and substring-search
        against ``op.name``. Falls back to every op in the module when
        no match succeeds so the caller still sees representative data.
        """
        if self.module is None:
            return []
        node_names = list(cluster.node_names)
        keys: set[str] = set()
        for n in node_names:
            base = n.rsplit("_", 1)[0] if n.split("_")[-1].isdigit() else n
            base = base.lower()
            keys.add(base)
            # common FX → xDSL aliases
            if base == "mm":
                keys.add("matmul")
            elif base == "bmm":
                keys.add("batch_matmul")
                keys.add("bmm")
            elif base == "rmsnorm":
                keys.add("rms_norm")
            elif base == "softmax":
                keys.add("_softmax")
        matched: list[Any] = []
        for op in self.module.walk():
            name = op.name.lower()
            if any(k in name for k in keys):
                matched.append(op)
        return matched

    def _cluster_dtypes(self, cluster: Any) -> tuple[str, ...]:
        seen: list[str] = []
        for op in self._cluster_ops(cluster):
            for res in op.results:
                if isinstance(res.type, TensorType):
                    d = _dtype_name(res.type)
                    if d not in seen:
                        seen.append(d)
        if not seen and self.module is not None:
            # Fallback — surface the module-wide dtype set so callers
            # still see something useful for pattern-free chunks.
            for _op, ttype in _walk_tensor_results(self.module):
                d = _dtype_name(ttype)
                if d not in seen:
                    seen.append(d)
        return tuple(seen)

    def _cluster_dim_roles(self, cluster: Any) -> tuple[str, ...]:
        """Collect the stamped ``compgen.dim_role`` attrs for ops in the cluster."""
        seen: list[str] = []
        for op in self._cluster_ops(cluster):
            for role in dim_roles_for_op(op):
                if role.value not in seen:
                    seen.append(role.value)
        return tuple(seen)

    def _quant_attrs(self, dtypes: tuple[str, ...]) -> dict[str, Any]:
        flags = {d: True for d in dtypes if d.startswith("fp8") or d.startswith("int")}
        return {"has_quantization": bool(flags), "quant_dtypes": sorted(flags.keys())}

    def _envelope_facts(self) -> dict[str, Any]:
        if self.envelope is None:
            return {"target": self.target.name}
        env = self.envelope
        return {
            "target": env.target_name,
            "vector_lanes": env.vector_lanes,
            "scratchpad_bytes": env.scratchpad_bytes,
            "register_bytes": env.register_bytes,
            "peak_bandwidth_gbps": env.peak_bandwidth_gbps,
            "native_dtypes": list(env.native_dtypes),
            "mma_shapes": {k: list(v) for k, v in env.mma_shapes.items()},
        }

    def _enumerate_knobs(self, cluster: Any) -> DecisionKnobs:
        """Ask each oracle for real candidates relevant to this cluster.

        - **Granularity**: run :func:`recommend_granularity` per
          synthesized :class:`KernelContractV3` in the chunk; surface
          all three enum values but annotate which one the oracle
          recommends per op plus its reason.
        - **Memory tiers**: the full 5-value enum filtered by what the
          target envelope actually has (for example, drop
          ``scratchpad`` when the envelope's scratchpad_bytes is 0).
        - **Tile**: call :func:`recommend_tile` across the envelope's
          native dtypes × the cluster's shape set → multiple candidates.
        - **Fusion**: for every adjacent-op pair, build two contracts
          via :meth:`_contract_for_op_family` and call
          :func:`should_fuse`. Each candidate carries the oracle's real
          ``FusionDecision`` + estimated speedup + reason.
        """
        envelope = self.envelope
        memory_tiers = self._viable_memory_tiers()
        tile_options = self._enumerate_tile_candidates(cluster)
        granularity_options = self._enumerate_granularity_candidates(cluster)
        fusion_options = self._enumerate_fusion_candidates(cluster)
        return DecisionKnobs(
            granularity_options=granularity_options,
            tile_options=tile_options,
            memory_tier_options=memory_tiers,
            fusion_options=fusion_options,
            alternatives=(),
        )

    # -- viable memory tiers ---------------------------------------------

    def _viable_memory_tiers(self) -> tuple[str, ...]:
        """Filter :class:`MemoryTier` by what the envelope actually has."""
        if self.envelope is None:
            return tuple(t.value for t in MemoryTier)
        out: list[str] = [MemoryTier.REGISTER.value]
        if self.envelope.scratchpad_bytes > 0:
            out.append(MemoryTier.SCRATCHPAD.value)
        # L2 is near-universal on GPU/CPU targets; include when bandwidth > 0.
        if self.envelope.peak_bandwidth_gbps > 0:
            out.append(MemoryTier.L2.value)
        out.append(MemoryTier.DEVICE_DRAM.value)
        out.append(MemoryTier.HOST.value)
        return tuple(out)

    # -- tile candidates -------------------------------------------------

    def _enumerate_tile_candidates(self, cluster: Any) -> tuple[dict[str, Any], ...]:
        if self.envelope is None:
            return ()
        op_family = _pattern_to_op_family(cluster.pattern_type)
        # Sweep the native dtypes the envelope declares — that gives
        # real per-dtype tile recommendations instead of a single pick.
        dtype_list = list(self.envelope.native_dtypes) or ["bf16"]
        # Normalise HF-style names like "float16" to short codes used by
        # the oracle's dtype table.
        alias = {
            "float16": "f16",
            "float32": "f32",
            "float64": "f64",
            "bfloat16": "bf16",
            "int8": "i8",
            "int16": "i16",
            "int32": "i32",
        }
        dtype_list = [alias.get(d, d) for d in dtype_list]
        shapes: list[tuple[int | None, ...]] = []
        for s in cluster.input_shapes.values():
            shapes.append(tuple(s))
        if not shapes:
            shapes = [()]
        out: list[dict[str, Any]] = []
        advisory_key: tuple[str, str] | None = None
        for dtype in dtype_list:
            for shape in shapes:
                try:
                    rec: TileRecommendation = recommend_tile(
                        op_family=op_family,
                        shape=shape,
                        dtype=dtype,
                        envelope=self.envelope,
                    )
                except Exception:  # noqa: BLE001
                    continue
                is_advisory = advisory_key is None and rec.confidence >= 0.5
                if is_advisory:
                    advisory_key = (dtype, str(shape))
                out.append(
                    {
                        "op_family": op_family,
                        "source": "oracle:tile",
                        "dtype": dtype,
                        "shape": list(shape),
                        "block_m": rec.block_m,
                        "block_n": rec.block_n,
                        "block_k": rec.block_k,
                        "num_warps": rec.num_warps,
                        "num_stages": rec.num_stages,
                        "group_m": rec.group_m,
                        "rationale": rec.rationale,
                        "confidence": rec.confidence,
                        "oracle_advisory": is_advisory,
                    }
                )
        return tuple(out)

    # -- granularity candidates -----------------------------------------

    def _enumerate_granularity_candidates(self, cluster: Any) -> tuple[dict[str, Any], ...]:
        """Run ``recommend_granularity`` on a contract synthesized from
        the cluster's pattern type, so the three MICRO/NORMAL/MEGA
        options are annotated with the oracle's live pick + reason.
        """
        if self.envelope is None:
            return tuple(
                {
                    "granularity": g,
                    "source": "oracle:granularity",
                    "oracle_advisory": False,
                    "reason": "no envelope — candidates unranked",
                }
                for g in ("MICRO", "NORMAL", "MEGA")
            )
        contract = _contract_for_pattern(cluster.pattern_type, envelope=self.envelope)
        chain: list[KernelContractV3] = [contract]
        if len(cluster.node_names) >= 2:
            # A multi-op cluster stresses the NORMAL vs MEGA decision;
            # append a second contract so the oracle runs its chain path.
            chain.append(contract)
        try:
            verdict = recommend_granularity(chain, self.envelope)
            picked = verdict.granularity.value.upper()
            reason = verdict.reason
            confidence = verdict.confidence
        except Exception as exc:  # noqa: BLE001
            picked = "NORMAL"
            reason = f"granularity oracle error: {exc}"
            confidence = 0.0
        return tuple(
            {
                "granularity": g,
                "source": "oracle:granularity",
                "oracle_advisory": g == picked,
                "reason": reason if g == picked else "",
                "confidence": confidence if g == picked else 0.0,
            }
            for g in ("MICRO", "NORMAL", "MEGA")
        )

    # -- fusion candidates ----------------------------------------------

    def _enumerate_fusion_candidates(self, cluster: Any) -> tuple[dict[str, Any], ...]:
        """For every adjacent (producer, consumer) pair in the cluster,
        call :func:`should_fuse` with contracts synthesized from the
        op's inferred pattern. Falls back to "unknown" only when the
        oracle raises.
        """
        if self.envelope is None or len(cluster.node_names) < 2:
            return tuple()
        out: list[dict[str, Any]] = []
        pattern = cluster.pattern_type
        producer = _contract_for_pattern(pattern, envelope=self.envelope)
        consumer = _contract_for_pattern(pattern, envelope=self.envelope)
        for a, b in zip(cluster.node_names, cluster.node_names[1:]):
            try:
                verdict = should_fuse(producer, consumer)
                out.append(
                    {
                        "src": a,
                        "dst": b,
                        "source": "oracle:fusion",
                        "oracle_verdict": verdict.decision.value,
                        "est_speedup": round(verdict.est_speedup_ratio, 3),
                        "reason": verdict.reason,
                        "eligibility_failures": list(verdict.eligibility_failures),
                        "binding": False,
                    }
                )
            except Exception as exc:  # noqa: BLE001
                out.append(
                    {
                        "src": a,
                        "dst": b,
                        "source": "oracle:fusion",
                        "oracle_verdict": "unknown",
                        "error": str(exc),
                        "binding": False,
                    }
                )
        return tuple(out)

    def _describe_dof(self, cluster: Any) -> DoFDescription:
        """Free-form design-space description for LLM creativity."""
        axes: list[str] = []
        if self.module is not None:
            for _op, ttype in _walk_tensor_results(self.module):
                shape = list(ttype.get_shape())
                if not shape:
                    continue
                # Emit abstract axes like "dim0", "dim1"... once per rank.
                for i, _ in enumerate(shape):
                    name = f"dim{i}"
                    if name not in axes:
                        axes.append(name)
        archetypes = (
            "COMPUTE_TILED",
            "POINTWISE",
            "REDUCE",
            "MEMORY",
            "ACTIVATION",
        )
        heuristic_hints: list[str] = []
        if self.envelope is not None:
            heuristic_hints.extend(self.envelope.codegen_hints)
        # Encode opportunities that mention this cluster
        for opp in self.analysis.optimization_opportunities:
            if cluster.cluster_id in opp or cluster.pattern_type in opp:
                heuristic_hints.append(opp)
        fusion_boundaries = tuple(
            f"{a}→{b}"
            for a, b in zip(cluster.node_names, cluster.node_names[1:])
        )
        return DoFDescription(
            axes=tuple(axes[:8]),
            memory_tiers=tuple(t.value for t in MemoryTier),
            archetypes=archetypes,
            fusion_boundaries=fusion_boundaries,
            heuristic_hints=tuple(heuristic_hints[:8]),
        )


def _best_op_label(op: Any) -> str:
    """Best-effort label matching FX node naming conventions."""
    return getattr(op, "name", "") or type(op).__name__


def _contract_for_pattern(pattern: str, *, envelope: HardwareEnvelope) -> KernelContractV3:
    """Synthesize a real :class:`KernelContractV3` for the given pattern.

    Reuses the authored reference contracts in
    :mod:`compgen.kernels.contract_v3_references` and re-homes them onto
    the caller's live :class:`HardwareEnvelope` so the downstream
    oracles (``should_fuse`` / ``recommend_granularity``) see the
    target's real caps instead of the reference's default envelope.
    """
    from dataclasses import replace

    from compgen.kernels.contract_v3 import (
        ExecutionEnvelope,
        OrchestrationSpec,
    )

    pat = pattern.lower()
    if "matmul" in pat or "linear" in pat or "attention" in pat or "mm" == pat:
        base = reference_matmul_contract()
    elif "softmax" in pat:
        base = reference_softmax_contract()
    elif "silu" in pat or "activation" in pat:
        base = reference_silu_contract()
    else:
        base = reference_pointwise_add_contract()

    exec_env = ExecutionEnvelope(hardware=envelope)
    new_orch = replace(base.orchestration, execution=exec_env)
    return replace(base, orchestration=new_orch)


@dataclass(frozen=True)
class _SyntheticCluster:
    """Stand-in for :class:`PatternCluster` when no clusters match.

    Only carries the fields :class:`ChunkExtractor` reads.
    """

    cluster_id: str
    pattern_type: str
    node_names: tuple[str, ...] = ()
    input_shapes: dict[str, tuple[int, ...]] = field(default_factory=dict)


def _pattern_to_op_family(pattern: str) -> str:
    mapping = {
        "matmul": "matmul",
        "linear": "matmul",
        "batch_matmul": "batch_matmul",
        "gqa_attention": "matmul",
        "attention": "matmul",
        "softmax": "softmax",
        "rms_norm": "rmsnorm",
        "rmsnorm": "rmsnorm",
        "silu": "silu",
    }
    key = pattern.lower()
    for needle, family in mapping.items():
        if needle in key:
            return family
    return "matmul"


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------


def build_digest(
    analysis: NetworkAnalysis,
    *,
    module: ModuleOp | None = None,
    target_name: str = "",
) -> GraphDigest:
    return GraphDigester(analysis, module=module, target_name=target_name).digest()


def build_chunk_view(
    analysis: NetworkAnalysis,
    target: TargetProfile,
    selector: dict[str, Any],
    *,
    module: ModuleOp | None = None,
    include_concrete_shapes: bool = False,
) -> ChunkView:
    return ChunkExtractor(analysis, target, module=module).extract(
        selector, include_concrete_shapes=include_concrete_shapes
    )


def digest_to_json(digest: GraphDigest) -> str:
    return json.dumps(digest.to_dict(), default=str)


__all__ = [
    "ByteDistribution",
    "ChunkExtractor",
    "ChunkSelector",
    "ChunkView",
    "DecisionKnobs",
    "DimSpectrum",
    "DoFDescription",
    "FlopDistribution",
    "GraphDigest",
    "GraphDigester",
    "build_chunk_view",
    "build_digest",
    "digest_to_json",
]
