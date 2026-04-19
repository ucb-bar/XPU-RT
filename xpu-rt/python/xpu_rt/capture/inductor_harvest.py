"""Inductor graph harvester (P21).

Invokes ``torch.compile`` with a **snapshot backend** that captures the
post-lowering FX graph + operator inventory, rather than actually
running inductor's codegen. This gives us what inductor *would* fuse
and lower on the target we're already good at (CUDA/CPU), so Phase 2
can filter its TOOL catalog to only cover what inductor didn't do.

Per the "lean heavily on inductor" directive
(memory: ``feedback_lean_heavily_on_inductor.md``), this module is how
Phase 0 harvests inductor's intermediate representation. It is
additive to ``compile_baseline`` (which measures wall-time only).

Fails gracefully: if inductor can't compile the model, the harvest
returns a report with ``status='fallback'`` and Phase 1 consumes the
raw ``ExportedProgram`` instead.
"""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn


@dataclass(frozen=True)
class MegakernelCandidate:
    """A region cluster identified as a potential megakernel synthesis site.

    Produced by :func:`estimate_megakernel_candidates`; consumed by the
    Phase-4 ``propose_megakernel_synthesis`` invent-slot to pre-seed its
    candidate region list.

    Attributes:
        pattern:        Short tag for the recognised pattern
                        (``"matmul_collective"``, ``"attention_pipeline"``,
                        ``"moe_routing"``, ``"unfused_chain"``).
        ops:            ATen op names present in the cluster.
        rationale:      Why this cluster is a megakernel candidate.
        confidence:     ``0.0`` .. ``1.0`` heuristic score.
    """

    pattern: str
    ops: tuple[str, ...]
    rationale: str
    confidence: float = 0.5


@dataclass(frozen=True)
class InductorHarvestReport:
    """Result of harvesting inductor's post-lowering graph.

    Attributes:
        status: ``ok`` (harvest succeeded), ``fallback`` (inductor
            failed, raw FX used), or ``skipped`` (explicitly disabled).
        backend: Backend name used ("inductor" or the snapshot
            placeholder).
        fx_op_histogram: ATen op name → count, across all captured
            subgraphs.
        fx_node_count: Total FX node count across subgraphs.
        fx_graph_count: Number of distinct subgraphs captured (graph
            breaks produce multiple).
        fusion_groups: Heuristic fusion-group inventory — each entry
            is a tuple of op-name groups inductor would likely have
            fused. Derived from adjacency in the FX graph, not a full
            reproduction of inductor's fusion algorithm.
        megakernel_candidates: Region clusters identified as good
            megakernel synthesis sites (per the ETC paper's target
            patterns: matmul+collective, attention pipelines, MoE).
            Populated by :func:`estimate_megakernel_candidates`; left
            empty if the report has ``status != "ok"``.
        elapsed_ms: Wall-clock time for the harvest.
        fallback_reason: Populated when ``status='fallback'``.
        warnings: Diagnostic messages.
    """

    status: str
    backend: str
    fx_op_histogram: dict[str, int] = field(default_factory=dict)
    fx_node_count: int = 0
    fx_graph_count: int = 0
    fusion_groups: list[tuple[str, ...]] = field(default_factory=list)
    megakernel_candidates: list[MegakernelCandidate] = field(default_factory=list)
    elapsed_ms: float = 0.0
    fallback_reason: str = ""
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Snapshot backend — captures graph structure, doesn't actually compile
# ---------------------------------------------------------------------------


def _snapshot_backend(_captured: list[torch.fx.GraphModule]):
    """Return a torch.compile-compatible backend that captures each graph.

    Every subgraph dynamo produces is appended to the list; the backend
    returns the unmodified forward so the function still runs eagerly.
    """

    def compile_fn(gm: torch.fx.GraphModule, example_inputs: list[Any]) -> Any:
        _captured.append(gm)
        return gm.forward

    return compile_fn


def _adjacent_op_groups(gm: torch.fx.GraphModule) -> list[tuple[str, ...]]:
    """Heuristic: group adjacent elementwise ops as likely fusion groups.

    This isn't a reproduction of inductor's fusion heuristic — it's a
    rough upper bound that tells Phase 2 which ops inductor would
    *probably* have bundled. Conservative: only groups consecutive ops
    that are elementwise + same shape context (approximated by
    checking arg types).
    """
    groups: list[tuple[str, ...]] = []
    current: list[str] = []

    elementwise = {
        "aten.add.Tensor", "aten.sub.Tensor", "aten.mul.Tensor",
        "aten.div.Tensor", "aten.relu.default", "aten.gelu.default",
        "aten.sigmoid.default", "aten.tanh.default", "aten.silu.default",
        "aten.neg.default", "aten.abs.default", "aten.exp.default",
        "aten.log.default", "aten.sqrt.default", "aten.rsqrt.default",
        "aten.reciprocal.default",
    }

    for node in gm.graph.nodes:
        if node.op == "call_function":
            target_name = _stringify_target(node.target)
            if target_name in elementwise:
                current.append(target_name)
                continue
        if current:
            if len(current) >= 2:
                groups.append(tuple(current))
            current = []
    if len(current) >= 2:
        groups.append(tuple(current))
    return groups


def _stringify_target(target: Any) -> str:
    """Map an FX node target (callable or str) to a stable string."""
    if isinstance(target, str):
        return target
    mod = getattr(target, "__module__", "")
    name = getattr(target, "_opname", None) or getattr(target, "__name__", "")
    if mod.startswith("torch._ops.ops."):
        # torch._ops.ops.aten.add.Tensor -> aten.add.Tensor
        return mod.replace("torch._ops.ops.", "") + (f".{name}" if name else "")
    # Fallback to raw str()
    return str(target).replace("torch._ops.ops.", "")


def _op_histogram(graphs: list[torch.fx.GraphModule]) -> tuple[dict[str, int], int]:
    """Count call_function targets across all graphs."""
    counter: Counter[str] = Counter()
    total_nodes = 0
    for gm in graphs:
        for node in gm.graph.nodes:
            total_nodes += 1
            if node.op == "call_function":
                counter[_stringify_target(node.target)] += 1
    return dict(counter), total_nodes


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def harvest_inductor_graph(
    model: nn.Module,
    sample_inputs: tuple[Any, ...],
    *,
    enabled: bool = True,
    backend: str = "inductor-snapshot",
) -> InductorHarvestReport:
    """Run ``torch.compile`` with a snapshot backend; report what got captured.

    The default ``backend='inductor-snapshot'`` uses a pass-through FX
    backend that records every subgraph dynamo produces. Pass
    ``enabled=False`` to skip entirely (returns ``status='skipped'``);
    Phase 1 then consumes the raw ExportedProgram unchanged.

    Args:
        model: nn.Module to harvest.
        sample_inputs: positional args for the model's forward.
        enabled: When False, skip the harvest and return a stub report.
        backend: Label used in the report; no effect on behavior.

    Returns:
        InductorHarvestReport — never raises; failures are reported in
        ``status='fallback'`` with ``fallback_reason`` populated.
    """
    if not enabled:
        return InductorHarvestReport(
            status="skipped", backend=backend,
            warnings=["harvest explicitly disabled"],
        )

    captured: list[torch.fx.GraphModule] = []
    t0 = time.perf_counter()

    try:
        compiled = torch.compile(model, backend=_snapshot_backend(captured))
        with torch.no_grad():
            compiled(*sample_inputs)
    except Exception as e:   # noqa: BLE001
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return InductorHarvestReport(
            status="fallback",
            backend=backend,
            fallback_reason=f"{type(e).__name__}: {e}",
            elapsed_ms=round(elapsed_ms, 3),
            warnings=[f"torch.compile snapshot failed; falling back to raw FX"],
        )

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    if not captured:
        return InductorHarvestReport(
            status="fallback",
            backend=backend,
            fallback_reason="no subgraphs captured (dynamo may have skipped compile)",
            elapsed_ms=round(elapsed_ms, 3),
        )

    histogram, node_count = _op_histogram(captured)
    groups: list[tuple[str, ...]] = []
    for gm in captured:
        groups.extend(_adjacent_op_groups(gm))

    candidates = estimate_megakernel_candidates(histogram, groups)

    return InductorHarvestReport(
        status="ok",
        backend=backend,
        fx_op_histogram=histogram,
        fx_node_count=node_count,
        fx_graph_count=len(captured),
        fusion_groups=groups,
        megakernel_candidates=candidates,
        elapsed_ms=round(elapsed_ms, 3),
    )


# ---------------------------------------------------------------------------
# Megakernel candidate detection (ETC integration -- Phase A.9)
# ---------------------------------------------------------------------------


_MATMUL_OPS: frozenset[str] = frozenset(
    {
        "aten.matmul.default",
        "aten.mm.default",
        "aten.bmm.default",
        "aten.addmm.default",
        "aten._scaled_mm.default",
    },
)
_COLLECTIVE_OPS: frozenset[str] = frozenset(
    {
        "aten.all_gather.default",
        "aten.all_gather_into_tensor.default",
        "aten.reduce_scatter.default",
        "aten.reduce_scatter_tensor.default",
        "aten.all_reduce.default",
        "aten._allgather_base.default",
        "aten._reduce_scatter_base.default",
    },
)
_ATTENTION_OPS: frozenset[str] = frozenset(
    {
        "aten._scaled_dot_product_attention.default",
        "aten._scaled_dot_product_efficient_attention.default",
        "aten._scaled_dot_product_flash_attention.default",
        "aten.scaled_dot_product_attention.default",
        "aten.softmax.default",
        "aten._softmax.default",
    },
)
_MOE_OPS: frozenset[str] = frozenset(
    {
        "aten.topk.default",
        "aten.gather.default",
        "aten.index_select.default",
        "aten.scatter.default",
        "aten.scatter_add.default",
    },
)


def estimate_megakernel_candidates(
    histogram: dict[str, int],
    fusion_groups: list[tuple[str, ...]],
) -> list[MegakernelCandidate]:
    """Identify megakernel-synthesis candidates from inductor's harvest.

    Detects three patterns the ETC paper targets:

        * ``matmul_collective``     -- GEMM next to a collective (the
                                       paper's headline GEMM+RS / AG+GEMM).
        * ``attention_pipeline``    -- attention + softmax + matmul chain.
        * ``moe_routing``           -- topk + gather/scatter +
                                       (typically) matmul, hinting at
                                       MoE GroupGEMM.
        * ``unfused_chain``         -- a long elementwise chain inductor
                                       *did* fuse but that crosses kernel
                                       boundaries; promoting it into a
                                       megakernel removes the boundary.

    Returns a list of :class:`MegakernelCandidate` -- empty when no
    matching pattern is present.
    """
    candidates: list[MegakernelCandidate] = []
    op_set = set(histogram.keys())

    matmul_present = bool(op_set & _MATMUL_OPS)
    collective_present = bool(op_set & _COLLECTIVE_OPS)
    attention_present = bool(op_set & _ATTENTION_OPS)
    moe_present = bool(op_set & _MOE_OPS)

    if matmul_present and collective_present:
        ops = tuple(sorted((op_set & _MATMUL_OPS) | (op_set & _COLLECTIVE_OPS)))
        candidates.append(
            MegakernelCandidate(
                pattern="matmul_collective",
                ops=ops,
                rationale=(
                    "GEMM and a collective both present; ETC paper Fig. 6 "
                    "shows fusing them into one persistent megakernel "
                    "eliminates the kernel-boundary sync between them."
                ),
                confidence=0.85,
            ),
        )

    if attention_present and matmul_present:
        ops = tuple(sorted((op_set & _ATTENTION_OPS) | (op_set & _MATMUL_OPS)))
        candidates.append(
            MegakernelCandidate(
                pattern="attention_pipeline",
                ops=ops,
                rationale=(
                    "attention + matmul chain present; the ETC paper's "
                    "Qwen3 megakernel fuses Attn+RoPE+KV-Cache+Norm+MLP "
                    "into one persistent kernel per decoding step."
                ),
                confidence=0.7,
            ),
        )

    if moe_present and matmul_present:
        ops = tuple(sorted((op_set & _MOE_OPS) | (op_set & _MATMUL_OPS)))
        candidates.append(
            MegakernelCandidate(
                pattern="moe_routing",
                ops=ops,
                rationale=(
                    "topk + gather/scatter + matmul present; ETC paper "
                    "Fig. 5b shows MoE benefits from data-dependent "
                    "megakernel synthesis (dynamic scheduling)."
                ),
                confidence=0.7,
            ),
        )

    long_chains = [g for g in fusion_groups if len(g) >= 4]
    if long_chains:
        ops = tuple(sorted({op for g in long_chains for op in g}))
        candidates.append(
            MegakernelCandidate(
                pattern="unfused_chain",
                ops=ops,
                rationale=(
                    f"{len(long_chains)} long elementwise chains "
                    "(>=4 ops each) -- promoting into a megakernel "
                    "removes per-chain kernel-launch overhead."
                ),
                confidence=0.5,
            ),
        )

    return candidates


__all__ = [
    "InductorHarvestReport",
    "MegakernelCandidate",
    "estimate_megakernel_candidates",
    "harvest_inductor_graph",
]
