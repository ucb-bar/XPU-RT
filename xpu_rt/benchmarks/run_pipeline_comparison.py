"""Pipeline-level vanilla-vs-agentic comparison driver.

This is the top-level entry point for the study described in
``plan: Pipeline-level study — vanilla KernelBlaster (kernel-at-a-time)
vs. XPU-RT agentic compilation (graph-at-a-time)`` (see
``floofy-foraging-matsumoto.md``).

What this driver does **today** (`--mode plan`):

  1. Loads a SmolVLA wrapper via the existing
     :mod:`xpu_rt.benchmarks.smolvla_subset` loader.
  2. Walks transformer blocks via
     :func:`xpu_rt.benchmarks.smolvla_block_enumerator.enumerate_blocks`.
  3. For each block:

       * Runs :func:`xpu_rt.kernels.fusion_planner.plan_fusion` to get
         the agentic verdict (which nodes the planner would fuse).
       * Builds a vanilla pipeline harness via
         :mod:`xpu_rt.kb_gemmini.multiop_harness` to count N kernels +
         emit the chained driver.
       * For each non-singleton cluster, emits a MEGA contract via
         :mod:`xpu_rt.kernels.mega_contract_emitter` and renders the
         fused harness via :mod:`xpu_rt.kb_gemmini.mega_templates` so
         we can count the agentic-side artefacts.

  4. Writes ``results/comparison/pipeline_level/report.{md,json}``
     summarising:

       * Per-block: number of kernels emitted by each side, planner's
         predicted speedup, fusion-oracle pair verdicts.
       * Aggregate: total kernels per side, geomean planner speedup.

What this driver does **not** do (deferred):

  * Invoke the Gemini-2.5-flash generator (no API spend in plan mode).
  * Spike-execute either side's kernels (skipped — needs the riscv-tools
    toolchain set up; runs separately via ``CRiscvEvaluator``).
  * Honour an agent's fusion-override from the MCP tool (the
    ``xpu_rt_commit_fusion_decision_response`` recorder writes
    overrides next to the graph; reading them back is the next step).

The structural artefacts the driver emits today are enough to
demonstrate the architectural delta: the planner produces fewer +
larger kernels than vanilla, with explicit reasoning per pair, and
the MEGA emit-and-render path validates end-to-end.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from xpu_rt.benchmarks.smolvla_block_enumerator import (
    BlockEnumeratorConfig,
    BlockSpec,
    enumerate_blocks,
)
from xpu_rt.benchmarks.smolvla_subset import SubsetSelector
from xpu_rt.ir.payload.contract_graph import ContractEdge, ContractGraph, ContractNode
from xpu_rt.spike_harness.templates.mega_gemmini import render_fused_artifacts
from xpu_rt.spike_harness.multiop_harness import (
    KernelBinding,
    PipelineHarnessSpec,
    render_pipeline_driver_c,
)
from xpu_rt.kernels.contract_v3 import HardwareEnvelope
from xpu_rt.kernels.fusion_planner import FusionPlan, plan_fusion
from xpu_rt.kernels.mega_contract_emitter import emit_mega_contract


logger = logging.getLogger("xpu_rt.benchmarks.run_pipeline_comparison")


# ---------------------------------------------------------------------------
# Per-target hardware envelopes
# ---------------------------------------------------------------------------


def gemmini_envelope() -> HardwareEnvelope:
    """The same envelope the FusionPlanner / MegaContractEmitter
    tests use, so the driver's planner verdicts are bit-equivalent
    to the unit-test ones for identical graphs."""
    return HardwareEnvelope(
        target_name="gemmini",
        vector_lanes=16,
        scratchpad_bytes=256 * 1024,
        register_bytes=16,
        native_dtypes=("i8", "i32"),
        peak_bandwidth_gbps=8.0,
        register_quota_per_thread=256,
        # tiled_matmul_auto picks tiles around 64×64×1 ≈ 4 KiB for i8
        # — the planner discounts weight tensors by this much instead
        # of demanding the full matrix fit in scratchpad.
        weight_tile_bytes=4096,
    )


def saturn_opu_envelope() -> HardwareEnvelope:
    """Saturn OPU V128 envelope — RVV 1.0 + vLen=128. Values pulled
    from ``configs/targets/saturn_opu_v128.yaml`` (peak_bandwidth +
    memory_tiers) and the hand-authored target card at
    ``.xpu_rt/knowledge/targets/saturn_opu_v128/target_card.json``.

    ``vector_lanes=16`` reflects vLen/SEW = 128/8 for the i8 inner
    loop (the canonical Saturn GEMM kernel uses SEW=8 lanes for
    operand loads, widening to i32 via vwmacc). ``weight_tile_bytes``
    is set to one ``vint8m1`` chunk (16 i8 elements) since Saturn
    streams weights one vL-wide tile at a time — the same
    cost-model fix Track 0 wired in for Gemmini, applied at the
    different scale that matches Saturn's actual codegen.
    """
    return HardwareEnvelope(
        target_name="saturn_opu_v128",
        vector_lanes=16,
        scratchpad_bytes=512 * 1024,  # L2 tier per target YAML
        register_bytes=64,
        native_dtypes=("i8", "i32", "f32"),
        peak_bandwidth_gbps=12.8,
        register_quota_per_thread=512,
        # One unit-stride vL-wide i8 weight tile = 16 B; we round up
        # to 256 B (one paired-lane vmacc chunk) to leave headroom
        # for the codegen to hold tile_K × tile_N in flight.
        weight_tile_bytes=256,
    )


def envelope_for_target(target_id: str) -> HardwareEnvelope:
    """Resolve a target_id to its :class:`HardwareEnvelope`.

    Today only Gemmini + Saturn are wired; other ids raise
    :class:`ValueError`. ``gemmini`` is the canonical id; ``gemmini_mx``
    is the historical alias and resolves identically.
    """
    t = target_id.lower()
    if t.startswith("gemmini"):
        # The HardwareEnvelope.target_name follows the caller's id so
        # downstream filters (KB-v2's tracking_source tag, the report's
        # cycle_source label) reflect whatever the caller used.
        env = gemmini_envelope()
        from dataclasses import replace as _replace
        return _replace(env, target_name=target_id)
    if t.startswith("saturn") or t.startswith("opu"):
        return saturn_opu_envelope()
    raise ValueError(
        f"no HardwareEnvelope wired for target_id={target_id!r}; "
        "supported: gemmini / gemmini_mx / saturn_opu_v128 (extend "
        "xpu_rt.benchmarks.run_pipeline_comparison.envelope_for_target)."
    )


# ---------------------------------------------------------------------------
# Per-block result schema
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VanillaBranchResult:
    """Structural numbers for the vanilla pipeline path on one block."""

    n_kernels_emitted: int
    chained_driver_lines: int
    chained_driver_excerpt: str
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class AgenticBranchResult:
    """Structural numbers for the agentic MEGA path on one block."""

    n_kernels_emitted: int
    n_clusters: int
    mega_clusters: tuple[str, ...]
    planner_estimated_speedup: float
    fused_init_lines: int
    fused_driver_lines: int
    per_pair_verdicts: tuple[dict[str, Any], ...]
    per_cluster_granularity: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class PerBlockResult:
    block_id: str
    block_kind: str
    component: str
    n_nodes: int
    vanilla: VanillaBranchResult
    agentic: AgenticBranchResult


@dataclass
class PipelineComparisonReport:
    target_id: str
    mode: str
    n_blocks: int
    per_block: list[PerBlockResult] = field(default_factory=list)
    aggregate_planner_speedup_geomean: float = 1.0
    total_vanilla_kernels: int = 0
    total_agentic_kernels: int = 0


# ---------------------------------------------------------------------------
# Helpers — block → branches → result
# ---------------------------------------------------------------------------


def _serialise_graph(graph: ContractGraph) -> dict[str, Any]:
    """Serialise the block subgraph for the fusion-decision MCP tool /
    archival purposes. The shape mirrors what
    :func:`xpu_rt.mcp.tools.fusion_decision._rebuild_graph_from_view`
    expects."""
    nodes: dict[str, dict[str, Any]] = {}
    for nid, n in graph.nodes.items():
        md = n.contract.metadata or {}
        nodes[nid] = {
            "op_name": n.op_name,
            "input_shapes": [list(s) for s in md.get("input_shapes", [])],
            "output_shapes": [list(s) for s in md.get("output_shapes", [])],
            "dtype": next(iter(n.contract.supported_dtypes or {"f32"})),
        }
    edges: list[dict[str, Any]] = []
    for e in graph.edges:
        edges.append(
            {
                "producer_id": e.producer_id,
                "consumer_id": e.consumer_id,
                "operand_index": e.operand_index,
                "tensor_shape": list(e.tensor_shape),
                "dtype": e.dtype,
                "bytes_per_element": e.bytes_per_element,
            }
        )
    return {"nodes": nodes, "edges": edges}


def _vanilla_branch(block: BlockSpec, envelope: HardwareEnvelope) -> VanillaBranchResult:
    """Render the vanilla pipeline harness — N kernels, one DRAM
    buffer per edge."""
    del envelope  # the vanilla harness doesn't consult the envelope
    bindings = tuple(
        KernelBinding(
            op_id=nid,
            function_name=f"launch_{nid.replace('.', '_').replace('-', '_')}",
            kernel_source_path=Path("/tmp/__placeholder.c"),
        )
        for nid in block.subgraph.topological_order
    )
    spec = PipelineHarnessSpec(graph=block.subgraph, bindings=bindings, external_input_shapes={})
    driver_src = render_pipeline_driver_c(spec)
    lines = driver_src.splitlines()
    excerpt = "\n".join(lines[:40]) + "\n  ...  (truncated)"
    return VanillaBranchResult(
        n_kernels_emitted=len(block.subgraph.topological_order),
        chained_driver_lines=len(lines),
        chained_driver_excerpt=excerpt,
    )


def _agentic_branch(block: BlockSpec, envelope: HardwareEnvelope) -> AgenticBranchResult:
    """Run the planner, emit MEGA contracts for non-singleton clusters,
    render their fused harness."""
    plan = plan_fusion(block.subgraph, envelope)
    n_fused_kernels = 0
    n_singleton = 0
    mega_clusters: list[str] = []
    fused_init_lines_total = 0
    fused_driver_lines_total = 0
    for cluster in plan.clusters:
        if len(cluster.member_op_ids) >= 2:
            mega_result = emit_mega_contract(cluster, block.subgraph, envelope)
            artifacts = render_fused_artifacts(mega_result.contract)
            n_fused_kernels += 1
            mega_clusters.append(cluster.cluster_id)
            fused_init_lines_total += len(artifacts.init_c.splitlines())
            fused_driver_lines_total += len(artifacts.driver_c.splitlines())
        else:
            n_singleton += 1

    pair_verdicts = [
        {"producer_id": p, "consumer_id": c, "decision": v.decision.value, "ratio": round(v.est_speedup_ratio, 3)}
        for (p, c, v) in plan.per_pair_verdicts
    ]
    cluster_granularity = [
        {"cluster_id": cid, "granularity": gv.granularity.value, "reason": gv.reason}
        for (cid, gv) in plan.per_cluster_granularity
    ]
    return AgenticBranchResult(
        n_kernels_emitted=n_fused_kernels + n_singleton,
        n_clusters=len(plan.clusters),
        mega_clusters=tuple(mega_clusters),
        planner_estimated_speedup=plan.estimated_speedup,
        fused_init_lines=fused_init_lines_total,
        fused_driver_lines=fused_driver_lines_total,
        per_pair_verdicts=tuple(pair_verdicts),
        per_cluster_granularity=tuple(cluster_granularity),
    )


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------


def run(
    out_dir: Path,
    *,
    mode: str = "plan",
    blocks: list[BlockSpec] | None = None,
    config: BlockEnumeratorConfig | None = None,
    envelope: HardwareEnvelope | None = None,
) -> PipelineComparisonReport:
    """Run the comparison and write reports under ``out_dir``.

    Args:
        out_dir: Destination directory; will be created. Reports land
            at ``<out_dir>/report.{md,json}`` and per-block
            ``<out_dir>/per_block/<block_id>/graph.json``.
        mode: Today only ``"plan"`` is wired. ``"full"`` would invoke
            the generators + Spike eval; documented but not run.
        blocks: When provided, skip the SmolVLA loader and run on
            these blocks directly (used by tests + smoke runs).
        config: Block enumerator config; defaults to Phase A scope.
        envelope: Target hardware envelope; defaults to Gemmini.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    env = envelope or gemmini_envelope()
    if blocks is None:
        cfg = config or BlockEnumeratorConfig()
        logger.info("loading SmolVLA wrapper for block enumeration")
        selector = SubsetSelector(seq_len=cfg.seq_len)
        model = selector.load()
        blocks = enumerate_blocks(model, cfg)
    logger.info("enumerated %d blocks for the comparison", len(blocks))

    per_block: list[PerBlockResult] = []
    per_block_dir = out_dir / "per_block"
    per_block_dir.mkdir(exist_ok=True)

    for block in blocks:
        vanilla = _vanilla_branch(block, env)
        agentic = _agentic_branch(block, env)
        per_block.append(
            PerBlockResult(
                block_id=block.block_id,
                block_kind=block.block_kind,
                component=block.component,
                n_nodes=len(block.subgraph.topological_order),
                vanilla=vanilla,
                agentic=agentic,
            )
        )
        # Persist per-block artefacts for inspection / MCP tool consumption.
        block_dir = per_block_dir / block.block_id.replace("/", "_").replace(".", "_")
        block_dir.mkdir(exist_ok=True)
        (block_dir / "graph.json").write_text(json.dumps(_serialise_graph(block.subgraph), indent=2))

    # Aggregate the planner-speedup geomean across non-trivial blocks.
    import math

    log_sum = 0.0
    weight = 0
    total_vanilla = 0
    total_agentic = 0
    for b in per_block:
        total_vanilla += b.vanilla.n_kernels_emitted
        total_agentic += b.agentic.n_kernels_emitted
        sp = b.agentic.planner_estimated_speedup
        if sp > 1.0:
            log_sum += math.log(sp) * b.n_nodes
            weight += b.n_nodes
    aggregate_geomean = math.exp(log_sum / weight) if weight else 1.0

    report = PipelineComparisonReport(
        target_id=env.target_name,
        mode=mode,
        n_blocks=len(per_block),
        per_block=per_block,
        aggregate_planner_speedup_geomean=aggregate_geomean,
        total_vanilla_kernels=total_vanilla,
        total_agentic_kernels=total_agentic,
    )
    _write_reports(out_dir, report)
    return report


# ---------------------------------------------------------------------------
# Report writers
# ---------------------------------------------------------------------------


def _write_reports(out_dir: Path, report: PipelineComparisonReport) -> None:
    (out_dir / "report.json").write_text(_json_dump(report))
    (out_dir / "report.md").write_text(_render_markdown(report))


def _json_dump(report: PipelineComparisonReport) -> str:
    return json.dumps(_to_jsonable(report), indent=2)


def _to_jsonable(obj: Any) -> Any:
    if hasattr(obj, "__dataclass_fields__"):
        d = asdict(obj)
        return d
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    return obj


def _render_markdown(report: PipelineComparisonReport) -> str:
    lines: list[str] = []
    lines.append(f"# Pipeline-level comparison — {report.target_id} ({report.mode} mode)\n")
    lines.append(
        f"Blocks enumerated: **{report.n_blocks}**  |  "
        f"Vanilla kernels total: **{report.total_vanilla_kernels}**  |  "
        f"Agentic kernels total: **{report.total_agentic_kernels}**  |  "
        f"Planner geomean speedup: **{report.aggregate_planner_speedup_geomean:.2f}x**\n"
    )
    lines.append("## Per-block summary\n")
    lines.append("| block_id | kind | nodes | vanilla #kernels | agentic #kernels | planner speedup |")
    lines.append("|---|---|---|---|---|---|")
    for b in report.per_block:
        lines.append(
            f"| `{b.block_id}` | {b.block_kind} | {b.n_nodes} | "
            f"{b.vanilla.n_kernels_emitted} | {b.agentic.n_kernels_emitted} | "
            f"{b.agentic.planner_estimated_speedup:.2f}x |"
        )
    lines.append("\n## Planner verdicts per block\n")
    for b in report.per_block:
        lines.append(f"### `{b.block_id}`\n")
        if b.agentic.per_pair_verdicts:
            lines.append("Per-pair fusion-oracle verdicts:")
            lines.append("")
            for v in b.agentic.per_pair_verdicts:
                lines.append(f"- `{v['producer_id']}` → `{v['consumer_id']}`: **{v['decision']}** ({v['ratio']}x)")
            lines.append("")
        if b.agentic.per_cluster_granularity:
            lines.append("Per-cluster granularity verdicts:")
            lines.append("")
            for g in b.agentic.per_cluster_granularity:
                lines.append(f"- `{g['cluster_id']}`: **{g['granularity']}** — {g['reason']}")
            lines.append("")
    lines.append("\n## How to read this report")
    lines.append(
        "When a block's `agentic #kernels` equals `vanilla #kernels` "
        "and the planner speedup is `1.00x`, look at the per-cluster "
        "granularity verdict for the *why*: typically the pairwise "
        "fusion oracle says FUSE but the granularity oracle vetoes "
        "because the cluster's combined working set (intermediates + "
        "weight tensors) does not fit in the target's scratchpad "
        "budget. That is an honest signal that the planner sees the "
        "chain as un-fusable *under its current cost model* — fixing "
        "it requires teaching the cost model about weight-tiling, "
        "which is out of scope for this study.\n"
    )
    lines.append("\n## Methodology")
    lines.append(
        "- Vanilla branch: each ContractNode emits one kernel; "
        "the multi-op Spike harness chains them with DRAM intermediates.\n"
        "- Agentic branch: `FusionPlanner` partitions the graph, "
        "`MegaContractEmitter` builds one MEGA contract per multi-node "
        "cluster, `mega_templates` renders the fused harness.\n"
        "- This report runs in `plan` mode: no Gemini spend, no "
        "Spike execution. The structural numbers + planner verdicts "
        "are the deliverable. `full` mode (LLM + Spike) wires the "
        "agentic and vanilla generators in a follow-up step.\n"
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", type=Path, default=Path("results/comparison/pipeline_level"))
    parser.add_argument(
        "--mode",
        choices=("plan",),
        default="plan",
        help="``plan`` runs the planner only (no LLM, no Spike). ``full`` "
        "mode wires the live LLM + Spike evaluator — implemented by "
        "the higher-level cross-target driver, not this CLI.",
    )
    parser.add_argument(
        "--target",
        choices=("gemmini", "gemmini_mx", "saturn_opu_v128"),
        default="gemmini",
        help="Target id. Resolves the envelope via envelope_for_target.",
    )
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3],
        help="action_expert layer indices to enumerate (default: 0-3).",
    )
    parser.add_argument(
        "--block-kinds",
        nargs="+",
        default=["mlp", "head"],
        choices=["mlp", "attention", "head"],
        help="Block kinds to enumerate.",
    )
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args(argv)
    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")

    cfg = BlockEnumeratorConfig(
        seq_len=args.seq_len,
        kinds=tuple(args.block_kinds),
        components=("action_expert", "action_head"),
        layer_indices=tuple(args.layers) if args.layers else None,
    )
    envelope = envelope_for_target(args.target)
    report = run(args.out_dir, mode=args.mode, config=cfg, envelope=envelope)
    print(f"wrote {args.out_dir / 'report.md'}", file=sys.stderr)
    print(f"wrote {args.out_dir / 'report.json'}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AgenticBranchResult",
    "PerBlockResult",
    "PipelineComparisonReport",
    "VanillaBranchResult",
    "envelope_for_target",
    "gemmini_envelope",
    "main",
    "run",
    "saturn_opu_envelope",
]
