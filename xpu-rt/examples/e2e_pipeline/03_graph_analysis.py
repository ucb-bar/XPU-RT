"""Phase 3: graph analysis for heterogeneous systems.

For `--workload` and an optional `--target` (one of our TargetResource
slugs, defaults to openq_5165rb as the primary robotics target), this
script:
  1. Re-captures the workload and runs
     `compgen.agent.analyzer.NetworkAnalyzer().analyze(ep, profile, ...)`.
  2. Serializes the full NetworkAnalysis (clusters, data flow, dossier)
     to `gap_analysis.json`.
  3. Builds `region_inventory.json` conforming to
     schemas/region_analysis.schema.yaml. Five of the eight annotations
     are heuristic (semantic_kind/shape_regime/etc. rules are documented
     inline in _classify_region below) so a future LLM proposal step can
     refine them.
  4. Writes `region_graph.dot` (Graphviz) and a short `region_report.md`
     as secondary artifacts.

Artifacts under
    user_perspective/artifacts/<workload>/stage_3_analysis/<target>/

This is the first "global system reasoning" stage — not for humans first,
machine-readable first.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(ROOT.parent))
sys.path.insert(0, str(ROOT))

from user_perspective.models import smolvla_slice, gemma_decode_slice   # noqa: E402
from user_perspective.scripts._helpers.schema import validate_or_raise  # noqa: E402
from compgen.capture import capture_frontend_artifact                    # noqa: E402
from compgen.agent.analyzer import NetworkAnalyzer                       # noqa: E402
import compgen                                                            # noqa: E402

log = logging.getLogger("phase3")

WORKLOADS = {
    "smolvla_slice": lambda: smolvla_slice.load("auto"),
    "gemma_decode_slice": lambda: gemma_decode_slice.load(),
}

HARDWARE_SPEC_PATH = {
    "bananapi_f3": ROOT / "configs" / "hardware_specs" / "bananapi_f3.yaml",
    "openq_5165rb": ROOT / "configs" / "hardware_specs" / "openq_5165rb.yaml",
    "openq_8250cs": ROOT / "configs" / "hardware_specs" / "openq_8250cs.yaml",
}


SEMANTIC_KIND_MAP = {
    "matmul": "matmul",
    "linear": "matmul",
    "mm": "matmul",
    "bmm": "matmul",
    "addmm": "matmul",
    "conv": "conv",
    "conv2d": "conv",
    "conv1d": "conv",
    "softmax": "reduction",
    "layernorm": "normalization",
    "rmsnorm": "normalization",
    "layer_norm": "normalization",
    "rms_norm": "normalization",
    "batchnorm": "normalization",
    "gelu": "elementwise",
    "relu": "elementwise",
    "silu": "elementwise",
    "sigmoid": "elementwise",
    "tanh": "elementwise",
    "add": "elementwise",
    "mul": "elementwise",
    "sub": "elementwise",
    "div": "elementwise",
    "reduce_sum": "reduction",
    "reduce_mean": "reduction",
    "attention": "attention",
    "sdpa": "attention",
    "embedding": "other",
    "transpose": "layout_op",
    "view": "layout_op",
    "reshape": "layout_op",
    "permute": "layout_op",
    "cat": "layout_op",
    "concat": "layout_op",
    "copy": "copy",
    "clone": "copy",
    "index_put": "cache_update",
    "index_copy": "cache_update",
    "scaled_dot_product_attention": "attention",
}


HEURISTIC_FIELDS = [
    "data_movement",
    "fusion_opportunity",
    "kernel_coverage_potential",
    "placement_affinity",
    "sync_sensitivity",
    "accuracy_sensitivity",
]


def _classify_semantic_kind(kind: str) -> str:
    k = kind.lower()
    for needle, semantic in SEMANTIC_KIND_MAP.items():
        if needle in k:
            return semantic
    return "other"


def _classify_shape_regime(dossier_region: Any) -> str:
    return "highly_dynamic" if dossier_region.dynamic_shapes else "static"


def _classify_data_movement(ai: float) -> str:
    # Arithmetic intensity heuristic:
    #   high AI (> 8 flop/byte) → reuse_heavy
    #   low AI (< 0.5)         → read_heavy or write_heavy (can't distinguish without writes)
    #   otherwise              → neutral
    if ai >= 8.0:
        return "reuse_heavy"
    if ai < 0.5:
        return "read_heavy"
    return "neutral"


def _classify_fusion_opportunity(semantic_kind: str) -> str:
    if semantic_kind in ("elementwise", "layout_op", "copy"):
        return "obviously_fusible"
    if semantic_kind in ("reduction", "normalization"):
        return "maybe_fusible"
    return "keep_separate"


def _classify_kernel_coverage(semantic_kind: str) -> str:
    if semantic_kind in ("matmul", "conv", "attention"):
        return "single_family"
    if semantic_kind in ("normalization", "reduction", "elementwise", "cache_update"):
        return "single_family"
    if semantic_kind in ("layout_op", "copy"):
        return "single_family"
    return "library_fallback"


def _placement_affinity(semantic_kind: str, target_devices: list[dict[str, Any]]) -> list[str]:
    # Crude affinity based on semantic kind and which device types are present.
    kinds = {d["device_type"] for d in target_devices}
    out: list[str] = []
    if semantic_kind in ("matmul", "attention", "conv"):
        if "npu" in kinds: out.append("npu")
        if "gpu" in kinds: out.append("gpu")
        if "dsp" in kinds: out.append("dsp")
        out.append("cpu")
    elif semantic_kind in ("normalization", "reduction"):
        if "dsp" in kinds: out.append("dsp")
        if "gpu" in kinds: out.append("gpu")
        if "npu" in kinds: out.append("npu")
        out.append("cpu")
    elif semantic_kind == "elementwise":
        if "gpu" in kinds: out.append("gpu")
        if "dsp" in kinds: out.append("dsp")
        out.append("cpu")
    elif semantic_kind in ("layout_op", "copy"):
        out.append("cpu")
        if "dsp" in kinds: out.append("dsp")
    elif semantic_kind == "cache_update":
        out.append("cpu")
    else:
        out.append("cpu")
    # De-duplicate, preserve order
    seen = set()
    out_unique = []
    for v in out:
        if v not in seen:
            seen.add(v)
            out_unique.append(v)
    return out_unique


def _sync_sensitivity(region: Any) -> str:
    if region.producers and region.consumers:
        return "producer_consumer_chain"
    if not region.producers and region.consumers:
        return "fan_out"
    if region.producers and not region.consumers:
        return "fan_in"
    return "independent"


def _accuracy_sensitivity(semantic_kind: str) -> str:
    if semantic_kind in ("elementwise", "layout_op", "copy"):
        return "quantization_safe"
    if semantic_kind == "cache_update":
        return "quantization_safe"
    if semantic_kind in ("normalization", "reduction"):
        return "quantization_sensitive"
    if semantic_kind in ("matmul", "conv"):
        return "quantization_sensitive"
    if semantic_kind == "attention":
        return "precision_critical"
    return "quantization_sensitive"


def _region_to_schema(region: Any, target_devices: list[dict[str, Any]]) -> dict[str, Any]:
    semantic_kind = _classify_semantic_kind(region.kind)
    return {
        "region_id": region.region_id,
        "ops": list(region.node_names),
        "semantic_kind": semantic_kind,
        "shape_regime": _classify_shape_regime(region),
        "data_movement": _classify_data_movement(region.arithmetic_intensity),
        "fusion_opportunity": _classify_fusion_opportunity(semantic_kind),
        "kernel_coverage_potential": _classify_kernel_coverage(semantic_kind),
        "placement_affinity": _placement_affinity(semantic_kind, target_devices),
        "sync_sensitivity": _sync_sensitivity(region),
        "accuracy_sensitivity": _accuracy_sensitivity(semantic_kind),
        "estimated_flops": int(region.flops),
        "estimated_bytes": int(region.bytes),
        "rationale": f"kind={region.kind!r} ai={region.arithmetic_intensity:.2f} "
                     f"repeats={region.repeated_count}",
        "provenance": "heuristic",
    }


def _serialize(obj: Any) -> Any:
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if is_dataclass(obj):
        return {k: _serialize(v) for k, v in asdict(obj).items()}
    if isinstance(obj, (list, tuple)):
        return [_serialize(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _serialize(v) for k, v in obj.items()}
    if isinstance(obj, set):
        return sorted(_serialize(x) for x in obj)
    return repr(obj)


def _summarize_devices(dev: compgen.CompGenDevice) -> list[dict[str, Any]]:
    return [
        {"device_type": d.device_type, "name": d.name, "vendor": d.vendor}
        for d in dev.profile.devices
    ]


def run(workload: str, target_slug: str) -> int:
    if workload not in WORKLOADS:
        raise SystemExit(f"unknown workload {workload!r}")
    if target_slug not in HARDWARE_SPEC_PATH:
        raise SystemExit(f"unknown target {target_slug!r}")

    out_dir = ROOT / "artifacts" / workload / "stage_3_analysis" / target_slug
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- load target ---
    log.info("loading target=%s", target_slug)
    dev = compgen.device(HARDWARE_SPEC_PATH[target_slug],
                          output_dir=out_dir / "_targetgen")
    target_device_summary = _summarize_devices(dev)
    # We use the HardwareSpec-derived TargetProfile for NetworkAnalyzer.analyze
    # so its arithmetic/memory heuristics align with what compgen.device models.
    profile = dev.profile

    # --- load workload ---
    log.info("loading workload=%s", workload)
    bundle = WORKLOADS[workload]()
    if bundle.capture_mode != "torch_export":
        raise SystemExit(
            f"capture_mode={bundle.capture_mode!r} — Phase 3 requires torch.export; "
            "rerun with a workload that uses torch_export."
        )

    # Re-capture for determinism. This is cheap for both workloads.
    log.info("capture for analysis")
    artifact = capture_frontend_artifact(bundle.model, tuple(bundle.sample_inputs))
    ep = artifact.exported_program
    assert ep is not None

    # --- analyze ---
    log.info("NetworkAnalyzer.analyze(...)")
    analysis = NetworkAnalyzer().analyze(
        ep, profile, model_name=workload
    )
    dossier = analysis.dossier
    assert dossier is not None, "NetworkAnalysis.dossier is None — unexpected"

    # --- gap_analysis.json ---
    gap_payload = {
        "workload": workload,
        "target": target_slug,
        "model_name": analysis.model_name,
        "total_params": analysis.total_params,
        "total_flops": analysis.total_flops,
        "total_bytes": analysis.total_bytes,
        "clusters": [_serialize(c) for c in analysis.clusters],
        "unclustered_ops": list(analysis.unclustered_ops),
        "data_flow": [_serialize(e) for e in analysis.data_flow],
        "bottleneck_clusters": list(analysis.bottleneck_clusters),
        "optimization_opportunities": list(analysis.optimization_opportunities),
        "dossier": {
            "total_regions": dossier.total_regions,
            "total_flops": dossier.total_flops,
            "total_bytes": dossier.total_bytes,
            "op_histogram": dict(dossier.op_histogram),
            "repeated_patterns": dict(dossier.repeated_patterns),
            "critical_path": list(dossier.critical_path),
            "independent_region_sets": [list(s) for s in dossier.independent_region_sets],
            "dynamic_shape_regions": list(dossier.dynamic_shape_regions),
            "unsupported_targets": list(dossier.unsupported_targets),
            "region_count": len(dossier.regions),
        },
    }
    (out_dir / "gap_analysis.json").write_text(
        json.dumps(gap_payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    log.info("gap_analysis: clusters=%d regions=%d bottlenecks=%d",
             len(analysis.clusters), dossier.total_regions, len(analysis.bottleneck_clusters))

    # --- region_inventory.json ---
    schema_regions = [_region_to_schema(r, target_device_summary) for r in dossier.regions]
    semantic_counts: Counter[str] = Counter(r["semantic_kind"] for r in schema_regions)
    inventory_payload = {
        "workload": workload,
        "target": target_slug,
        "schema_version": "1.0",
        "regions": schema_regions,
        "summary": {
            "total_regions": len(schema_regions),
            "regions_by_kind": dict(semantic_counts),
            "bottleneck_region_ids": list(analysis.bottleneck_clusters),
            "heuristic_fields": HEURISTIC_FIELDS,
        },
    }
    validate_or_raise(inventory_payload, "region_analysis")
    (out_dir / "region_inventory.json").write_text(
        json.dumps(inventory_payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    log.info("region_inventory: %d regions (%s)",
             len(schema_regions), dict(semantic_counts))

    # --- region_graph.dot ---
    dot_lines = ["digraph regions {", "  rankdir=LR;", "  node [shape=box, fontsize=10];"]
    region_by_id = {r.region_id: r for r in dossier.regions}
    for r in dossier.regions:
        label = f"{r.region_id}\\n{r.kind}\\nf={r.flops} b={r.bytes}"
        dot_lines.append(f'  "{r.region_id}" [label="{label}"];')
    edge_set: set[tuple[str, str]] = set()
    for r in dossier.regions:
        for consumer in r.consumers:
            if consumer in region_by_id and (r.region_id, consumer) not in edge_set:
                edge_set.add((r.region_id, consumer))
                dot_lines.append(f'  "{r.region_id}" -> "{consumer}";')
    dot_lines.append("}")
    (out_dir / "region_graph.dot").write_text("\n".join(dot_lines), encoding="utf-8")

    # --- region_report.md (secondary) ---
    lines = [
        f"# Region report — {workload} @ {target_slug}",
        "",
        f"- total regions: **{len(schema_regions)}**",
        f"- total flops (dossier): {dossier.total_flops:,}",
        f"- total bytes (dossier): {dossier.total_bytes:,}",
        f"- critical path length: {len(dossier.critical_path)}",
        f"- bottleneck clusters: {len(analysis.bottleneck_clusters)}",
        "",
        "## By semantic kind",
        "",
    ]
    for kind, cnt in sorted(semantic_counts.items(), key=lambda kv: -kv[1]):
        lines.append(f"- `{kind}`: {cnt}")
    lines += ["", "## Optimization opportunities (from NetworkAnalyzer)", ""]
    if analysis.optimization_opportunities:
        for opp in analysis.optimization_opportunities:
            lines.append(f"- {opp}")
    else:
        lines.append("_(none reported)_")
    lines += ["", "## Heuristic fields", "",
              "These annotations are derived by user-side heuristics in "
              "`03_graph_analysis.py`; they are the candidates for future "
              "LLM-based refinement:",
              ""]
    for f in HEURISTIC_FIELDS:
        lines.append(f"- `{f}`")
    (out_dir / "region_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"\nPhase 3 complete for {workload} @ {target_slug}")
    print(f"  regions: {len(schema_regions)}  kinds: {dict(semantic_counts)}")
    print(f"  critical path: {len(dossier.critical_path)} regions  "
          f"bottlenecks: {len(analysis.bottleneck_clusters)}")
    print(f"  artifacts under {out_dir.relative_to(ROOT)}/")
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--workload", required=True, choices=sorted(WORKLOADS.keys()))
    p.add_argument("--target", default="openq_5165rb",
                   choices=sorted(HARDWARE_SPEC_PATH.keys()))
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    return run(args.workload, args.target)


if __name__ == "__main__":
    raise SystemExit(main())
