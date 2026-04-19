"""Phase 5 probe — dry-run of a target-feature-driven fusion plan.

Applies a hard-coded *canonical* fusion plan matching the
post-optimization kernel-family shortlist from analysis/workload_mapping.md
to the Phase 4 baseline. Emits a recipe.fusion_plan_op-shaped artifact
(validates against prototypes/schemas/recipe_semantic_global.schema.yaml
as a single `decision[]` entry) and the projected post-plan family
counts that script 06 consumes.

The plan is not the LLM's output; it is a deterministic demonstration
of what `propose_fusion` + `raise_special_ops` + `match_library_call`
*should* converge to for the given workload/target pair. It encodes:

  - per-region → target kernel family assignment, drawn from the target
    resource model's `supported_kernel_families`;
  - a `target_feature_justification` linking each assignment to a
    field in the target YAML;
  - a projected library vs autocomp attribution per family (feeds 06).

Artifacts:
  artifacts/<workload>/stage_5_fusion_plan/<target>/fusion_plan_dryrun.yaml
  artifacts/<workload>/stage_5_fusion_plan/<target>/post_plan_kernel_families.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(ROOT.parent))

from user_perspective.scripts._helpers.proto_schema import validate_or_raise_proto  # noqa: E402


# ---------------------------------------------------------------------------
# Canonical assignment maps per workload.
#
# Maps region semantic_kind (from region_inventory.json) to a target
# kernel-family (from target_resource.v2's supported_kernel_families).
# The map is workload-aware because smolVLA and Gemma-decode rely on
# different family subsets.
# ---------------------------------------------------------------------------

ASSIGNMENT_MAPS: dict[str, dict[str, str]] = {
    "smolvla_slice": {
        "matmul": "gemm_int8",
        "conv": "patch_embed_conv_as_gemm",
        "normalization": "rmsnorm_fused",
        "elementwise": "elementwise",
        "layout_op": "elementwise",         # zero-copy elementwise absorb
        "reduction": "softmax_fused",
        "other": "elementwise",
        "copy": "elementwise",
        "attention": "attention_decode_single_query",
    },
    "gemma_decode_slice": {
        "matmul": "gemm_int8",
        "normalization": "rmsnorm_qkv_epilogue",   # fused via propose_fusion
        "elementwise": "swiglu_fused",
        "layout_op": "elementwise",
        "reduction": "softmax_fused",
        "other": "elementwise",
        "copy": "elementwise",
        "attention": "attention_decode_single_query",
        "cache_update": "kv_update",
    },
}


JUSTIFICATION_MAP = {
    "gemm_int8":
        "supported_kernel_families[?family=='gemm_int8' && provider=='library']",
    "gemm":
        "supported_kernel_families[?family=='gemm' && provider=='library']",
    "patch_embed_conv_as_gemm":
        "supported_kernel_families[?family=='patch_embed_conv_as_gemm'] + lower_conv_to_img2col TOOL",
    "rmsnorm_fused":
        "supported_kernel_families[?family=='rmsnorm_fused'] raised via raise_special_ops",
    "rmsnorm_qkv_epilogue":
        "propose_fusion INVENT-SLOT: rmsnorm + QKV projection + RoPE epilogue",
    "softmax_fused":
        "supported_kernel_families[?family=='softmax_fused'] raised via raise_special_ops",
    "silu_fused":
        "supported_kernel_families[?family=='silu_fused'] raised via raise_special_ops",
    "swiglu_fused":
        "supported_kernel_families[?family=='swiglu_fused'] via propose_fusion",
    "rope_apply":
        "supported_kernel_families[?family=='rope_apply']",
    "kv_update":
        "supported_kernel_families[?family=='kv_update']",
    "attention_decode_single_query":
        "supported_kernel_families[?family=='attention_decode_single_query']",
    "elementwise":
        "supported_kernel_families[?family=='elementwise']",
    "embedding_lookup":
        "supported_kernel_families[?family=='embedding_lookup']",
}


def _load_target(target: str) -> dict[str, Any]:
    path = ROOT / "configs" / "targets" / f"{target}.v2.yaml"
    if not path.exists():
        raise SystemExit(f"missing v2 target YAML: {path}")
    doc = yaml.safe_load(path.read_text())
    validate_or_raise_proto(doc, "target_resource.v2")
    return doc


def _target_has_family(target_doc: dict[str, Any], family: str) -> bool:
    return any(f["family"] == family for f in target_doc["supported_kernel_families"])


def _family_provider(target_doc: dict[str, Any], family: str) -> str:
    for f in target_doc["supported_kernel_families"]:
        if f["family"] == family:
            return f["provider"]
    return "fallback"


def plan(workload: str, target: str) -> dict[str, Any]:
    baseline_path = (
        ROOT
        / "artifacts"
        / workload
        / "stage_4_kernel_families"
        / target
        / "kernel_families_baseline.yaml"
    )
    if not baseline_path.exists():
        raise SystemExit(
            f"missing {baseline_path}. Run scripts/04_probe_kernel_families.py first."
        )
    baseline = yaml.safe_load(baseline_path.read_text())
    target_doc = _load_target(target)

    assignment = ASSIGNMENT_MAPS.get(workload)
    if assignment is None:
        raise SystemExit(f"no assignment map defined for workload={workload!r}")

    # Assign each baseline family (== one region) to a target family.
    per_family_assignment: list[dict[str, Any]] = []
    target_family_regions: dict[str, list[str]] = defaultdict(list)
    missing_target_support: list[dict[str, str]] = []

    for family in baseline["families"]:
        region_id = family["region_ids"][0]
        kind = family["family_key"]["semantic_kind"]
        target_family = assignment.get(kind, "elementwise")
        if not _target_has_family(target_doc, target_family):
            missing_target_support.append(
                {"region_id": region_id, "semantic_kind": kind,
                 "requested_family": target_family}
            )
            target_family = "elementwise"    # safe fallback
        provider = _family_provider(target_doc, target_family)
        per_family_assignment.append(
            {
                "region_id": region_id,
                "from_semantic_kind": kind,
                "to_target_family": target_family,
                "provider": provider,
                "ops": family["ops"],
                "accuracy_sensitivity": family["accuracy_sensitivity"],
            }
        )
        target_family_regions[target_family].append(region_id)

    # Post-plan families: one per *distinct target family actually used*.
    post_plan_families = []
    for tf, regions in sorted(target_family_regions.items()):
        provider = _family_provider(target_doc, tf)
        post_plan_families.append(
            {
                "target_family": tf,
                "provider": provider,
                "region_count": len(regions),
                "region_ids": regions,
                "target_feature_justification": JUSTIFICATION_MAP.get(
                    tf, f"supported_kernel_families[?family=='{tf}']"
                ),
            }
        )

    # Build a decision-log entry shaped like a recipe.fusion_plan_op.
    fusion_plan_op = {
        "phase": 3,
        "kind": "invent_proposal",
        "op_name": "propose_fusion",
        "select_vs_invent": "invent",
        "target_feature_justification":
            "target_resource.v2.supported_kernel_families + fusion_cost_model",
        "candidates": [
            {
                "name": "one_family_per_region_baseline",
                "post_plan_family_count": len(baseline["families"]),
                "notes": "equivalent to running no fusion pass",
            },
            {
                "name": "coarse_by_semantic_kind",
                "post_plan_family_count": baseline["coarse_family_count"],
                "notes": "naïve same-kind fusion; ignores target features",
            },
            {
                "name": "target_feature_driven_canonical",
                "post_plan_family_count": len(post_plan_families),
                "notes": "mapped via ASSIGNMENT_MAPS[workload] + target supported_kernel_families",
            },
        ],
        "chosen": {
            "name": "target_feature_driven_canonical",
            "post_plan_family_count": len(post_plan_families),
            "assignments": per_family_assignment,
        },
        "gate_result": {
            "status": "accepted" if not missing_target_support else "deferred",
            "details": {
                "baseline_family_count": len(baseline["families"]),
                "post_plan_family_count": len(post_plan_families),
                "reduction_ratio": round(
                    len(baseline["families"]) / max(1, len(post_plan_families)), 2
                ),
                "missing_target_support": missing_target_support,
            },
        },
        "llm_turn_id": "dryrun_canonical",
        "args": {
            "workload": workload,
            "target": target,
            "cost_budget": "stage5_dryrun",
        },
    }

    # Wrap into a RecipeSemanticGlobalContract so it validates.
    recipe = {
        "schema_version": "1.0",
        "workload": workload,
        "target": target,
        "regions": [
            {
                "region_id": f["region_id"],
                "shape_report": {},
                "allowed_kernel_families": [f["to_target_family"]],
                "numerics_policy": {},
                "memory_hints": {},
                "sync_sensitivity": "loose",
            }
            for f in per_family_assignment
        ],
        "decisions": [fusion_plan_op],
        "summary": {
            "total_decisions": 1,
            "decisions_by_phase": {"3": 1},
            "tool_calls": 0,
            "invent_proposals": 1,
            "invent_accepted": 1 if not missing_target_support else 0,
            "invent_rejected": 0,
        },
    }

    return {
        "recipe_fusion_plan_op": recipe,
        "post_plan_families": post_plan_families,
        "baseline_family_count": len(baseline["families"]),
        "post_plan_family_count": len(post_plan_families),
        "missing_target_support": missing_target_support,
    }


def run(workload: str, target: str) -> int:
    result = plan(workload, target)
    out_dir = ROOT / "artifacts" / workload / "stage_5_fusion_plan" / target
    out_dir.mkdir(parents=True, exist_ok=True)

    fusion_path = out_dir / "fusion_plan_dryrun.yaml"
    fusion_path.write_text(
        yaml.safe_dump(result["recipe_fusion_plan_op"], sort_keys=False,
                       default_flow_style=False),
        encoding="utf-8",
    )

    post_plan_path = out_dir / "post_plan_kernel_families.yaml"
    post_plan_path.write_text(
        yaml.safe_dump(
            {
                "workload": workload,
                "target": target,
                "baseline_family_count": result["baseline_family_count"],
                "post_plan_family_count": result["post_plan_family_count"],
                "families": result["post_plan_families"],
                "missing_target_support": result["missing_target_support"],
            },
            sort_keys=False,
            default_flow_style=False,
        ),
        encoding="utf-8",
    )

    print(f"\nFusion plan dry-run — {workload} @ {target}")
    print(f"  baseline families (1-per-region):  {result['baseline_family_count']}")
    print(f"  post-plan families:                {result['post_plan_family_count']}")
    print(
        "  reduction ratio:                   "
        f"{result['baseline_family_count'] / max(1, result['post_plan_family_count']):.2f}×"
    )
    print("  post-plan families by provider:")
    counts = defaultdict(int)
    for f in result["post_plan_families"]:
        counts[f["provider"]] += 1
    for prov, cnt in sorted(counts.items()):
        print(f"    {prov:<10} {cnt} family(ies)")
    if result["missing_target_support"]:
        print(f"  WARNING: {len(result['missing_target_support'])} region(s) with no target support")
    print(f"  -> {fusion_path.relative_to(ROOT)}")
    print(f"  -> {post_plan_path.relative_to(ROOT)}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workload", required=True,
                   choices=["smolvla_slice", "gemma_decode_slice"])
    p.add_argument("--target", default="openq_5165rb")
    args = p.parse_args()
    return run(args.workload, args.target)


if __name__ == "__main__":
    sys.exit(main())
