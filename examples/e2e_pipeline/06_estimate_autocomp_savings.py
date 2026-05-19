"""Phase 6 probe — expected autocomp wall-time savings.

Reads the baseline (from 04) and post-plan family list (from 05), plus
the target resource model's `autocomp_cost_coefficients`, and computes:

    baseline_seconds   = sum over baseline families of
                           per_region_base  +  per_shape_variant  +  per_dtype_variant
                         (autocomp has to search every region independently)

    post_plan_seconds  = sum over post-plan families of
                           per_region_base * (1 - library_match_discount if provider==library else 1)
                           + (region_count - 1) * per_shape_variant
                           + per_dtype_variant
                         (autocomp reuses the search across regions in a
                          family, with a per-variant additional cost)

    savings = baseline_seconds - post_plan_seconds
    ratio   = baseline_seconds / post_plan_seconds

Writes a human-readable markdown report under
artifacts/<workload>/stage_6_autocomp_savings/<target>/autocomp_savings.md
and a structured companion YAML for machine consumption.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(ROOT.parent))

from user_perspective.scripts._helpers.proto_schema import validate_or_raise_proto  # noqa: E402


def _load_yaml(path: Path) -> Any:
    if not path.exists():
        raise SystemExit(f"missing: {path}")
    return yaml.safe_load(path.read_text())


def _baseline_cost(
    baseline: dict[str, Any], coeffs: dict[str, float]
) -> dict[str, Any]:
    per_region = coeffs["per_region_base"]
    per_shape = coeffs.get("per_shape_variant", 0.0)
    per_dtype = coeffs.get("per_dtype_variant", 0.0)
    per_layout = coeffs.get("per_layout_variant", 0.0)

    lines: list[dict[str, Any]] = []
    total = 0.0
    for fam in baseline["families"]:
        # Baseline: every family is one unique (region, shape, dtype,
        # layout) combination; autocomp sees them all distinctly.
        cost = per_region + per_shape + per_dtype + per_layout
        lines.append(
            {
                "family_id": fam["family_id"],
                "semantic_kind": fam["family_key"]["semantic_kind"],
                "seconds": round(cost, 3),
            }
        )
        total += cost
    return {
        "per_family_seconds": lines,
        "total_seconds": round(total, 3),
        "family_count": len(lines),
    }


def _post_plan_cost(
    post_plan: dict[str, Any], coeffs: dict[str, float]
) -> dict[str, Any]:
    per_region = coeffs["per_region_base"]
    per_shape = coeffs.get("per_shape_variant", 0.0)
    per_dtype = coeffs.get("per_dtype_variant", 0.0)
    library_discount = coeffs.get("library_match_discount", 1.0)

    lines: list[dict[str, Any]] = []
    total = 0.0
    for fam in post_plan["families"]:
        provider = fam["provider"]
        region_count = fam["region_count"]

        if provider == "library":
            # match_library_call succeeded → autocomp does nothing for
            # this family. `library_match_discount == 1.0` means full
            # discount (cost 0); otherwise fractional.
            cost = per_region * (1.0 - library_discount)
        elif provider == "ukernel":
            # Ukernel is pre-written; small pinning cost only, no autocomp.
            cost = per_region * 0.1
        else:  # autocomp
            # Autocomp searches once, then reuses across regions within
            # the family; each additional region adds a shape variant.
            cost = per_region + max(0, region_count - 1) * per_shape + per_dtype

        lines.append(
            {
                "target_family": fam["target_family"],
                "provider": provider,
                "region_count": region_count,
                "seconds": round(cost, 3),
            }
        )
        total += cost
    return {
        "per_family_seconds": lines,
        "total_seconds": round(total, 3),
        "family_count": len(lines),
    }


def compute(workload: str, target: str) -> dict[str, Any]:
    baseline_path = (
        ROOT / "artifacts" / workload / "stage_4_kernel_families" / target
        / "kernel_families_baseline.yaml"
    )
    post_plan_path = (
        ROOT / "artifacts" / workload / "stage_5_fusion_plan" / target
        / "post_plan_kernel_families.yaml"
    )
    target_path = ROOT / "configs" / "targets" / f"{target}.v2.yaml"

    baseline = _load_yaml(baseline_path)
    post_plan = _load_yaml(post_plan_path)
    target_doc = _load_yaml(target_path)
    validate_or_raise_proto(target_doc, "target_resource.v2")
    coeffs = target_doc["autocomp_cost_coefficients"]

    baseline_cost = _baseline_cost(baseline, coeffs)
    post_plan_cost = _post_plan_cost(post_plan, coeffs)
    ratio = baseline_cost["total_seconds"] / max(
        1e-9, post_plan_cost["total_seconds"]
    )
    savings = baseline_cost["total_seconds"] - post_plan_cost["total_seconds"]

    return {
        "workload": workload,
        "target": target,
        "coefficients": coeffs,
        "baseline": baseline_cost,
        "post_plan": post_plan_cost,
        "savings_seconds": round(savings, 3),
        "ratio": round(ratio, 2),
        "meets_2x_threshold": ratio >= 2.0,
    }


def _render_markdown(result: dict[str, Any]) -> str:
    lines = [
        f"# Autocomp-cost savings estimate — {result['workload']} @ {result['target']}",
        "",
        "Generated by `user_perspective/scripts/06_estimate_autocomp_savings.py`.",
        "Coefficients from the target_resource.v2 YAML's "
        "`autocomp_cost_coefficients` block.",
        "",
        "## Coefficients used",
        "",
        "```yaml",
        yaml.safe_dump(result["coefficients"], sort_keys=False).strip(),
        "```",
        "",
        "## Summary",
        "",
        f"- Baseline families (1 per region): **{result['baseline']['family_count']}**",
        f"- Post-plan families (canonical): **{result['post_plan']['family_count']}**",
        f"- Baseline total autocomp wall-time: **{result['baseline']['total_seconds']:,.1f} s** "
        f"(~{result['baseline']['total_seconds']/60:.1f} min)",
        f"- Post-plan total autocomp wall-time: **{result['post_plan']['total_seconds']:,.1f} s** "
        f"(~{result['post_plan']['total_seconds']/60:.1f} min)",
        f"- Savings: **{result['savings_seconds']:,.1f} s** "
        f"(~{result['savings_seconds']/60:.1f} min)",
        f"- Ratio: **{result['ratio']}×**",
        f"- Meets ≥ 2× target: **{'✓' if result['meets_2x_threshold'] else '✗'}**",
        "",
        "## Baseline breakdown",
        "",
        "| family_id | semantic_kind | seconds |",
        "|---|---|---:|",
    ]
    for row in result["baseline"]["per_family_seconds"]:
        lines.append(
            f"| `{row['family_id']}` | {row['semantic_kind']} | {row['seconds']:,.1f} |"
        )
    lines.extend(["", "## Post-plan breakdown", "",
                  "| target_family | provider | region_count | seconds |",
                  "|---|---|---:|---:|"])
    for row in result["post_plan"]["per_family_seconds"]:
        lines.append(
            f"| {row['target_family']} | {row['provider']} | "
            f"{row['region_count']} | {row['seconds']:,.1f} |"
        )
    lines.extend(["", "## Caveats", "",
                  "- Cost coefficients are seed estimates; recalibrate after "
                  "real autocomp runs on this target (see "
                  "`analysis/prototype_experiments.md` §13).",
                  "- `library_match_discount` collapses library-matched families "
                  "to near-zero autocomp cost; this is correct for QNN HTP / "
                  "cuBLAS / oneDNN paths but overly optimistic if the library "
                  "match is partial.",
                  "- Baseline assumes zero fusion; post-plan assumes the "
                  "canonical `propose_fusion` invent-slot output from "
                  "`scripts/05_dryrun_fusion_plan.py`. The LLM may do better or "
                  "worse than the canonical plan; this estimate is the *target*.",
                  ""])
    return "\n".join(lines)


def run(workload: str, target: str) -> int:
    result = compute(workload, target)
    out_dir = ROOT / "artifacts" / workload / "stage_6_autocomp_savings" / target
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / "autocomp_savings.md"
    yaml_path = out_dir / "autocomp_savings.yaml"
    md_path.write_text(_render_markdown(result), encoding="utf-8")
    yaml_path.write_text(
        yaml.safe_dump(result, sort_keys=False, default_flow_style=False),
        encoding="utf-8",
    )

    print(f"\nAutocomp-cost savings — {workload} @ {target}")
    print(f"  baseline:  {result['baseline']['total_seconds']:>10,.1f} s  "
          f"({result['baseline']['family_count']} families)")
    print(f"  post-plan: {result['post_plan']['total_seconds']:>10,.1f} s  "
          f"({result['post_plan']['family_count']} families)")
    print(f"  savings:   {result['savings_seconds']:>10,.1f} s  "
          f"(ratio = {result['ratio']}×)")
    print(f"  meets 2× threshold: {result['meets_2x_threshold']}")
    print(f"  -> {md_path.relative_to(ROOT)}")
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
