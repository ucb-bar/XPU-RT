"""Phase 4 probe — baseline kernel-family count per workload.

Reads the Phase 3 `region_inventory.json` and reports the baseline
kernel-family count: one kernel-generation unit per region. This is
what autocomp would have to search over if no Phase 2/3 fusion
happened (no `raise_special_ops`, no `propose_fusion`, no
`match_library_call`). Phase 2/3 reduce this count; script 05
computes the post-plan count; script 06 translates the delta into
expected autocomp wall-time savings.

Why one-family-per-region (rather than one-family-per-semantic-kind):
  Each region is a distinct set of ops the codegen has to handle
  together. Even two regions with the same semantic_kind (e.g. both
  'matmul') typically have different shapes, different neighbour ops,
  and therefore land on different autocomp search trajectories. The
  coarse count by semantic_kind is also reported, for orientation.

No CompGen imports — this probe consumes only pre-existing artifacts,
so it works even if compgen.agent internals change.

Artifacts:
  artifacts/<workload>/stage_4_kernel_families/<target>/kernel_families_baseline.yaml
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent


@dataclass
class FamilyKey:
    semantic_kind: str
    shape_regime: str
    shape_bucket: int  # log2-MFLOP bucket; -1 for "no flops"

    def as_tuple(self) -> tuple[str, str, int]:
        return (self.semantic_kind, self.shape_regime, self.shape_bucket)

    def as_dict(self) -> dict[str, Any]:
        return {
            "semantic_kind": self.semantic_kind,
            "shape_regime": self.shape_regime,
            "shape_bucket": self.shape_bucket,
        }


def _shape_bucket(flops: int) -> int:
    if flops <= 0:
        return -1
    mflops = flops / (1024 * 1024)
    if mflops < 1.0:
        return 0
    return int(min(16, max(1, round(math.log2(mflops)))))


def _classify(region: dict[str, Any]) -> FamilyKey:
    return FamilyKey(
        semantic_kind=region["semantic_kind"],
        shape_regime=region["shape_regime"],
        shape_bucket=_shape_bucket(int(region.get("estimated_flops") or 0)),
    )


def probe(workload: str, target: str) -> dict[str, Any]:
    inventory_path = (
        ROOT
        / "artifacts"
        / workload
        / "stage_3_analysis"
        / target
        / "region_inventory.json"
    )
    if not inventory_path.exists():
        raise SystemExit(
            f"missing {inventory_path}. Run scripts/03_graph_analysis.py "
            f"--workload {workload} --target {target} first."
        )

    inventory = json.loads(inventory_path.read_text())
    regions: list[dict[str, Any]] = inventory["regions"]

    # Baseline: one family per region. Each region is a distinct
    # kernel-search unit absent fusion.
    families: list[dict[str, Any]] = []
    coverage_counts: dict[str, int] = defaultdict(int)
    by_kind: dict[str, int] = defaultdict(int)
    for r in regions:
        key = _classify(r)
        families.append(
            {
                "family_id": f"baseline_{r['region_id']}",
                "region_ids": [r["region_id"]],
                "family_key": key.as_dict(),
                "kernel_coverage_potential": r.get("kernel_coverage_potential", "single_family"),
                "placement_affinity": r.get("placement_affinity", []),
                "accuracy_sensitivity": r.get("accuracy_sensitivity", "quantization_sensitive"),
                "ops": list(r.get("ops", [])),
                "estimated_flops": int(r.get("estimated_flops") or 0),
                "estimated_bytes": int(r.get("estimated_bytes") or 0),
                "region_count": 1,
            }
        )
        coverage_counts[r.get("kernel_coverage_potential", "single_family")] += 1
        by_kind[r["semantic_kind"]] += 1

    # Coarse view: collapse all regions sharing a semantic_kind (what a
    # very naïve "fuse all same-kind regions" policy would give; for
    # orientation only, since real Phase 2/3 fusion considers shapes,
    # dtypes, and target feature alignment).
    coarse_families_by_kind = {k: {"region_count": v} for k, v in by_kind.items()}

    return {
        "workload": workload,
        "target": target,
        "baseline_family_count": len(families),
        "region_count": len(regions),
        "coverage_counts": dict(coverage_counts),
        "coarse_families_by_semantic_kind": coarse_families_by_kind,
        "coarse_family_count": len(by_kind),
        "families": families,
        "source_inventory": str(inventory_path.relative_to(ROOT)),
        "bucketing": {
            "primary": "one_family_per_region",
            "rationale": "each region is a distinct kernel-search unit before fusion",
            "secondary_by_semantic_kind": "orientation only",
        },
    }


def run(workload: str, target: str) -> int:
    out_dir = ROOT / "artifacts" / workload / "stage_4_kernel_families" / target
    out_dir.mkdir(parents=True, exist_ok=True)

    result = probe(workload, target)
    out_path = out_dir / "kernel_families_baseline.yaml"
    out_path.write_text(
        yaml.safe_dump(result, sort_keys=False, default_flow_style=False),
        encoding="utf-8",
    )

    print(f"\nBaseline kernel-family probe — {workload} @ {target}")
    print(f"  regions:                 {result['region_count']}")
    print(f"  baseline families:       {result['baseline_family_count']} (one per region)")
    print(f"  coarse (by kind):        {result['coarse_family_count']}")
    print(f"  coverage:                {result['coverage_counts']}")
    print("  by semantic_kind:")
    for kind, info in sorted(result["coarse_families_by_semantic_kind"].items(),
                              key=lambda kv: -kv[1]["region_count"]):
        print(f"    {kind:<16} {info['region_count']} region(s)")
    print(f"  -> {out_path.relative_to(ROOT)}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workload", required=True,
                   choices=["smolvla_slice", "gemma_decode_slice"])
    p.add_argument("--target", default="openq_5165rb",
                   help="Must match a target whose stage_3_analysis has run.")
    args = p.parse_args()
    return run(args.workload, args.target)


if __name__ == "__main__":
    sys.exit(main())
