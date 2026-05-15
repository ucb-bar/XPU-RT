#!/usr/bin/env python
"""Run the Section 20 stack on a set of small models and produce
per-model inspection packs + a cross-model OVERVIEW.

Usage::

    uv run python scripts/dev/inspect_models.py [--models ...] [--out <dir>]

Default model set:
    tiny_mlp, merlin_mlp, residual_branch, tiny_attention,
    tiny_conv_block, holdout_mlp_odd_shapes
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from compgen.benchmarks.model_inspection import (
    InspectionPack,
    aggregate_inspection_packs,
    inspect_model_run,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_MODELS = (
    "tiny_mlp",
    "merlin_mlp",
    "residual_branch",
    "tiny_attention",
    "tiny_conv_block",
    "holdout_mlp_odd_shapes",
)


def _git_short_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"],
            cwd=REPO_ROOT, capture_output=True, text=True, check=True,
        )
        return out.stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=None,
                   help="Output dir (default: results/audit/<commit>/inspection/)")
    p.add_argument("--models", nargs="+", default=None,
                   help="Model ids (default: 6 small + 1 holdout)")
    p.add_argument("--target", default="host_cpu",
                   help="Target id (default: host_cpu)")
    p.add_argument("--stop-after", default="agent-decision-request",
                   help="Pipeline stop_after stage")
    args = p.parse_args(argv)

    commit = _git_short_commit()
    out_dir = args.out or (REPO_ROOT / "results" / "audit" / commit / "inspection")
    out_dir.mkdir(parents=True, exist_ok=True)

    models = list(args.models or DEFAULT_MODELS)
    target_yaml = REPO_ROOT / "configs" / "targets" / f"{args.target}.yaml"
    if not target_yaml.exists():
        print(f"FAIL: target config not found: {target_yaml}", file=sys.stderr)
        return 2

    packs: list[InspectionPack] = []
    for model_id in models:
        model_yaml = REPO_ROOT / "configs" / "models" / f"{model_id}.yaml"
        if not model_yaml.exists():
            print(f"WARN: skipping {model_id} (no config at {model_yaml})", file=sys.stderr)
            continue
        print(f"\n=== inspecting {model_id} ===")
        model_dir = out_dir / model_id
        pack = inspect_model_run(
            model_yaml=model_yaml,
            target_yaml=target_yaml,
            out_dir=model_dir,
            stop_after=args.stop_after,
        )
        packs.append(pack)
        d = pack.decision_summary
        w = pack.warm_cache_summary
        print(f"  outcome:      {pack.typed_outcome}")
        print(f"  decision:     {d.get('candidate_kind') or '(none)'} / "
              f"{(d.get('selected_candidate_id') or '(none)')[:60]}")
        print(f"  warm-hit:     {d.get('warm_cache_hit')}")
        print(f"  promoted:     {w.get('promoted_candidates_count', 0)} surfaced, "
              f"{'hit' if w.get('promoted_hit') else 'miss'}")
        print(f"  cards seen:   {pack.pass_card_visibility.get('card_count', 0)}")
        print(f"  summaries:    "
              f"{pack.analysis_summary_index.get('available_count', 0)}/"
              f"{pack.analysis_summary_index.get('summary_count', 0)} available")
        print(f"  inspection:   {model_dir}/INSPECTION.md")

    if packs:
        overview = aggregate_inspection_packs(packs, out_path=out_dir / "OVERVIEW.md")
        print(f"\noverview:    {overview}")
        print(f"per-model:   {out_dir}/<model_id>/INSPECTION.md")
    return 0 if packs else 1


if __name__ == "__main__":
    raise SystemExit(main())
