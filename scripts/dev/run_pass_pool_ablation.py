#!/usr/bin/env python
"""Run the pass-pool ablation on a model set (M-36).

Compares CompGen pipeline behavior across selection modes on the same
models. Today the harness exercises ``greedy`` mode and any
operator-supplied ``agent-file`` responses; ``llm-live`` is opt-in via
the operator (api keys / network) and not driven from this CLI.

Default model set: merlin_mlp_wide + proxy_vla + the 5 holdouts.

Usage::

    uv run python scripts/dev/run_pass_pool_ablation.py \\
        --out results/audit/ablation/<commit>/

Exit 0 on success. Writes ``ablation_pack.json`` + ``ablation_pack.md``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from compgen.benchmarks.pass_pool_ablation import (
    AblationCellSpec,
    AblationPack,
    emit_pack,
    run_suite,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_MODELS = (
    "merlin_mlp_wide",
    "proxy_vla",
    "holdout_mlp_odd_shapes",
    "holdout_mlp_large_k",
    "holdout_pointwise_chain_renamed",
    "holdout_two_matmuls_shared_input",
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


def render_markdown(pack: AblationPack) -> str:
    """Render the pack as a paper-ready markdown table."""
    summary = pack.summary()
    lines: list[str] = []
    lines.append(f"# Pass-pool ablation — {pack.commit}")
    lines.append("")
    lines.append(f"_Generated: `{pack.generated_at_utc}`_")
    lines.append("")
    lines.append(f"**{summary['cell_count']} cells** across {len(summary['modes'])} modes "
                 f"({len(summary['models'])} models).")
    lines.append("")
    lines.append("## Per-mode summary")
    lines.append("")
    lines.append("| Mode | Cells | Verified | Typed-blocked | Error | "
                 "Promoted hits | Mean decision seconds |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for mode in summary["modes"]:
        row = summary["per_mode"][mode]
        lines.append(
            f"| `{mode}` | {row['cell_count']} | {row['verified']} | "
            f"{row['typed_blocked']} | {row['error']} | "
            f"{row['promoted_hit_count']} / {row['promoted_candidates_total']} | "
            f"{row['mean_decision_seconds']:.2f} |"
        )
    lines.append("")
    if summary.get("promoted_hit_count_total", 0) > 0 or any(
        row.get("promoted_candidates_total", 0) > 0
        for row in summary.get("per_mode", {}).values()
    ):
        lines.append("## Warm-cache effectiveness (M-37.2)")
        lines.append("")
        lines.append(
            f"- **Promoted hits**: "
            f"{summary['promoted_hit_count_total']} / {summary['cell_count']} "
            f"cells ({summary['promoted_hit_rate']:.1%}) saw the agent pick "
            f"a candidate that matched a promoted recipe."
        )
        lines.append("")
    lines.append("## Per-cell details")
    lines.append("")
    lines.append("| Model | Mode | Selected candidate | Pass | Outcome |")
    lines.append("| --- | --- | --- | --- | --- |")
    for c in pack.cells:
        lines.append(
            f"| `{c.model_id}` | `{c.mode}` | `{c.selected_candidate_id or '(none)'}` | "
            f"`{c.pass_id or '(none)'}` | {c.typed_outcome} |"
        )
    lines.append("")
    if pack.divergences():
        lines.append("## Divergences (modes disagreed on candidate)")
        lines.append("")
        for row in pack.divergences():
            lines.append(f"### {row['model_id']}")
            lines.append("")
            for mode, pick in row["picks_by_mode"].items():
                lines.append(f"- `{mode}` → `{pick or '(none)'}`")
            lines.append("")
    else:
        lines.append("## Divergences")
        lines.append("")
        lines.append("No divergence between modes on this run "
                     "(every cell's modes agreed on selected_candidate_id).")
        lines.append("")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=None,
                   help="Output dir (default: results/audit/<commit>/ablation/)")
    p.add_argument("--models", nargs="+", default=None,
                   help="Model ids to run (default: 6 canonical + holdouts)")
    p.add_argument("--target", default="host_cpu",
                   help="Target id (default: host_cpu)")
    p.add_argument("--modes", nargs="+", default=["greedy"],
                   help="Modes to run (default: greedy only)")
    p.add_argument("--agent-response-dir", type=Path, default=None,
                   help="Directory containing operator-authored "
                        "agent_decision_response.json files for agent-file mode "
                        "(file naming: <model_id>.json)")
    args = p.parse_args(argv)

    commit = _git_short_commit()
    out_dir = args.out or (REPO_ROOT / "results" / "audit" / commit / "ablation")
    out_dir.mkdir(parents=True, exist_ok=True)

    models = list(args.models or DEFAULT_MODELS)
    target_yaml = REPO_ROOT / "configs" / "targets" / f"{args.target}.yaml"
    if not target_yaml.exists():
        print(f"FAIL: target config not found: {target_yaml}", file=sys.stderr)
        return 2

    cells: list[AblationCellSpec] = []
    for model_id in models:
        model_yaml = REPO_ROOT / "configs" / "models" / f"{model_id}.yaml"
        if not model_yaml.exists():
            print(f"WARN: skipping {model_id} (no config at {model_yaml})", file=sys.stderr)
            continue
        for mode in args.modes:
            agent_resp = None
            if mode == "agent-file" and args.agent_response_dir:
                candidate_resp = args.agent_response_dir / f"{model_id}.json"
                if candidate_resp.exists():
                    agent_resp = candidate_resp
                else:
                    print(
                        f"WARN: skipping {model_id}/agent-file "
                        f"(no response at {candidate_resp})",
                        file=sys.stderr,
                    )
                    continue
            cells.append(AblationCellSpec(
                model_yaml=model_yaml,
                target_yaml=target_yaml,
                mode=mode,
                agent_response_path=agent_resp,
            ))

    if not cells:
        print("FAIL: no cells to run", file=sys.stderr)
        return 1

    print(f"running {len(cells)} cells under {sorted(set(c.mode for c in cells))} ...")
    pack = run_suite(cells, out_root=out_dir / "runs", commit=commit)

    json_path = out_dir / "ablation_pack.json"
    md_path = out_dir / "ablation_pack.md"
    emit_pack(pack, out_path=json_path)
    md_path.write_text(render_markdown(pack), encoding="utf-8")

    summary = pack.summary()
    print(f"\nresults: {json_path}")
    print(f"         {md_path}")
    print(f"  cells: {summary['cell_count']} ({summary['divergence_count']} divergences)")
    for mode, row in summary["per_mode"].items():
        print(
            f"  {mode}: {row['verified']} verified / "
            f"{row['typed_blocked']} blocked / {row['error']} error"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
