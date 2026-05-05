"""Measure cold-vs-warm promotion efficiency on a model suite (M-30).

Runs the canonical-22 (or any user-supplied) suite twice, and writes a
paper-ready comparison table demonstrating Section 19's falsifiable
claim:

    Cold-run vs warm-run on the same suite shows
    ``fresh_emit_count_warm < fresh_emit_count_cold`` and
    ``gemini_token_delta < 0`` while every correctness gate in
    ``verification_report.json`` still passes.

Usage::

    uv run python scripts/dev/measure_promotion_efficiency.py \
        --models tiny_mlp tiny_attention proxy_vla \
        --target host_cpu \
        --library .compgen_cache/recipes \
        --out results/paper/promotion_efficiency

The script does *not* attempt to be clever about caching — it
literally runs each model end-to-end against an empty library
(``cold``) and again against the populated library produced by the
cold pass (``warm``). The aggregator reads the resulting run dirs
and emits ``promotion_efficiency_pack.json`` plus a Markdown summary.

The script is best-effort: per-model failures are recorded in the
output but do not abort the suite. This matters because some Phase
B models (e.g. tiny_mlp) hit known M-15B downstream rejections that
are unrelated to Section 19; we still want a useful comparison
across the rest of the suite.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from compgen.graph_compilation.efficiency_report import (
    EfficiencyDelta,
    compare_runs,
)


@dataclass(frozen=True)
class RunOutcome:
    model_id: str
    cold_run_dir: Path | None
    warm_run_dir: Path | None
    cold_status: str  # "ok" | "error: ..."
    warm_status: str


def _run_one(
    *,
    model_config: Path,
    target_config: Path,
    out_dir: Path,
    stop_after: str,
    repo_root: Path,
) -> tuple[Path | None, str]:
    """Invoke ``python -m compgen.graph_compilation`` for one model."""
    cmd = [
        sys.executable,
        "-m",
        "compgen.graph_compilation",
        "run",
        "--model",
        str(model_config),
        "--target",
        str(target_config),
        "--out",
        str(out_dir),
        "--stop-after",
        stop_after,
        "--selection-mode",
        "greedy",
    ]
    result = subprocess.run(
        cmd, cwd=repo_root, capture_output=True, text=True, timeout=600,
    )
    if result.returncode != 0:
        return None, f"exit={result.returncode}: {result.stderr.strip().splitlines()[-1] if result.stderr else 'no stderr'}"
    return out_dir, "ok"


def _measure(
    *,
    models: list[str],
    target_id: str,
    library: Path,
    out_dir: Path,
    repo_root: Path,
    stop_after: str,
) -> list[RunOutcome]:
    """Run every model cold + warm and collect outcomes."""
    target_config = repo_root / "configs" / "targets" / f"{target_id}.yaml"
    cold_root = out_dir / "cold"
    warm_root = out_dir / "warm"
    cold_root.mkdir(parents=True, exist_ok=True)
    warm_root.mkdir(parents=True, exist_ok=True)

    # --- Cold: nuke library before EACH model so no prior promotion
    # ever surfaces during the cold pass.
    outcomes: list[RunOutcome] = []
    for model_id in models:
        model_config = repo_root / "configs" / "models" / f"{model_id}.yaml"
        if library.exists():
            shutil.rmtree(library)
        cold_dir = cold_root / model_id
        cold_run, cold_status = _run_one(
            model_config=model_config,
            target_config=target_config,
            out_dir=cold_dir,
            stop_after=stop_after,
            repo_root=repo_root,
        )
        # Warm: keep the library as the cold pass populated it (single-
        # model warmup is the simplest form; cross-model reuse picks
        # up whatever survived the cold run).
        warm_dir = warm_root / model_id
        warm_run, warm_status = _run_one(
            model_config=model_config,
            target_config=target_config,
            out_dir=warm_dir,
            stop_after=stop_after,
            repo_root=repo_root,
        )
        outcomes.append(
            RunOutcome(
                model_id=model_id,
                cold_run_dir=cold_run,
                warm_run_dir=warm_run,
                cold_status=cold_status,
                warm_status=warm_status,
            )
        )
    return outcomes


def _format_md(deltas: list[EfficiencyDelta], errors: list[RunOutcome]) -> str:
    lines: list[str] = ["# Promotion-efficiency cold-vs-warm comparison\n"]
    lines.append("| model | cold fresh | warm fresh | delta | claim_supported |")
    lines.append("|---|---:|---:|---:|---|")
    for d in deltas:
        lines.append(
            f"| {d.model_id} | {d.cold.fresh_emit_count} | "
            f"{d.warm.fresh_emit_count} | {d.fresh_emit_delta():+} | "
            f"{'YES' if d.to_dict()['claim_supported'] else 'NO'} |"
        )
    if errors:
        lines.append("\n## Errors\n")
        for e in errors:
            lines.append(
                f"- **{e.model_id}**: cold={e.cold_status!r}, warm={e.warm_status!r}"
            )
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--target", default="host_cpu")
    ap.add_argument(
        "--library",
        default=".compgen_cache/recipes",
        help="Recipe library path (gets nuked between cold runs).",
    )
    ap.add_argument(
        "--out",
        default="results/paper/promotion_efficiency",
        help="Output directory for run dirs and the summary pack.",
    )
    ap.add_argument(
        "--stop-after",
        default="agent-decision-request",
        help="Phase B stop_after stage; defaults to where promotion bridge fires.",
    )
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    library = (repo_root / args.library).resolve() if not Path(args.library).is_absolute() else Path(args.library)
    out_dir = (repo_root / args.out).resolve() if not Path(args.out).is_absolute() else Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    outcomes = _measure(
        models=args.models,
        target_id=args.target,
        library=library,
        out_dir=out_dir,
        repo_root=repo_root,
        stop_after=args.stop_after,
    )

    deltas: list[EfficiencyDelta] = []
    errors: list[RunOutcome] = []
    for o in outcomes:
        if o.cold_run_dir is None or o.warm_run_dir is None:
            errors.append(o)
            continue
        try:
            deltas.append(
                compare_runs(
                    model_id=o.model_id,
                    cold_run=o.cold_run_dir,
                    warm_run=o.warm_run_dir,
                    library_path=library,
                    repo_root=repo_root,
                )
            )
        except Exception as exc:  # noqa: BLE001
            errors.append(
                RunOutcome(
                    model_id=o.model_id,
                    cold_run_dir=o.cold_run_dir,
                    warm_run_dir=o.warm_run_dir,
                    cold_status=o.cold_status,
                    warm_status=f"compare_runs failed: {type(exc).__name__}: {exc}",
                )
            )

    pack: dict[str, Any] = {
        "schema_version": "promotion_efficiency_pack_v1",
        "deltas": [d.to_dict() for d in deltas],
        "errors": [
            {
                "model_id": e.model_id,
                "cold_status": e.cold_status,
                "warm_status": e.warm_status,
            }
            for e in errors
        ],
    }
    pack_path = out_dir / "promotion_efficiency_pack.json"
    pack_path.write_text(
        json.dumps(pack, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    md_path = out_dir / "promotion_efficiency_pack.md"
    md_path.write_text(_format_md(deltas, errors), encoding="utf-8")

    print(f"wrote {pack_path}")
    print(f"wrote {md_path}")
    return 0 if all(d.to_dict()["claim_supported"] for d in deltas) else 1


if __name__ == "__main__":
    sys.exit(main())
