#!/usr/bin/env python
"""Empirical cost-calibration CLI.

For each model in the requested set, run the graph-compilation
pipeline once per viable candidate (forced via agent-file mode)
and measure CPU latency. Emit a per-model JSON calibration table
plus an aggregate markdown summary that shows, for each model:

- the empirically-best candidate (lowest min-latency, verified),
- the static-priority pick (lowest static_relative_cost),
- the **regret** percentage of the static pick vs the best.

Regret > 0 means the static cost model is leaving performance on
the table. The open caveat `fusion_cost_model_uncalibrated`
closes when a learned cost model drives mean regret to near-zero
across the canonical+holdout set.

Usage::

    uv run python scripts/dev/run_cost_calibration.py \\
        --models merlin_mlp_wide proxy_vla \\
        --target host_cpu

Outputs land at ``results/audit/<commit>/cost_calibration/`` with:
- ``<model_id>.json`` per model,
- ``summary.md`` aggregate.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from xpu_rt.benchmarks.cost_calibration import (
    ModelCalibration,
    calibrate_model,
    emit_calibration,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
PKG_ROOT = REPO_ROOT / "xpu-rt"

DEFAULT_MODELS = (
    "tiny_mlp",
    "merlin_mlp_wide",
    "proxy_vla",
    "holdout_pointwise_chain_renamed",
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


def _render_summary(cals: list[ModelCalibration], commit: str) -> str:
    lines: list[str] = []
    lines.append(f"# Cost calibration — {commit}")
    lines.append("")
    lines.append(f"_{len(cals)} model(s) calibrated._")
    lines.append("")
    lines.append("| Model | Best candidate | Best min µs | Static pick | "
                 "Static min µs | Regret % | # viable |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for c in cals:
        b = c.best()
        s = c.static_pick()
        r = c.regret_pct()
        if b is None or s is None:
            lines.append(
                f"| `{c.model_id}` | — | — | — | — | — "
                f"| {len(c.measurements)} |"
            )
            continue
        same = b.candidate_id == s.candidate_id
        regret_cell = (
            "0.00 (match)" if same else f"{r:+.2f}" if r is not None else "—"
        )
        lines.append(
            f"| `{c.model_id}` "
            f"| `{b.candidate_kind}` ({b.candidate_id[:32]}…) "
            f"| {b.latency_min_us:.1f} "
            f"| `{s.candidate_kind}` ({s.candidate_id[:32]}…) "
            f"| {s.latency_min_us:.1f} "
            f"| **{regret_cell}** "
            f"| {len(c.measurements)} |"
        )
    lines.append("")
    # Aggregate regret.
    regrets = [
        c.regret_pct() for c in cals if c.regret_pct() is not None
    ]
    nz_regrets = [r for r in regrets if r is not None and r > 1.0]
    if regrets:
        lines.append(
            f"**Aggregate:** mean regret "
            f"{sum(r for r in regrets if r is not None) / len(regrets):+.2f}%, "
            f"{len(nz_regrets)} of {len(regrets)} models leave "
            f">1% on the table with the static cost model."
        )
        lines.append("")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", nargs="+", default=None)
    p.add_argument("--target", default="host_cpu")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--latency-iters", type=int, default=100)
    p.add_argument("--latency-warmup", type=int, default=10)
    p.add_argument(
        "--cache", action="store_true",
        help=(
            "Also write each per-model calibration to the production "
            "cache at .xpu_rt_cache/cost_calibration/<model_id>.json "
            "(picked up automatically by greedy's oracle picker)."
        ),
    )
    args = p.parse_args(argv)

    # See run_subsystem_ablation.py for the same anchor — model
    # configs reference `tests/graph_compilation/models/<id>.py`
    # relative to xpu-rt/.
    if Path.cwd() != PKG_ROOT.resolve():
        os.chdir(PKG_ROOT)

    commit = _git_short_commit()
    out_dir = args.out or (
        PKG_ROOT / "results" / "audit" / commit / "cost_calibration"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    target_yaml = PKG_ROOT / "configs" / "targets" / f"{args.target}.yaml"
    if not target_yaml.exists():
        print(f"FAIL: target config not found: {target_yaml}", file=sys.stderr)
        return 2

    models = list(args.models or DEFAULT_MODELS)
    cals: list[ModelCalibration] = []
    for model_id in models:
        cfg = PKG_ROOT / "configs" / "models" / f"{model_id}.yaml"
        if not cfg.exists():
            print(f"WARN: skipping {model_id} (no config)", file=sys.stderr)
            continue
        print(f"calibrating {model_id} ...", file=sys.stderr)
        cal = calibrate_model(
            model_yaml=cfg, target_yaml=target_yaml,
            out_dir=out_dir / model_id,
            n_warmup=args.latency_warmup,
            n_iters=args.latency_iters,
        )
        cal.commit = commit
        emit_calibration(cal, out_path=out_dir / f"{model_id}.json")
        if args.cache:
            # Production cache location, auto-discovered by greedy.
            cache_path = (
                PKG_ROOT / ".xpu_rt_cache" / "cost_calibration"
                / f"{model_id}.json"
            )
            emit_calibration(cal, out_path=cache_path)
        cals.append(cal)
        b = cal.best()
        s = cal.static_pick()
        r = cal.regret_pct()
        if b and s:
            print(
                f"  {model_id}: best={b.candidate_kind} ({b.latency_min_us:.1f} µs) "
                f"| static={s.candidate_kind} ({s.latency_min_us:.1f} µs) "
                f"| regret={r:+.2f}%" if r is not None else "—",
                file=sys.stderr,
            )

    (out_dir / "summary.md").write_text(_render_summary(cals, commit))
    print(f"\nresults: {out_dir / 'summary.md'}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
