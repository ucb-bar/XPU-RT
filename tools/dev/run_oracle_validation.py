#!/usr/bin/env python
"""Oracle picker validation CLI.

For every model that has a calibration table in
``--calibration-dir``, run greedy twice (static vs oracle) and
report the actually-delivered regret.

Closes the loop on the `cost_model_uncalibrated_across_decisions`
caveat: if mean delivered regret falls below 2% across the
canonical+holdout set, the caveat closes.

Usage::

    uv run python scripts/dev/run_oracle_validation.py \\
        --calibration-dir results/audit/<commit>/cost_calibration \\
        --target host_cpu
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from xpu_rt.benchmarks.oracle_validation import (
    OracleValidationPack,
    emit_pack,
    validate_oracle_for_model,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
PKG_ROOT = REPO_ROOT / "xpu-rt"


def _git_short_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"],
            cwd=REPO_ROOT, capture_output=True, text=True, check=True,
        )
        return out.stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _render_summary(pack: OracleValidationPack) -> str:
    lines: list[str] = []
    s = pack.summary()
    lines.append(f"# Oracle validation — {pack.commit}")
    lines.append("")
    lines.append(f"_{s['model_count']} model(s); calibration dir: `{pack.calibration_dir}`_")
    lines.append("")
    lines.append(
        f"**Mean delivered regret (static vs oracle):** "
        f"{s['mean_delivered_regret_pct']:+.2f}%. "
        f"{s['models_with_regret_ge_2pct']} models ≥2%, "
        f"{s['models_with_regret_ge_5pct']} models ≥5%, "
        f"max {s['max_delivered_regret_pct']:+.2f}%."
    )
    lines.append("")
    lines.append(
        "| Model | Static pick | Static µs | Oracle pick | Oracle µs | "
        "Predicted µs | Regret % | Delivered−Predicted % |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for r in pack.rows:
        regret = (
            f"{r.delivered_regret_pct:+.2f}"
            if r.delivered_regret_pct is not None else "—"
        )
        gap = (
            f"{r.delivered_vs_predicted_pct:+.2f}"
            if r.delivered_vs_predicted_pct is not None else "—"
        )
        lines.append(
            f"| `{r.model_id}` "
            f"| `{r.static_candidate_id[:34]}…` "
            f"| {r.static_latency_min_us:.1f} "
            f"| `{r.oracle_candidate_id[:34]}…` "
            f"| {r.oracle_latency_min_us:.1f} "
            f"| {r.predicted_oracle_min_us:.1f} "
            f"| **{regret}** | {gap} |"
        )
    lines.append("")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--calibration-dir", type=Path, required=True,
        help="Directory containing <model_id>.json calibration tables.",
    )
    p.add_argument("--target", default="host_cpu")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--latency-iters", type=int, default=100)
    p.add_argument("--latency-warmup", type=int, default=10)
    args = p.parse_args(argv)

    if Path.cwd() != PKG_ROOT.resolve():
        os.chdir(PKG_ROOT)

    cal_dir = args.calibration_dir.resolve()
    if not cal_dir.is_dir():
        print(f"FAIL: calibration dir not found: {cal_dir}", file=sys.stderr)
        return 2

    commit = _git_short_commit()
    out_dir = args.out or (
        PKG_ROOT / "results" / "audit" / commit / "oracle_validation"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    target_yaml = PKG_ROOT / "configs" / "targets" / f"{args.target}.yaml"
    if not target_yaml.exists():
        print(f"FAIL: target config not found: {target_yaml}", file=sys.stderr)
        return 2

    model_ids = sorted(
        p.stem for p in cal_dir.glob("*.json")
        if p.is_file() and p.stem != "summary"
    )
    if not model_ids:
        print(f"FAIL: no *.json calibration files in {cal_dir}", file=sys.stderr)
        return 1

    pack = OracleValidationPack(commit=commit, calibration_dir=str(cal_dir))
    for mid in model_ids:
        cfg = PKG_ROOT / "configs" / "models" / f"{mid}.yaml"
        if not cfg.exists():
            print(f"WARN: no config for {mid}; skipping", file=sys.stderr)
            continue
        print(f"validating {mid} ...", file=sys.stderr)
        row = validate_oracle_for_model(
            model_yaml=cfg, target_yaml=target_yaml,
            calibration_dir=cal_dir, out_dir=out_dir / mid,
            n_warmup=args.latency_warmup, n_iters=args.latency_iters,
        )
        pack.rows.append(row)
        if row.delivered_regret_pct is not None:
            print(
                f"  {mid}: static={row.static_latency_min_us:.1f}µs "
                f"oracle={row.oracle_latency_min_us:.1f}µs "
                f"regret={row.delivered_regret_pct:+.2f}%",
                file=sys.stderr,
            )

    emit_pack(pack, out_path=out_dir / "pack.json")
    (out_dir / "summary.md").write_text(_render_summary(pack))
    s = pack.summary()
    print(f"\nresults: {out_dir / 'summary.md'}")
    print(
        f"  {s['model_count']} models | "
        f"mean delivered regret: {s['mean_delivered_regret_pct']:+.2f}% | "
        f"≥2%: {s['models_with_regret_ge_2pct']} | "
        f"≥5%: {s['models_with_regret_ge_5pct']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
