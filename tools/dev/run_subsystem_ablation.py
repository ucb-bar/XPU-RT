#!/usr/bin/env python
"""Subsystem prove-or-kill ablation CLI.

Runs the graph-compilation pipeline twice per model — once with every
subsystem on (control), once with the named flag flipped off
(treatment) — and emits a typed diff pack. Use this to make the
kill/keep decision for any single subsystem flag named in
`SubsystemMask`.

Usage::

    uv run python scripts/dev/run_subsystem_ablation.py \\
        --subsystem kernels.codegen_fallback \\
        --models merlin_mlp_wide proxy_vla \\
        --target host_cpu

Subcommand `calibrate-noise` runs the all-on mask N times per model
to populate the per-model decision-seconds stddev used by the kill
rule's "below noise floor" check::

    uv run python scripts/dev/run_subsystem_ablation.py calibrate-noise \\
        --repeats 3

Outputs land at ``results/audit/<commit>/subsystem_<flag>/`` (or
``noise_floor/`` for calibration), with both ``rows.jsonl`` /
``report.md`` and a ``subsystem_mask.json`` sidecar inside every cell
run directory.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from xpu_rt.benchmarks.subsystem_ablation import (
    SubsystemAblationPack,
    SubsystemAblationSpec,
    calibrate_noise_floor,
    emit_pack,
    run_subsystem_ablation,
)
from xpu_rt.benchmarks.subsystem_mask import SubsystemMask

REPO_ROOT = Path(__file__).resolve().parents[2]
# Merged-repo layout: configs/ + models/ live under xpu-rt/, not the
# top-level repo root.
PKG_ROOT = REPO_ROOT / "xpu-rt"

DEFAULT_MODELS = (
    "merlin_mlp_wide",
    "proxy_vla",
    "holdout_mlp_odd_shapes",
    "holdout_mlp_large_k",
    "holdout_pointwise_chain_renamed",
    "holdout_two_matmuls_shared_input",
)

# Full prove-or-kill set: 8 small canonical + 5 holdouts. All known
# to have working factories under tests/graph_compilation/models/.
# Bigger LLMs/VLMs (llama4, deepseek, qwen, gemma, etc.) are
# deliberately excluded — host_cpu capture would take prohibitively
# long and is not the workload the prove-or-kill rule targets.
FULL_MODELS = (
    "tiny_mlp",
    "tiny_conv_block",
    "tiny_attention",
    "merlin_mlp",
    "merlin_mlp_wide",
    "merlin_dronet",
    "proxy_vla",
    "graph_break_mlp",
    "holdout_mlp_odd_shapes",
    "holdout_mlp_large_k",
    "holdout_pointwise_chain_renamed",
    "holdout_two_matmuls_shared_input",
    "holdout_unsupported_attention",
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


def _resolve_models(model_ids: list[str]) -> list[Path]:
    paths: list[Path] = []
    for mid in model_ids:
        p = PKG_ROOT / "configs" / "models" / f"{mid}.yaml"
        if not p.exists():
            print(f"WARN: skipping {mid} (no config at {p})", file=sys.stderr)
            continue
        paths.append(p)
    return paths


def _render_markdown(pack: SubsystemAblationPack) -> str:
    summary = pack.summary()
    lines: list[str] = []
    lines.append(f"# Subsystem ablation — `{pack.subsystem_flag}` @ {pack.commit}")
    lines.append("")
    lines.append(f"_Generated: `{pack.generated_at_utc}`_")
    lines.append("")
    lines.append(
        f"**{summary['row_count']} models** — "
        f"{summary['candidate_changed_count']} changed candidate, "
        f"{summary['outcome_changed_count']} changed outcome."
    )
    lines.append("")
    lines.append(
        "| Side | verified | verification_fail | typed_blocked | error |"
    )
    lines.append("| --- | --- | --- | --- | --- |")
    lines.append(
        f"| control | {summary['control_verified_count']} | "
        f"{summary['control_verification_fail_count']} | "
        f"{summary['control_typed_blocked_count']} | "
        f"{summary['control_error_count']} |"
    )
    lines.append(
        f"| treatment | {summary['treatment_verified_count']} | "
        f"{summary['treatment_verification_fail_count']} | "
        f"{summary['treatment_typed_blocked_count']} | "
        f"{summary['treatment_error_count']} |"
    )
    lines.append("")
    lines.append(
        f"**Codegen-success uplift** (control − treatment): "
        f"{summary['codegen_success_uplift']:+d} model(s). "
        f"Mean Δ decision: {summary['mean_decision_seconds_delta']:+.3f}s."
    )
    lines.append("")
    if summary["latency_measured_count"] > 0:
        lines.append(
            f"**Latency — min-based (primary kill signal):** "
            f"{summary['latency_measured_count']} measured, "
            f"**{summary['control_speedup_min_ge5pct_count']}** models "
            f"≥5% faster with subsystem ON, "
            f"**{summary['control_speedup_min_le_neg5pct_count']}** models "
            f"≥5% slower with subsystem ON. "
            f"Median-of-mins speedup: "
            f"{summary['median_control_speedup_min_pct']:+.2f}%."
        )
        lines.append("")
        lines.append(
            f"**Latency — median-based (context only):** "
            f"{summary['control_speedup_ge5pct_count']} ≥5%-faster, "
            f"{summary['control_speedup_le_neg5pct_count']} ≥5%-slower, "
            f"median-of-medians: "
            f"{summary['median_control_speedup_pct']:+.2f}%. "
            f"**{summary['noise_divergent_row_count']}** rows have "
            f"|Δmedian − Δmin| > 5pp — median is unreliable on those."
        )
        lines.append("")
    lines.append("## Per-model rows")
    lines.append("")
    lines.append(
        "| Model | Control pick | Treatment pick | Outcome (c/t) | "
        "Min µs (c/t) | **Δmin %** | Δmed % | σ (c/t) | Noise? |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for r in pack.rows:
        cl, tl = r.control.latency, r.treatment.latency
        if cl and tl and r.control_speedup_min_pct is not None:
            lat_cell = f"{cl.latency_min_us:.1f} / {tl.latency_min_us:.1f}"
            min_pct = f"**{r.control_speedup_min_pct:+.2f}**"
            med_pct = (
                f"{r.control_speedup_pct:+.2f}"
                if r.control_speedup_pct is not None else "—"
            )
            sigma = f"{cl.latency_stddev_us:.1f} / {tl.latency_stddev_us:.1f}"
            div = r.latency_noise_divergence_pp
            noise = "⚠️ " + f"{div:.1f}pp" if div is not None and div > 5.0 else ""
        else:
            lat_cell = "—"
            min_pct = "—"
            med_pct = "—"
            sigma = "—"
            noise = ""
        lines.append(
            f"| `{r.model_id}` "
            f"| `{r.control.result.selected_candidate_id or '(none)'}` "
            f"| `{r.treatment.result.selected_candidate_id or '(none)'}` "
            f"| {r.control.result.typed_outcome}/{r.treatment.result.typed_outcome} "
            f"| {lat_cell} | {min_pct} | {med_pct} | {sigma} | {noise} |"
        )
    lines.append("")
    return "\n".join(lines) + "\n"


def _emit_rows_jsonl(pack: SubsystemAblationPack, *, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in pack.rows:
            f.write(json.dumps(r.to_dict(), sort_keys=True) + "\n")
    return out_path


def _cmd_ablate(args: argparse.Namespace) -> int:
    commit = _git_short_commit()
    flag = args.subsystem
    if flag not in SubsystemMask.flag_names():
        print(
            f"FAIL: unknown subsystem flag {flag!r}. Known flags:\n  "
            + "\n  ".join(SubsystemMask.flag_names()),
            file=sys.stderr,
        )
        return 2

    out_dir = args.out or (
        PKG_ROOT / "results" / "audit" / commit / f"subsystem_{flag.replace('.', '_')}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    target_yaml = PKG_ROOT / "configs" / "targets" / f"{args.target}.yaml"
    if not target_yaml.exists():
        print(f"FAIL: target config not found: {target_yaml}", file=sys.stderr)
        return 2

    requested = (
        list(args.models) if args.models
        else list(FULL_MODELS) if getattr(args, "full", False)
        else list(DEFAULT_MODELS)
    )
    model_paths = _resolve_models(requested)
    if not model_paths:
        print("FAIL: no models resolved", file=sys.stderr)
        return 1

    specs = [
        SubsystemAblationSpec(
            model_yaml=p,
            target_yaml=target_yaml,
            subsystem_flag=flag,
            mode=args.mode,
            stop_after=args.stop_after,
            latency_iters=args.latency_iters,
            latency_warmup=args.latency_warmup,
        )
        for p in model_paths
    ]

    print(
        f"running subsystem ablation: flag={flag} models={len(specs)} "
        f"target={args.target} mode={args.mode} stop_after={args.stop_after}",
        file=sys.stderr,
    )
    pack = run_subsystem_ablation(
        specs, out_root=out_dir / "runs", commit=commit, subsystem_flag=flag,
    )

    json_path = out_dir / "pack.json"
    rows_path = out_dir / "rows.jsonl"
    md_path = out_dir / "report.md"
    emit_pack(pack, out_path=json_path)
    _emit_rows_jsonl(pack, out_path=rows_path)
    md_path.write_text(_render_markdown(pack), encoding="utf-8")

    s = pack.summary()
    print(f"\nresults: {json_path}")
    print(f"         {rows_path}")
    print(f"         {md_path}")
    print(
        f"  {s['row_count']} rows | "
        f"candidate changed: {s['candidate_changed_count']} | "
        f"outcome changed: {s['outcome_changed_count']} | "
        f"codegen-success uplift (ctrl−trt): "
        f"{s['codegen_success_uplift']:+d} | "
        f"verified ctrl/trt: "
        f"{s['control_verified_count']}/{s['treatment_verified_count']} | "
        f"mean Δ decision: {s['mean_decision_seconds_delta']:+.3f}s",
    )
    if s["latency_measured_count"] > 0:
        print(
            f"  latency measured: {s['latency_measured_count']} | "
            f"min-based ≥5% faster: {s['control_speedup_min_ge5pct_count']} | "
            f"min-based ≥5% slower: "
            f"{s['control_speedup_min_le_neg5pct_count']} | "
            f"median-of-mins: {s['median_control_speedup_min_pct']:+.2f}% | "
            f"noisy rows (|Δmed−Δmin|>5pp): "
            f"{s['noise_divergent_row_count']}",
        )
    return 0


def _cmd_calibrate_noise(args: argparse.Namespace) -> int:
    commit = _git_short_commit()
    out_dir = args.out or (PKG_ROOT / "results" / "audit" / commit / "noise_floor")
    out_dir.mkdir(parents=True, exist_ok=True)

    target_yaml = PKG_ROOT / "configs" / "targets" / f"{args.target}.yaml"
    if not target_yaml.exists():
        print(f"FAIL: target config not found: {target_yaml}", file=sys.stderr)
        return 2

    model_paths = _resolve_models(list(args.models or DEFAULT_MODELS))
    if not model_paths:
        print("FAIL: no models resolved", file=sys.stderr)
        return 1

    print(
        f"calibrating noise floor: models={len(model_paths)} "
        f"repeats={args.repeats} target={args.target}",
        file=sys.stderr,
    )
    entries = calibrate_noise_floor(
        model_paths, target_yaml,
        out_root=out_dir / "runs", n_repeats=args.repeats, mode=args.mode,
    )
    summary_path = out_dir / "runs" / "noise_floor.json"
    print(f"\nresults: {summary_path}")
    for e in entries:
        print(
            f"  {e.model_id}: mean={e.mean_decision_seconds:.3f}s "
            f"stddev={e.stddev_decision_seconds:.3f}s (n={e.n_repeats})"
        )
    return 0


def _anchor_cwd_to_pkg_root() -> None:
    """Chdir to xpu-rt/ so model YAMLs' relative `model_path` resolves.

    Model configs store `model_path` as a path relative to the
    xpu-rt/ package root (e.g. `tests/graph_compilation/models/<x>.py`).
    `_load_model_factory` resolves that against cwd, so the harness
    must run with cwd anchored at xpu-rt/. Callers anywhere in the
    merged repo root see the same behavior.
    """
    if Path.cwd() != PKG_ROOT.resolve():
        os.chdir(PKG_ROOT)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=False)

    # Default verb: ablate. Kept at top-level for ergonomics.
    p.add_argument("--subsystem", default=None,
                   help="Subsystem flag to flip off (dotted form, e.g. "
                        "kernels.codegen_fallback). Required for default verb.")
    p.add_argument("--models", nargs="+", default=None,
                   help="Model ids to run (default: 6 canonical+holdout).")
    p.add_argument("--full", action="store_true",
                   help="Use the full 13-model prove-or-kill set (8 small "
                        "canonical + 5 holdouts). Overridden by --models.")
    p.add_argument("--target", default="host_cpu",
                   help="Target id (default: host_cpu).")
    p.add_argument("--mode", default="greedy",
                   help="Selection mode passed to run_graph_compilation "
                        "(default: greedy).")
    p.add_argument("--stop-after", default="agent-decision-request",
                   help="Pipeline stop_after stage (default: "
                        "agent-decision-request; use "
                        "post-lowering-verification for codegen-success "
                        "signal).")
    p.add_argument("--latency-iters", type=int, default=0,
                   help="If >0, run the CPU latency probe on each cell's "
                        "transformed_payload.mlir with this many timed "
                        "iterations (after --latency-warmup warmup runs). "
                        "Requires --stop-after to reach post-lowering-verification.")
    p.add_argument("--latency-warmup", type=int, default=5,
                   help="Warmup iterations before timing (default: 5).")
    p.add_argument("--out", type=Path, default=None,
                   help="Output dir (default: results/audit/<commit>/subsystem_<flag>/).")

    sub_noise = sub.add_parser("calibrate-noise",
                               help="Run the all-on mask N times per model "
                                    "and record decision-seconds stddev.")
    sub_noise.add_argument("--repeats", type=int, default=3)
    sub_noise.add_argument("--models", nargs="+", default=None)
    sub_noise.add_argument("--target", default="host_cpu")
    sub_noise.add_argument("--mode", default="greedy")
    sub_noise.add_argument("--out", type=Path, default=None)

    args = p.parse_args(argv)
    _anchor_cwd_to_pkg_root()
    if args.cmd == "calibrate-noise":
        return _cmd_calibrate_noise(args)
    if not args.subsystem:
        p.error("--subsystem is required (or use the `calibrate-noise` subcommand)")
    return _cmd_ablate(args)


if __name__ == "__main__":
    raise SystemExit(main())
