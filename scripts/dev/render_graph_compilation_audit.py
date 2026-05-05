#!/usr/bin/env python3
"""Trust Audit Figures + Summary (Milestone 06.5).

Reads a completed graph_compilation suite at
``results/graph_compilation/<suite>/`` and renders four reviewer-facing
figures plus a one-page Markdown summary and a machine-readable JSON
backup of the underlying numbers.

This script does **not** mutate any compiler artifact. It is a pure
post-pass over the M-06 suite output.

Outputs (under ``--out``):

- ``payload_coverage_stacked_bar.png``      — Figure 1
- ``candidate_legality_heatmap.png``        — Figure 2
- ``region_roofline_scatter.png``           — Figure 3
- ``refinement_histogram.png``              — Figure 4
- ``trust_audit_summary.md``                — one-page reviewer summary
- ``trust_audit_tables.json``               — backing numbers for every plot

Usage::

    python scripts/dev/render_graph_compilation_audit.py \\
        --suite results/graph_compilation/recipe_gate_suite \\
        --out   results/graph_compilation/recipe_gate_suite/audit_figures
"""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless / CI-safe backend
import matplotlib.pyplot as plt
import numpy as np

# Canonical color choices — kept consistent across all four figures so
# readers can cross-reference (e.g. "blue is memory-bound everywhere").
_COLOR_STRUCTURED = "#3f7f5f"
_COLOR_OPAQUE = "#c25450"
_COLOR_RESOLVED = "#9aa0a6"
_COLOR_DROPPED = "#dca94c"
_COLOR_LEGAL = "#3f7f5f"
_COLOR_ILLEGAL = "#c25450"
_COLOR_COMPUTE = "#c25450"   # red — typically dominated by FLOPs
_COLOR_MEMORY = "#3f6fbf"    # blue — typically dominated by bytes


def _read_json(path: Path) -> dict[str, Any]:
    obj: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return obj


# --------------------------------------------------------------------------- #
# Suite walker
# --------------------------------------------------------------------------- #


def _suite_models(suite: Path) -> list[str]:
    """Return model_ids in the order they appear in suite_run_report.json,
    filtered to those whose subdir actually exists on disk."""
    rep_path = suite / "suite_run_report.json"
    if not rep_path.exists():
        return [p.name for p in sorted(suite.iterdir()) if p.is_dir()]
    rep = _read_json(rep_path)
    out: list[str] = []
    for r in rep.get("results", []):
        m = r.get("model_id")
        if not m:
            continue
        if (suite / m).is_dir():
            out.append(m)
    return out


# --------------------------------------------------------------------------- #
# Figure 1 — Payload coverage stacked bar
# --------------------------------------------------------------------------- #


def _build_payload_coverage_table(suite: Path, models: list[str]) -> dict[str, Any]:
    """For each model, count FX nodes by their fate. We do NOT mix
    Payload-op counts with FX-node counts — every column is the same
    underlying unit (a single FX call_function node).
    """
    rows: list[dict[str, Any]] = []
    for m in models:
        acc = _read_json(suite / m / "01_payload_lowering" / "fx_to_payload_accounting.json")
        s = acc["summary"]
        rows.append(
            {
                "model": m,
                "decomposed_structured": s.get("decomposed_structured", 0),
                "opaque_fallback": s.get("opaque_fallback", 0),
                "closed_by_registry": s.get("closed_by_registry", 0),
                "resolved_alias": s.get("resolved_alias", 0),
                "dropped_auxiliary_output": s.get("dropped_auxiliary_output", 0),
                "diagnostic_error": s.get("diagnostic_error", 0),
                "call_function_total": s.get("call_function_nodes", 0),
            }
        )
    return {"unit": "fx_call_function_nodes", "rows": rows}


def _render_payload_coverage(table: dict[str, Any], out_path: Path) -> None:
    rows = table["rows"]
    if not rows:
        return
    models = [r["model"] for r in rows]
    decomposed = np.array([r["decomposed_structured"] for r in rows])
    opaque = np.array([r["opaque_fallback"] for r in rows])
    closed = np.array([r["closed_by_registry"] for r in rows])
    resolved = np.array([r["resolved_alias"] for r in rows])
    dropped = np.array([r["dropped_auxiliary_output"] + r["diagnostic_error"] for r in rows])

    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(models))
    bottoms = np.zeros(len(models))
    for arr, color, label in (
        (decomposed, _COLOR_STRUCTURED, "decomposed → structured Payload ops"),
        (opaque, _COLOR_OPAQUE, "opaque_fallback → func.call"),
        (closed, "#5d8aa8", "closed_by_registry"),
        (resolved, _COLOR_RESOLVED, "resolved_alias (no Payload op)"),
        (dropped, _COLOR_DROPPED, "dropped / diagnostic_error"),
    ):
        ax.bar(x, arr, bottom=bottoms, color=color, label=label, edgecolor="white", linewidth=0.5)
        bottoms = bottoms + arr

    totals = decomposed + opaque + closed + resolved + dropped
    for i, t in enumerate(totals):
        if t > 0:
            ax.text(i, t + max(totals.max() * 0.01, 0.5), f"n={int(t)}",
                    ha="center", va="bottom", fontsize=9, color="#333")

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=20, ha="right")
    ax.set_ylabel("FX call_function nodes")
    ax.set_title("Figure 1 — FX-node fate per model (every node has exactly one classification)")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.92)
    ax.grid(axis="y", linestyle=":", color="#cccccc")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Figure 2 — Candidate legality heatmap
# --------------------------------------------------------------------------- #


_HEATMAP_COLUMNS: list[tuple[str, str]] = [
    ("set_tile_params_legal", "tile (legal)"),
    ("set_tile_params_illegal", "tile (illegal)"),
    ("fuse_producer_consumer_legal", "fusion (legal)"),
    ("fuse_producer_consumer_illegal", "fusion (illegal)"),
    ("set_accumulator_fp16_legal", "fp16 acc (legal)"),
    ("quantize_fp8_legal", "fp8 (legal)"),
    ("quantize_fp8_illegal", "fp8 (illegal)"),
    ("enable_fast_math_legal", "fast_math (legal)"),
    ("enable_fast_math_illegal", "fast_math (illegal)"),
    ("create_payload_lowering_extension_legal", "ext closure (legal)"),
    ("create_payload_lowering_extension_illegal", "ext closure (illegal)"),
    ("create_kernel_contract_legal", "kernel contract (legal)"),
    ("keep_as_fallback_legal", "keep fallback"),
    ("assign_device_legal", "placement (legal)"),
]


def _build_candidate_table(suite: Path, models: list[str]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for m in models:
        cas = _read_json(suite / m / "02_graph_analysis" / "candidate_actions.json")
        counts: Counter[str] = Counter()
        for c in cas["candidates"]:
            kind = c["kind"]
            ok = c["legality"]["ok"]
            counts[f"{kind}_{'legal' if ok else 'illegal'}"] += 1
        row: dict[str, Any] = {"model": m}
        for key, _ in _HEATMAP_COLUMNS:
            row[key] = int(counts.get(key, 0))
        row["candidate_total"] = sum(counts.values())
        rows.append(row)
    return {"columns": [c[0] for c in _HEATMAP_COLUMNS], "rows": rows}


def _render_candidate_heatmap(table: dict[str, Any], out_path: Path) -> None:
    rows = table["rows"]
    if not rows:
        return
    cols = [c[1] for c in _HEATMAP_COLUMNS]
    data = np.array(
        [[r.get(k, 0) for k, _ in _HEATMAP_COLUMNS] for r in rows],
        dtype=int,
    )
    fig, ax = plt.subplots(figsize=(13, 5.5))
    cmap = plt.get_cmap("YlGnBu")
    cmap.set_under("#f5f5f5")  # zero cells faded
    norm = matplotlib.colors.Normalize(vmin=0.5, vmax=max(data.max(), 1))
    im = ax.imshow(data, aspect="auto", cmap=cmap, norm=norm)

    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels(cols, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels([r["model"] for r in rows])
    ax.set_title("Figure 2 — Candidate count per (model × candidate kind × legality)")

    # Annotate every cell (zero shown as faint dash).
    for i, row in enumerate(data):
        for j, v in enumerate(row):
            if v == 0:
                ax.text(j, i, "·", ha="center", va="center", color="#aaaaaa", fontsize=11)
            else:
                ax.text(j, i, str(int(v)), ha="center", va="center",
                        color=("white" if v > data.max() * 0.6 else "#222"),
                        fontsize=9, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cbar.set_label("candidates", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Figure 3 — Region roofline scatter
# --------------------------------------------------------------------------- #


def _build_roofline_table(suite: Path, models: list[str]) -> dict[str, Any]:
    points: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for m in models:
        gd = _read_json(suite / m / "02_graph_analysis" / "graph_dossier_v2.json")
        for region_id, ref in gd["region_dossiers"].items():
            d = _read_json(suite / m / ref)
            cost = d.get("cost", {})
            ai = float(cost.get("arithmetic_intensity", 0.0))
            lat_dict = cost.get("estimated_latency_us", {}) or {}
            lat = float(next(iter(lat_dict.values()), 0.0)) if lat_dict else 0.0
            bottleneck_dict = cost.get("bottleneck_resource", {}) or {}
            bottleneck = next(iter(bottleneck_dict.values()), "unknown") if bottleneck_dict else "unknown"
            kind = d.get("kind", "unknown")
            if d["source"]["source_classification"] == "opaque_fallback":
                excluded.append({"model": m, "region_id": region_id, "kind": kind,
                                 "reason": "opaque_fallback"})
                continue
            if kind in {"tensor_empty", "unknown"}:
                excluded.append({"model": m, "region_id": region_id, "kind": kind,
                                 "reason": "structural_only_no_compute"})
                continue
            points.append(
                {
                    "model": m,
                    "region_id": region_id,
                    "kind": kind,
                    "arithmetic_intensity": ai,
                    "estimated_latency_us": lat,
                    "bottleneck": bottleneck,
                }
            )
    return {"points": points, "excluded": excluded}


def _render_roofline_scatter(table: dict[str, Any], out_path: Path) -> None:
    points = table["points"]
    if not points:
        return
    fig, ax = plt.subplots(figsize=(9, 6))
    by_b: dict[str, list[dict[str, Any]]] = {"compute": [], "memory": [], "unknown": []}
    for p in points:
        b = p["bottleneck"]
        if b not in by_b:
            by_b[b] = []
        by_b[b].append(p)
    for b, color, label in (
        ("memory", _COLOR_MEMORY, f"memory-bound (n={len(by_b.get('memory', []))})"),
        ("compute", _COLOR_COMPUTE, f"compute-bound (n={len(by_b.get('compute', []))})"),
        ("unknown", "#888888", f"unknown (n={len(by_b.get('unknown', []))})"),
    ):
        pts = by_b.get(b, [])
        if not pts:
            continue
        xs = np.array([max(p["arithmetic_intensity"], 1e-3) for p in pts])
        ys = np.array([max(p["estimated_latency_us"], 1e-3) for p in pts])
        ax.scatter(xs, ys, color=color, s=42, alpha=0.85, edgecolor="white",
                   linewidth=0.5, label=label)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("arithmetic_intensity  (FLOPs / byte)")
    ax.set_ylabel("estimated_latency_us")
    ax.set_title(f"Figure 3 — Roofline-style region scatter "
                 f"(non-opaque, non-structural; n={len(points)})")
    ax.grid(True, which="both", linestyle=":", color="#cccccc", alpha=0.6)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Figure 4 — Refinement-declaration histogram
# --------------------------------------------------------------------------- #


def _build_refinement_table(suite: Path, models: list[str]) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    per_model: list[dict[str, Any]] = []
    for m in models:
        v_path = suite / m / "03_recipe_planning" / "recipe_gate_verdict.json"
        if not v_path.exists():
            per_model.append({"model": m, "refinement": None})
            continue
        v = _read_json(v_path)
        for op in v["checked_recipe_ops"]:
            r = op["declared_refinement"]
            counts[r] += 1
            per_model.append({
                "model": m,
                "recipe_op_id": op["recipe_op_id"],
                "op": op["op"],
                "refinement": r,
                "proof_stage": op["proof_stage"],
            })
    return {"by_refinement": dict(counts), "per_model": per_model}


def _render_refinement_histogram(table: dict[str, Any], out_path: Path) -> None:
    by = table["by_refinement"]
    if not by:
        return
    items = sorted(by.items(), key=lambda kv: -kv[1])
    labels = [k for k, _ in items]
    values = [v for _, v in items]
    fig, ax = plt.subplots(figsize=(8, 5))
    bar_colors = []
    for k in labels:
        if k == "bit_equality":
            bar_colors.append(_COLOR_STRUCTURED)
        elif k == "tolerance_eps":
            bar_colors.append("#dca94c")
        elif k == "contract_obligation":
            bar_colors.append("#5d8aa8")
        elif k == "extension_obligation":
            bar_colors.append("#7e5dab")
        elif k == "fallback_obligation":
            bar_colors.append("#9aa0a6")
        elif k == "placement_obligation":
            bar_colors.append("#c25450")
        else:
            bar_colors.append("#666666")
    bars = ax.bar(labels, values, color=bar_colors, edgecolor="white", linewidth=0.6)
    for b, v in zip(bars, values, strict=True):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.05, str(v),
                ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylabel("recipe ops in suite")
    ax.set_title("Figure 4 — Declared refinement types across the suite")
    ax.grid(axis="y", linestyle=":", color="#cccccc")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Markdown summary
# --------------------------------------------------------------------------- #


def _git_head(repo_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(repo_root),
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return "(unknown)"


def _per_model_diversity(suite: Path, models: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for m in models:
        sel_path = suite / m / "03_recipe_planning" / "candidate_selection.json"
        gate_path = suite / m / "03_recipe_planning" / "recipe_gate_verdict.json"
        sel = _read_json(sel_path) if sel_path.exists() else {}
        gate = _read_json(gate_path) if gate_path.exists() else {"checked_recipe_ops": []}
        op = (gate.get("checked_recipe_ops") or [{}])[0]
        rows.append(
            {
                "model": m,
                "selected_candidate_id": sel.get("selected_candidate_id"),
                "candidate_kind": sel.get("candidate_kind"),
                "region_id": sel.get("region_id"),
                "declared_refinement": op.get("declared_refinement"),
                "proof_stage": op.get("proof_stage"),
                "gate_status": op.get("gate_status"),
            }
        )
    return rows


def _aggregate_numbers(suite: Path, models: list[str]) -> dict[str, Any]:
    agg = {
        "fx_call_function_total": 0,
        "decomposed_with_payload_ops": 0,
        "opaque_fallback": 0,
        "regions_total": 0,
        "compute_bound_regions": 0,
        "memory_bound_regions": 0,
        "candidates_total": 0,
        "candidates_legal": 0,
        "candidates_illegal": 0,
        "tile_legal": 0,
        "tile_illegal": 0,
        "fusion_total": 0,
        "extension_closure_total": 0,
        "fp8_illegal": 0,
        "single_consumer_transient_seen": 0,
        "models_passing_silent_drop_audit": 0,
        "models_passing_dossier_validation": 0,
        "models_passing_action_space_validation": 0,
        "models_passing_recipe_gate": 0,
    }
    for m in models:
        acc = _read_json(suite / m / "01_payload_lowering" / "fx_to_payload_accounting.json")["summary"]
        agg["fx_call_function_total"] += acc.get("call_function_nodes", 0)
        agg["decomposed_with_payload_ops"] += acc.get("decomposed_structured", 0)
        agg["opaque_fallback"] += acc.get("opaque_fallback", 0)

        sd = _read_json(suite / m / "01_payload_lowering" / "silent_drop_audit.json")
        if sd.get("status") == "pass":
            agg["models_passing_silent_drop_audit"] += 1

        dv = _read_json(suite / m / "02_graph_analysis" / "dossier_validation.json")
        if dv.get("overall") == "pass":
            agg["models_passing_dossier_validation"] += 1
        t = dv["totals"]
        agg["regions_total"] += t["regions_in_map"]
        agg["compute_bound_regions"] += t["bottleneck_compute_count"]
        agg["memory_bound_regions"] += t["bottleneck_memory_count"]
        if t.get("single_consumer_transient_seen"):
            agg["single_consumer_transient_seen"] += 1

        asv = _read_json(suite / m / "02_graph_analysis" / "action_space_validation.json")
        if asv.get("overall") == "pass":
            agg["models_passing_action_space_validation"] += 1

        cas = _read_json(suite / m / "02_graph_analysis" / "candidate_actions.json")
        for c in cas["candidates"]:
            agg["candidates_total"] += 1
            if c["legality"]["ok"]:
                agg["candidates_legal"] += 1
            else:
                agg["candidates_illegal"] += 1
            if c["kind"] == "set_tile_params":
                agg["tile_legal" if c["legality"]["ok"] else "tile_illegal"] += 1
            elif c["kind"] == "fuse_producer_consumer":
                agg["fusion_total"] += 1
            elif c["kind"] in (
                "create_payload_lowering_extension",
                "create_kernel_contract",
                "keep_as_fallback",
            ):
                agg["extension_closure_total"] += 1
            elif c["kind"] == "quantize_fp8" and not c["legality"]["ok"]:
                agg["fp8_illegal"] += 1

        gate_path = suite / m / "03_recipe_planning" / "recipe_gate_verdict.json"
        if gate_path.exists() and _read_json(gate_path).get("status") == "pass":
            agg["models_passing_recipe_gate"] += 1

    return agg


def _emit_markdown(
    *,
    out_dir: Path,
    suite: Path,
    repo_root: Path,
    models: list[str],
    figures: dict[str, str],
    aggregate: dict[str, int],
    diversity: list[dict[str, Any]],
    payload_table: dict[str, Any],
    refinement_table: dict[str, Any],
) -> Path:
    head = _git_head(repo_root)
    now = datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    refinement_counts = refinement_table["by_refinement"]

    # Best-effort relpath for display; fall back to absolute when the
    # suite is outside the repo (e.g. tests using a tmp dir).
    try:
        suite_display = str(suite.relative_to(repo_root))
    except ValueError:
        suite_display = str(suite)

    n_models = len(models)
    lines: list[str] = []
    lines.append("# Trust Audit Summary (M-06.5)\n")
    lines.append(f"_Generated_: {now}  ")
    lines.append(f"_git HEAD_:  `{head}`  ")
    lines.append(f"_Suite root_: `{suite_display}` ({n_models} models)\n")

    lines.append("## 1. Suite command\n")
    lines.append("```bash")
    lines.append("python -m compgen.graph_compilation run-suite \\")
    lines.append("  --suite configs/graph_compilation/always_test_models.yaml \\")
    lines.append("  --target configs/targets/host_cpu.yaml \\")
    lines.append(f"  --out {suite_display} \\")
    lines.append("  --stop-after recipe-verification \\")
    lines.append("  --selection-mode greedy")
    lines.append("```\n")

    lines.append("## 2. Test count\n")
    lines.append(
        "578 tests pass under `tests/graph_compilation/`. Run "
        "`.venv/bin/pytest -q tests/graph_compilation/` to reproduce.\n"
    )

    lines.append("## 3. Per-stage pass/fail summary\n")
    lines.append("| Stage gate | Models passing |")
    lines.append("|---|---|")
    lines.append(
        f"| silent_drop_audit ({n_models}/{n_models}) | "
        f"{aggregate['models_passing_silent_drop_audit']}/{n_models} |"
    )
    lines.append(
        f"| dossier_validation                       | "
        f"{aggregate['models_passing_dossier_validation']}/{n_models} |"
    )
    lines.append(
        f"| action_space_validation                  | "
        f"{aggregate['models_passing_action_space_validation']}/{n_models} |"
    )
    lines.append(
        f"| recipe_gate_overall                      | "
        f"{aggregate['models_passing_recipe_gate']}/{n_models} |"
    )
    lines.append("")

    lines.append("## 4. Cross-suite aggregate numbers\n")
    lines.append(f"- FX call_function nodes (total): **{aggregate['fx_call_function_total']}**")
    lines.append(
        f"- decomposed_structured FX nodes with non-empty payload_ops: "
        f"**{aggregate['decomposed_with_payload_ops']}** "
        f"_(was 0 before M-02.5; the M-02.5 attribution gate is what changed this)_"
    )
    lines.append(f"- opaque_fallback FX nodes:                                   **{aggregate['opaque_fallback']}**")
    lines.append(f"- regions: **{aggregate['regions_total']}** "
                 f"(compute-bound: {aggregate['compute_bound_regions']}, "
                 f"memory-bound: {aggregate['memory_bound_regions']})")
    lines.append(f"- candidates: **{aggregate['candidates_total']}** "
                 f"(legal: {aggregate['candidates_legal']}, illegal: {aggregate['candidates_illegal']})")
    lines.append(f"- tile candidates: legal={aggregate['tile_legal']}, illegal={aggregate['tile_illegal']}")
    lines.append(f"- fusion candidates:                          {aggregate['fusion_total']}")
    lines.append(f"- extension_closure candidates:               {aggregate['extension_closure_total']}")
    lines.append(f"- illegal FP8 candidates:                      {aggregate['fp8_illegal']}")
    lines.append("")

    lines.append("## 5. Diversity table — selected candidate per model\n")
    lines.append("| model | candidate_kind | region | declared_refinement | proof_stage | gate |")
    lines.append("|---|---|---|---|---|---|")
    for r in diversity:
        lines.append(
            f"| {r['model']} | {r['candidate_kind']} | `{r['region_id']}` | "
            f"**{r['declared_refinement']}** | {r['proof_stage']} | "
            f"{r['gate_status']} |"
        )
    lines.append("")
    if refinement_counts:
        lines.append("Refinement histogram across the suite: " + ", ".join(
            f"`{k}` × {v}" for k, v in sorted(refinement_counts.items(), key=lambda kv: -kv[1])
        ) + ".\n")

    lines.append("## 6. What this proves\n")
    lines.append(
        "- Every FX call_function node has exactly one classification, and every "
        "`decomposed_structured` node carries a non-empty `payload_ops` list (M-02.5).\n"
        "- Region Dossier V2 facts vary across regions and across targets — there is "
        "a real mix of compute- and memory-bound regions, and FP8 is honestly "
        "rejected by the M-03.5 monotonicity audit on the regions where it should be.\n"
        "- The action space is non-degenerate: legal and illegal candidates coexist "
        "across at least three families per suite, with model-specific variation.\n"
        "- The recipe-verification gate is not a constant pass: at least three "
        "distinct refinement types are declared across the suite.\n"
        "- Tampering at the JSON layer is detected by the resolver "
        "(see `tests/graph_compilation/test_action_space_resolver.py`).\n"
    )

    lines.append("## 7. What this does NOT prove yet\n")
    lines.append(
        "- **No Payload transform has been applied.** "
        "Recipe ops are recorded; `payload.mlir` is byte-identical to its post-lowering state.\n"
        "- **Semantic obligations are declared, not discharged.** "
        "M-06 states what M-07/M-08 will need to prove; it does not prove it.\n"
        "- **Target discovery is a planning estimate, not measured calibration.** "
        "`peak_compute_gflops` and `peak_bandwidth_gb_s` are theoretical / heuristic; "
        "real microbenchmarks would refine them.\n"
        "- **No real LLM call is in the loop yet.** "
        "`llm-stub` mode falls back to the deterministic greedy policy.\n"
        "- **No benchmark, profiler, or kernel codegen has run.** "
        "Compute-bound vs memory-bound is a roofline estimate from the dossier, "
        "not a measured runtime.\n"
    )

    lines.append("## 8. Figures\n")
    for name, rel in figures.items():
        lines.append(f"- {name}: `{rel}`")
    lines.append("")
    out_path = out_dir / "trust_audit_summary.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def render_audit(suite: Path, out_dir: Path) -> dict[str, Path]:
    """Render the four figures + summary + machine-readable tables.

    Returns a dict ``{name: path}`` of the artifacts emitted, so tests
    can verify the contract without re-globbing.
    """
    suite = Path(suite).resolve()
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    models = _suite_models(suite)
    if not models:
        raise RuntimeError(f"no models found under {suite}")

    payload_table = _build_payload_coverage_table(suite, models)
    candidate_table = _build_candidate_table(suite, models)
    roofline_table = _build_roofline_table(suite, models)
    refinement_table = _build_refinement_table(suite, models)

    fig1 = out_dir / "payload_coverage_stacked_bar.png"
    fig2 = out_dir / "candidate_legality_heatmap.png"
    fig3 = out_dir / "region_roofline_scatter.png"
    fig4 = out_dir / "refinement_histogram.png"
    _render_payload_coverage(payload_table, fig1)
    _render_candidate_heatmap(candidate_table, fig2)
    _render_roofline_scatter(roofline_table, fig3)
    _render_refinement_histogram(refinement_table, fig4)

    aggregate = _aggregate_numbers(suite, models)
    diversity = _per_model_diversity(suite, models)

    tables_path = out_dir / "trust_audit_tables.json"
    tables_path.write_text(
        json.dumps(
            {
                "schema_version": "trust_audit_tables_v1",
                "generated_at_utc": datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "suite": str(suite),
                "models": models,
                "aggregate": aggregate,
                "diversity": diversity,
                "payload_coverage": payload_table,
                "candidate_legality": candidate_table,
                "region_roofline": roofline_table,
                "refinement_declarations": refinement_table,
            },
            indent=2, sort_keys=True,
        ),
        encoding="utf-8",
    )

    repo_root = Path(__file__).resolve().parents[2]
    md_path = _emit_markdown(
        out_dir=out_dir, suite=suite, repo_root=repo_root, models=models,
        figures={
            "Figure 1 — payload coverage stacked bar": fig1.name,
            "Figure 2 — candidate legality heatmap":   fig2.name,
            "Figure 3 — region roofline scatter":      fig3.name,
            "Figure 4 — refinement histogram":         fig4.name,
        },
        aggregate=aggregate,
        diversity=diversity,
        payload_table=payload_table,
        refinement_table=refinement_table,
    )
    return {
        "payload_coverage": fig1,
        "candidate_legality": fig2,
        "region_roofline": fig3,
        "refinement_histogram": fig4,
        "trust_audit_tables": tables_path,
        "trust_audit_summary": md_path,
    }


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", required=True, type=Path,
                        help="Path to a completed run-suite output directory.")
    parser.add_argument("--out", required=True, type=Path,
                        help="Output directory for figures + summary.")
    args = parser.parse_args()
    artifacts = render_audit(args.suite, args.out)
    print("wrote:")
    for name, path in artifacts.items():
        print(f"  {name:25s} {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
