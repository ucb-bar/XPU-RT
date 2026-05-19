"""Exp 20 — The headline composite "optimization path" figure (v1/v2/v3).

Aggregates the v1 (per-backend), v2 (per-workload base), and v3
(per-workload base × per-workload contention) calibration results from
exp16/exp18/exp19 into a single 2x2 plot. Shows where prediction error
lives, how the per-workload split changed the overhead constants, and
how the multi-rate recommendation moves under each calibration.

Outputs (under ``build/experiments/exp20_optimization_path/``):

  * ``optimization_path.png``  — 2x2 composite headline figure.
  * ``trajectory.png``         — per-workload arc with v3 point added.
  * ``summary.md``             — annotation key + 4-condition headline
                                  table + verdict paragraph.
  * ``results.jsonl``          — numbers behind every panel,
                                  schema_version = ``exp20_path_v2``.

Usage:
    uv run python scripts/experiments/exp20_optimization_path.py

This script reads the v1/v2 archives saved alongside each experiment's
output (``*.v{1,2}_{per_backend,per_workload}.{md,jsonl,json}``) and
the freshly-regenerated v3 outputs. It does NOT re-run the calibration
pipeline; if v1 archives are missing, the script falls back to the
analytical numbers cited in the spec.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "xpu-rt" / "data"
COST_MATRIX_PATH = DATA_ROOT / "profiled" / "qnn_cost_matrix.json"
CALIBRATION_PATH = DATA_ROOT / "calibration" / "qrb5165.json"
E2E_PATH = DATA_ROOT / "profiled" / "qnn_e2e" / "measurements.json"

EXP18_DIR = REPO_ROOT / "build" / "experiments" / "exp18_proof"
EXP19_DIR = REPO_ROOT / "build" / "experiments" / "exp19_multi_rate"
EXP16_DIR = REPO_ROOT / "build" / "experiments" / "exp16_before_after"

OUT_DIR = REPO_ROOT / "build" / "experiments" / "exp20_optimization_path"
COMPOSITE_PATH = OUT_DIR / "optimization_path.png"
TRAJECTORY_PATH = OUT_DIR / "trajectory.png"
SUMMARY_PATH = OUT_DIR / "summary.md"
RESULTS_PATH = OUT_DIR / "results.jsonl"

WORKLOADS: tuple[str, ...] = ("yolov8n", "dronet")
BACKENDS: tuple[str, ...] = ("CPU", "GPU", "DSP")
CLOSED_LOOP_HARDCODED_DRONET = 12

sys.path.insert(0, str(REPO_ROOT / "xpu-rt" / "python"))


def _require_matplotlib() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: F401

    return matplotlib


def _chain_sum_us(cost_matrix: dict, workload: str, backend: str) -> float:
    total = 0.0
    for _op, lanes in cost_matrix[workload].items():
        if not isinstance(lanes, dict):
            continue
        if backend in lanes:
            total += float(lanes[backend])
    return total


def _per_workload_pct_errors(
    cost_matrix: dict,
    e2e: dict,
    overhead_v1: dict[str, float],
    overhead_v2: dict[str, dict[str, float]],
    contention_v3: dict[str, dict[str, float]],
) -> dict[str, dict[str, float]]:
    """Compute mean abs % solo-E2E error per (workload, condition).

    v3 multiplies the v2 base by ``contention[w][b]`` (default 1.0),
    so cells with no contended data look identical to v2.
    """
    out: dict[str, dict[str, float]] = {w: {} for w in WORKLOADS}
    for w in WORKLOADS:
        chain_errs: list[float] = []
        v1_errs: list[float] = []
        v2_errs: list[float] = []
        v3_errs: list[float] = []
        for b in BACKENDS:
            chain_ms = _chain_sum_us(cost_matrix, w, b) / 1000.0
            measured_ms = float(e2e["matrix"][w][b]["mean_us"]) / 1000.0
            if measured_ms <= 0.0:
                continue
            chain_errs.append(abs(chain_ms - measured_ms) / measured_ms * 100.0)
            v1_pred = chain_ms + overhead_v1.get(b, 0.0) / 1000.0
            v1_errs.append(abs(v1_pred - measured_ms) / measured_ms * 100.0)
            v2_o = overhead_v2.get(w, {}).get(b, 0.0)
            v2_pred = chain_ms + v2_o / 1000.0
            v2_errs.append(abs(v2_pred - measured_ms) / measured_ms * 100.0)
            cf = float(contention_v3.get(w, {}).get(b, 1.0))
            v3_pred = v2_pred * cf
            v3_errs.append(abs(v3_pred - measured_ms) / measured_ms * 100.0)
        out[w]["chain_sum"] = sum(chain_errs) / max(len(chain_errs), 1)
        out[w]["v1_per_backend"] = sum(v1_errs) / max(len(v1_errs), 1)
        out[w]["v2_per_workload"] = sum(v2_errs) / max(len(v2_errs), 1)
        out[w]["v3_two_term"] = sum(v3_errs) / max(len(v3_errs), 1)
    return out


def _closed_loop_pct_errors(
    cost_matrix: dict,
    overhead_v1: dict[str, float],
    overhead_v2: dict[str, dict[str, float]],
    contention_v3: dict[str, dict[str, float]],
    measured_per_iter_ms: list[float],
    workload: str = "yolov8n",
    backend: str = "DSP",
) -> dict[str, float]:
    """Per-condition mean abs % error against the 4 closed-loop rounds."""
    chain_ms = _chain_sum_us(cost_matrix, workload, backend) / 1000.0
    v1_pred = chain_ms + overhead_v1.get(backend, 0.0) / 1000.0
    v2_pred = chain_ms + overhead_v2.get(workload, {}).get(backend, 0.0) / 1000.0
    cf = float(contention_v3.get(workload, {}).get(backend, 1.0))
    v3_pred = v2_pred * cf

    def mean_abs_err(pred_ms: float) -> float:
        errs = [abs(pred_ms - m) / m * 100.0 for m in measured_per_iter_ms if m > 0]
        return sum(errs) / max(len(errs), 1)

    return {
        "chain_sum": mean_abs_err(chain_ms),
        "v1_per_backend": mean_abs_err(v1_pred),
        "v2_per_workload": mean_abs_err(v2_pred),
        "v3_two_term": mean_abs_err(v3_pred),
    }


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _read_v1_overhead_from_archive() -> dict[str, float] | None:
    md = EXP18_DIR / "summary.v1_per_backend.md"
    if not md.exists():
        return None
    overhead: dict[str, float] = {}
    table_re = re.compile(
        r"\|\s*(yolov8n|dronet)\s*\|\s*(CPU|GPU|DSP)\s*\|"
        r"\s*[\d.]+\s*\|\s*([\d.]+)\s*\|"
    )
    for line in md.read_text().splitlines():
        m = table_re.search(line)
        if not m:
            continue
        _w, b, oh_ms = m.group(1), m.group(2), float(m.group(3))
        overhead[b] = oh_ms * 1000.0
    if set(overhead) >= {"CPU", "GPU", "DSP"}:
        return overhead
    return None


def _read_multi_rate_recommendation(json_path: Path) -> int | None:
    if not json_path.exists():
        return None
    payload = json.loads(json_path.read_text())
    for r in payload.get("rates", []):
        if r.get("workload_id") == "dronet":
            return int(r.get("multiplicity", 0))
    return None


def _read_random_control(jsonl_path: Path) -> dict[str, float] | None:
    if not jsonl_path.exists():
        return None
    rows = _load_jsonl(jsonl_path)
    for row in rows:
        if row.get("test") == "D_random_overhead_control":
            return {
                "seeded_mean_abs_pct": float(row["seeded_mean_abs_pct"]),
                "fraction_random_beats_seeded": float(row["fraction_random_beats_seeded"]),
                "n_trials": int(row["n_trials"]),
                "seeded_overhead_dsp_us": float(row["seeded_overhead_dsp_us"]),
                "max_overhead_us_seed": float(row["max_overhead_us_seed"]),
            }
    return None


def _percentile_of_seed(frac_better: float) -> float:
    return (1.0 - frac_better) * 100.0


def _measured_per_iter_ms_from_exp16() -> list[float]:
    """Read the 4 measured per-iter ms from exp16's results.jsonl (D rows)."""
    path = EXP16_DIR / "results.jsonl"
    if not path.exists():
        # Fall back to hard-coded round table from final_report.md.
        return [254.8, 350.9, 255.6, 257.3]
    seen: dict[int, float] = {}
    for row in _load_jsonl(path):
        rnd = int(row.get("round", 0))
        meas = float(row.get("measured_ms", 0.0))
        seen.setdefault(rnd, meas)
    return [seen[k] for k in sorted(seen)]


def _gather_payload() -> dict[str, Any]:
    cost_matrix = json.loads(COST_MATRIX_PATH.read_text())
    e2e = json.loads(E2E_PATH.read_text())

    v3_cal = json.loads(CALIBRATION_PATH.read_text())
    v3_overhead = {
        w: {b: float(v) for b, v in per_b.items()}
        for w, per_b in v3_cal["overhead_us"].items()
    }
    v3_contention = {
        w: {b: float(v) for b, v in per_b.items()}
        for w, per_b in v3_cal.get("contention_factor", {}).items()
    }
    v3_provenance = v3_cal.get("contention_provenance", {})

    v1_overhead = _read_v1_overhead_from_archive()
    if v1_overhead is None:
        v1_overhead = {"CPU": 28_715.46, "DSP": 226_025.44, "GPU": 186_749.32}

    per_workload = _per_workload_pct_errors(
        cost_matrix, e2e, v1_overhead, v3_overhead, v3_contention,
    )

    closed_loop_meas = _measured_per_iter_ms_from_exp16()
    closed_loop_err = _closed_loop_pct_errors(
        cost_matrix, v1_overhead, v3_overhead, v3_contention, closed_loop_meas,
    )

    rec_v1 = _read_multi_rate_recommendation(
        EXP19_DIR / "results.v1_per_backend.json"
    )
    rec_v2 = _read_multi_rate_recommendation(
        EXP19_DIR / "results.v2_per_workload.json"
    )
    rec_v3 = _read_multi_rate_recommendation(EXP19_DIR / "results.json")

    rand_v1 = _read_random_control(EXP18_DIR / "results.v1_per_backend.jsonl")
    rand_v2 = _read_random_control(EXP18_DIR / "results.v2_per_workload.jsonl")
    rand_v3 = _read_random_control(EXP18_DIR / "results.jsonl")

    return {
        "v1_overhead_us": v1_overhead,
        "v2_overhead_us": v3_overhead,  # v3 base term equals v2 base by construction
        "v3_overhead_us": v3_overhead,
        "v3_contention_factor": v3_contention,
        "v3_contention_provenance": v3_provenance,
        "per_workload_errors": per_workload,
        "closed_loop_errors": closed_loop_err,
        "closed_loop_measured_ms": closed_loop_meas,
        "multi_rate": {
            "closed_loop_hand_tuned": CLOSED_LOOP_HARDCODED_DRONET,
            "v1": rec_v1,
            "v2": rec_v2,
            "v3": rec_v3,
        },
        "random_control": {"v1": rand_v1, "v2": rand_v2, "v3": rand_v3},
    }


def _plot_composite(payload: dict[str, Any]) -> None:
    plt = sys.modules["matplotlib.pyplot"]
    import numpy as np

    fig, axes = plt.subplots(2, 2, figsize=(16.0, 11.0))

    # ---- Top-left: per-workload error bars across 4 conditions ----------
    ax = axes[0][0]
    conds = ("chain_sum", "v1_per_backend", "v2_per_workload", "v3_two_term")
    labels = ("chain_sum", "v1 per-backend", "v2 per-workload", "v3 two-term")
    workload_colors = {"yolov8n": "#cc1f1f", "dronet": "#1f7fcc"}
    n_groups = len(conds)
    width = 0.38
    x = np.arange(n_groups, dtype=float)
    for i, w in enumerate(WORKLOADS):
        vals = [payload["per_workload_errors"][w][c] for c in conds]
        bars = ax.bar(
            x + (i - 0.5) * width, vals, width,
            label=w, color=workload_colors[w],
            edgecolor="black", linewidth=0.4,
        )
        for rect, v in zip(bars, vals, strict=True):
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                rect.get_height() + 2.0,
                f"{v:.0f}%",
                ha="center", va="bottom", fontsize=9,
            )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Mean abs % prediction error (vs solo E2E)")
    ax.set_title("Top-left: per-workload solo prediction error per fix")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    max_v = max(
        payload["per_workload_errors"][w][c]
        for w in WORKLOADS for c in conds
    )
    ax.set_ylim(0, max_v * 1.18)

    # ---- Top-right: stacked overhead + contention heatmaps --------------
    ax = axes[0][1]
    ax.axis("off")  # We'll lay out two sub-axes manually for paired heatmaps.
    sub_overhead = fig.add_axes([0.55, 0.62, 0.20, 0.27])
    sub_contention = fig.add_axes([0.78, 0.62, 0.18, 0.27])

    grid_o = np.zeros((len(WORKLOADS), len(BACKENDS)))
    grid_c = np.zeros((len(WORKLOADS), len(BACKENDS)))
    for r, w in enumerate(WORKLOADS):
        for c, b in enumerate(BACKENDS):
            grid_o[r, c] = payload["v3_overhead_us"][w].get(b, 0.0) / 1000.0
            grid_c[r, c] = float(payload["v3_contention_factor"].get(w, {}).get(b, 1.0))

    im_o = sub_overhead.imshow(grid_o, cmap="YlOrRd", aspect="auto", vmin=0, vmax=300)
    sub_overhead.set_xticks(range(len(BACKENDS)))
    sub_overhead.set_xticklabels(BACKENDS, fontsize=9)
    sub_overhead.set_yticks(range(len(WORKLOADS)))
    sub_overhead.set_yticklabels(WORKLOADS, fontsize=9)
    for r in range(grid_o.shape[0]):
        for c in range(grid_o.shape[1]):
            v = grid_o[r, c]
            txtcolor = "white" if v > 200 else "black"
            sub_overhead.text(
                c, r, f"{v:.0f} ms",
                ha="center", va="center", fontsize=9, color=txtcolor,
            )
    sub_overhead.set_title("base overhead (ms)", fontsize=10)
    fig.colorbar(im_o, ax=sub_overhead, fraction=0.046, pad=0.04)

    im_c = sub_contention.imshow(grid_c, cmap="RdYlGn_r", aspect="auto", vmin=0.5, vmax=1.5)
    sub_contention.set_xticks(range(len(BACKENDS)))
    sub_contention.set_xticklabels(BACKENDS, fontsize=9)
    sub_contention.set_yticks(range(len(WORKLOADS)))
    sub_contention.set_yticklabels(WORKLOADS, fontsize=9)
    prov = payload["v3_contention_provenance"]
    for r, w in enumerate(WORKLOADS):
        for c, b in enumerate(BACKENDS):
            v = grid_c[r, c]
            tag = (prov.get(w, {}) or {}).get(b, "")
            marker = "*" if tag == "measured" else ""
            sub_contention.text(
                c, r, f"{v:.2f}{marker}",
                ha="center", va="center", fontsize=9, color="black",
            )
    sub_contention.set_title("contention factor (* = measured)", fontsize=10)
    fig.colorbar(im_c, ax=sub_contention, fraction=0.046, pad=0.04)

    # Title hint over the (hidden) parent ax for the panel.
    axes[0][1].set_title(
        "Top-right: v3 per-(workload, backend) base overhead + contention",
        fontsize=11, pad=12,
    )

    # ---- Bottom-left: multi-rate recommendation comparison --------------
    ax = axes[1][0]
    mr = payload["multi_rate"]
    names = [
        "closed-loop\nhand-tuned",
        "v1 per-backend",
        "v2 per-workload",
        "v3 two-term",
    ]
    vals = [
        mr["closed_loop_hand_tuned"],
        mr["v1"] if mr["v1"] is not None else 0,
        mr["v2"] if mr["v2"] is not None else 0,
        mr["v3"] if mr["v3"] is not None else 0,
    ]
    colors = ["#444444", "#cc1f1f", "#1f7fcc", "#2ca02c"]
    bars = ax.bar(names, vals, color=colors, edgecolor="black", linewidth=0.4)
    for rect, v in zip(bars, vals, strict=True):
        ax.text(
            rect.get_x() + rect.get_width() / 2,
            rect.get_height() + 0.3,
            str(v), ha="center", va="bottom", fontsize=11,
        )
    ax.set_ylabel("Recommended dronet multiplicity per yolov8n period")
    ax.set_title("Bottom-left: multi-rate recommendation (closed-loop vs v1/v2/v3)")
    ax.set_ylim(0, max(vals) * 1.25 + 1)
    ax.grid(axis="y", alpha=0.3)

    # ---- Bottom-right: random-overhead control + closed-loop error -----
    ax = axes[1][1]
    cl = payload["closed_loop_errors"]
    cl_conds = ("chain_sum", "v1_per_backend", "v2_per_workload", "v3_two_term")
    cl_labels = ("chain_sum", "v1", "v2", "v3")
    cl_vals = [cl[c] for c in cl_conds]
    bar_colors = ["#bbbbbb", "#cc1f1f", "#1f7fcc", "#2ca02c"]
    xb = np.arange(len(cl_conds))
    bars = ax.bar(xb, cl_vals, color=bar_colors, edgecolor="black", linewidth=0.4)
    for rect, v in zip(bars, cl_vals, strict=True):
        ax.text(
            rect.get_x() + rect.get_width() / 2,
            rect.get_height() + 0.5,
            f"{v:.1f}%",
            ha="center", va="bottom", fontsize=10,
        )
    ax.set_xticks(xb)
    ax.set_xticklabels(cl_labels)
    ax.set_ylabel("Mean abs % error (4 closed-loop rounds, yolov8n DSP)")
    ax.set_title("Bottom-right: closed-loop yolov8n DSP prediction error per fix")
    ax.set_ylim(0, max(cl_vals) * 1.15)
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        "Stage-1 calibration optimisation path: chain_sum → v1 per-backend → "
        "v2 per-workload → v3 two-term",
        fontsize=13,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(COMPOSITE_PATH, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _plot_trajectory(payload: dict[str, Any]) -> None:
    plt = sys.modules["matplotlib.pyplot"]

    stages = (
        "chain_sum",
        "+ per-backend (v1)",
        "+ per-workload (v2)",
        "+ contention (v3)",
        "+ board run\n(hypothetical)",
    )
    cl = payload["closed_loop_errors"]
    e = payload["per_workload_errors"]
    yolo = [
        e["yolov8n"]["chain_sum"],
        e["yolov8n"]["v1_per_backend"],
        e["yolov8n"]["v2_per_workload"],
        e["yolov8n"]["v3_two_term"],
        None,
    ]
    dro = [
        e["dronet"]["chain_sum"],
        e["dronet"]["v1_per_backend"],
        e["dronet"]["v2_per_workload"],
        e["dronet"]["v3_two_term"],
        None,
    ]
    cl_series = [
        cl["chain_sum"],
        cl["v1_per_backend"],
        cl["v2_per_workload"],
        cl["v3_two_term"],
        None,
    ]

    fig, ax = plt.subplots(figsize=(12.0, 5.5))
    x = list(range(len(stages)))
    ax.plot(
        x[:4], yolo[:4],
        marker="o", linewidth=2.2, color="#cc1f1f",
        label="yolov8n solo (mean abs %, 3 lanes)",
    )
    ax.plot(
        x[:4], dro[:4],
        marker="o", linewidth=2.2, color="#1f7fcc",
        label="dronet solo (mean abs %, 3 lanes)",
    )
    ax.plot(
        x[:4], cl_series[:4],
        marker="s", linewidth=2.2, color="#2ca02c",
        label="yolov8n closed-loop DSP (mean abs %, 4 rounds)",
    )

    # Dashed continuation to the hypothetical 5th point.
    for series, color in (
        (yolo, "#cc1f1f"), (dro, "#1f7fcc"), (cl_series, "#2ca02c"),
    ):
        if series[3] is not None:
            ax.plot(
                [x[3], x[4]], [series[3], series[3]],
                color=color, linewidth=1.2, linestyle="--", alpha=0.55,
            )

    ax.scatter(
        [x[4]] * 3,
        [yolo[3], dro[3], cl_series[3]],
        marker="x", s=80, color="black", zorder=5,
    )
    ax.annotate(
        "TODO — needs board run\n(per-workload contention\nfor dronet too)",
        xy=(x[4], max(yolo[3], dro[3], cl_series[3])),
        xytext=(x[4] - 0.85, max(yolo[3], dro[3], cl_series[3]) + 18.0),
        fontsize=9, ha="left",
        arrowprops={"arrowstyle": "->", "color": "black", "alpha": 0.6},
    )
    for series, color in ((yolo, "#cc1f1f"), (dro, "#1f7fcc"), (cl_series, "#2ca02c")):
        for xi, v in zip(x[:4], series[:4], strict=True):
            if v is not None:
                ax.annotate(
                    f"{v:.0f}%", xy=(xi, v), xytext=(0, 6),
                    textcoords="offset points",
                    fontsize=8, ha="center", color=color,
                )
    ax.set_xticks(x)
    ax.set_xticklabels(stages, fontsize=9)
    ax.set_ylabel("Mean abs % prediction error")
    ax.set_title("Stage-1 calibration optimisation arc (per workload + closed-loop)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(TRAJECTORY_PATH, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _write_results_jsonl(payload: dict[str, Any]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = [
        {"schema_version": "exp20_path_v2", "section": "per_workload_errors",
         **payload["per_workload_errors"]},
        {"schema_version": "exp20_path_v2", "section": "closed_loop_errors",
         **payload["closed_loop_errors"]},
        {"schema_version": "exp20_path_v2", "section": "v3_overhead_us",
         **payload["v3_overhead_us"]},
        {"schema_version": "exp20_path_v2", "section": "v3_contention_factor",
         **payload["v3_contention_factor"]},
        {"schema_version": "exp20_path_v2", "section": "v3_contention_provenance",
         **payload["v3_contention_provenance"]},
        {"schema_version": "exp20_path_v2", "section": "v1_overhead_us",
         **payload["v1_overhead_us"]},
        {"schema_version": "exp20_path_v2", "section": "multi_rate",
         **payload["multi_rate"]},
        {"schema_version": "exp20_path_v2", "section": "random_control",
         **payload["random_control"]},
    ]
    with RESULTS_PATH.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, default=str) + "\n")


def _write_summary(payload: dict[str, Any]) -> None:
    e = payload["per_workload_errors"]
    cl = payload["closed_loop_errors"]
    mr = payload["multi_rate"]
    rand_v1 = payload["random_control"]["v1"]
    rand_v2 = payload["random_control"]["v2"]
    rand_v3 = payload["random_control"]["v3"]
    pct_v1 = _percentile_of_seed(rand_v1["fraction_random_beats_seeded"]) if rand_v1 else float("nan")
    pct_v2 = _percentile_of_seed(rand_v2["fraction_random_beats_seeded"]) if rand_v2 else float("nan")
    pct_v3 = _percentile_of_seed(rand_v3["fraction_random_beats_seeded"]) if rand_v3 else float("nan")

    yolo_v1 = e["yolov8n"]["v1_per_backend"]
    yolo_v2 = e["yolov8n"]["v2_per_workload"]
    yolo_v3 = e["yolov8n"]["v3_two_term"]
    dro_v1 = e["dronet"]["v1_per_backend"]
    dro_v2 = e["dronet"]["v2_per_workload"]
    dro_v3 = e["dronet"]["v3_two_term"]

    cl_chain = cl["chain_sum"]
    cl_v1 = cl["v1_per_backend"]
    cl_v2 = cl["v2_per_workload"]
    cl_v3 = cl["v3_two_term"]

    v3_recovers_cl = cl_v3 <= cl_v1
    v3_keeps_solo_dronet = dro_v3 <= dro_v2 + 0.5
    v3_strict_win = v3_recovers_cl and v3_keeps_solo_dronet

    lines: list[str] = []
    lines.append("# Exp 20 — Stage-1 calibration optimisation path (v3 two-term)\n")
    lines.append(
        "Headline composite for the v1 per-backend → v2 per-workload base → "
        "v3 two-term (base × contention) calibration arc.\n"
    )

    lines.append("## 2x2 composite annotation key\n")
    lines.append(
        "* **Top-left** — Mean abs % solo-E2E prediction error per workload "
        "(yolov8n red, dronet blue) across 4 predictor conditions: raw "
        "chain_sum, v1 per-backend, v2 per-workload base, v3 two-term.\n"
        "* **Top-right** — paired heatmaps. Left: v3 per-(workload, backend) "
        "**base overhead** (ms). Right: v3 per-(workload, backend) "
        "**contention factor** (unitless; \\* = measured from closed-loop, "
        "no marker = default 1.0 because no contended ground truth yet).\n"
        "* **Bottom-left** — multi-rate recommendation: closed-loop hand-tuned, "
        "v1, v2, v3.\n"
        "* **Bottom-right** — closed-loop yolov8n-DSP per-condition error "
        "(4 measured rounds vs each predictor). The headline accuracy panel.\n"
    )

    lines.append("## Headline numbers (4-condition table)\n")
    lines.append(
        "| Metric | chain_sum | v1 per-backend | v2 per-workload | v3 two-term |\n"
        "|---|---:|---:|---:|---:|"
    )
    lines.append(
        f"| yolov8n closed-loop DSP mean abs % (4 rounds) | "
        f"{cl_chain:.1f}% | {cl_v1:.1f}% | {cl_v2:.1f}% | **{cl_v3:.1f}%** |"
    )
    lines.append(
        f"| yolov8n solo E2E mean abs % (3 lanes)         | "
        f"{e['yolov8n']['chain_sum']:.1f}% | {yolo_v1:.1f}% | {yolo_v2:.1f}% | {yolo_v3:.1f}% |"
    )
    lines.append(
        f"| dronet  solo E2E mean abs % (3 lanes)         | "
        f"{e['dronet']['chain_sum']:.1f}% | {dro_v1:.1f}% | {dro_v2:.1f}% | {dro_v3:.1f}% |"
    )
    lines.append(
        f"| dronet multiplicity recommendation            | — | "
        f"{mr['v1']} | {mr['v2']} | {mr['v3']} |"
    )
    lines.append(
        f"| Random-control percentile of seed (v3 closed-loop) | — | "
        f"{pct_v1:.0f}th | {pct_v2:.0f}th | {pct_v3:.0f}th |\n"
    )

    lines.append("## v3 contention factor table (per workload, per backend)\n")
    lines.append("| workload | backend | contention | provenance |")
    lines.append("|---|---|---:|---|")
    prov = payload["v3_contention_provenance"]
    for w in WORKLOADS:
        for b in BACKENDS:
            cf = float(payload["v3_contention_factor"].get(w, {}).get(b, 1.0))
            tag = (prov.get(w, {}) or {}).get(b, "default_no_data")
            lines.append(f"| {w} | {b} | {cf:.3f} | {tag} |")
    lines.append("")

    lines.append("## Side-by-side v1 / v2 / v3 numbers\n")
    lines.append(
        "| Metric | v1 per-backend | v2 per-workload | v3 two-term | Δ(v3 − v2) |\n"
        "|---|---:|---:|---:|---:|"
    )
    lines.append(
        f"| yolov8n closed-loop DSP mean abs % | "
        f"{cl_v1:.1f}% | {cl_v2:.1f}% | {cl_v3:.1f}% | "
        f"{cl_v3 - cl_v2:+.1f} pp |"
    )
    lines.append(
        f"| yolov8n solo E2E mean abs %        | "
        f"{yolo_v1:.1f}% | {yolo_v2:.1f}% | {yolo_v3:.1f}% | "
        f"{yolo_v3 - yolo_v2:+.1f} pp |"
    )
    lines.append(
        f"| dronet solo E2E mean abs %         | "
        f"{dro_v1:.1f}% | {dro_v2:.1f}% | {dro_v3:.1f}% | "
        f"{dro_v3 - dro_v2:+.1f} pp |"
    )
    lines.append(
        f"| dronet multiplicity                 | "
        f"{mr['v1']} | {mr['v2']} | {mr['v3']} | "
        f"{(mr['v3'] or 0) - (mr['v2'] or 0):+d} |\n"
    )

    lines.append("## Verdict\n")
    if v3_strict_win:
        verdict = "**v3 strictly improves on both v1 and v2.**"
    elif v3_recovers_cl and not v3_keeps_solo_dronet:
        verdict = (
            "**v3 partially wins.** Closed-loop accuracy recovered "
            "below v1, but solo-E2E regressed somewhere (likely yolov8n DSP, "
            "where the 0.788 contention factor pulls the predictor below "
            "the solo measurement)."
        )
    elif not v3_recovers_cl:
        verdict = "**v3 does NOT recover closed-loop accuracy below v1's level.**"
    else:
        verdict = "**v3 mixed result — see per-workload deltas.**"
    lines.append(verdict + "\n")
    lines.append(
        f"- Closed-loop yolov8n DSP error: chain {cl_chain:.1f}% → "
        f"v1 {cl_v1:.1f}% → v2 {cl_v2:.1f}% → **v3 {cl_v3:.1f}%** "
        f"(v3 vs v1: {cl_v3 - cl_v1:+.1f} pp; v3 vs v2: {cl_v3 - cl_v2:+.1f} pp).\n"
        f"- yolov8n solo DSP fit: v2 = 0.0% (tautology); v3 = "
        f"{abs(((payload['v3_overhead_us']['yolov8n']['DSP'] / 1000.0) + (_chain_sum_us(json.loads(COST_MATRIX_PATH.read_text()), 'yolov8n', 'DSP') / 1000.0)) * float(payload['v3_contention_factor']['yolov8n']['DSP']) - 354.88) / 354.88 * 100.0:.1f}% "
        "(v3's 0.788 contention pulls the prediction below the solo "
        "measurement — honest cost of the multiplicative fit).\n"
        f"- dronet solo: v2 = 0.0% (tautology); v3 = {dro_v3:.1f}% "
        f"({'unchanged' if abs(dro_v3 - dro_v2) < 0.1 else 'changed'} "
        "because contention defaults to 1.0 with no contended ground truth).\n"
        f"- Multi-rate recommendation: closed-loop empirical 12; "
        f"v1 = {mr['v1']}, v2 = {mr['v2']}, v3 = {mr['v3']}. "
        + (
            "Unchanged from v2 because the dominant lane is yolov8n CPU, "
            "and the only measured contention cell is yolov8n DSP — which "
            "doesn't shift CPU-vs-DSP dominance."
            if (mr['v3'] or 0) == (mr['v2'] or 0)
            else "Moved under v3."
        ) + "\n"
    )

    lines.append("## What this still does NOT prove\n")
    lines.append(
        "* **No fresh board run.** v3's contention factor is fit from the "
        "*same 4 closed-loop rounds* it is being evaluated against — that's "
        "an in-sample fit, not held-out evidence. The honest test would be "
        "round 5+ on the board.\n"
        "* **dronet contention is not measured.** All dronet (w, b) cells "
        "have contention 1.0 with provenance ``default_no_data``. The "
        "closed-loop trace doesn't separate the 12 dronet copies from the "
        "single yolov8n copy at per-workload granularity (only the aggregate "
        "'12/12 met 40 ms deadline'). To fit dronet contention we need a "
        "board run that records per-workload measurements while the same "
        "1× yolov8n + 12× dronet workload mix runs.\n"
        "* **Counter-intuitive direction.** v3's yolov8n DSP factor is "
        "**0.788** — under contention yolov8n on DSP runs *faster* than "
        "solo. The most plausible explanation: 12× dronet runs entirely on "
        "CPU, leaving DSP fully available, possibly with cache-warmth "
        "benefits from repeated DLC invocations. We have no formal "
        "explanation without further board measurements.\n"
        "* **Multi-rate recommendation doesn't move.** v3's only non-trivial "
        "contention cell is yolov8n DSP, but yolov8n's preferred lane is "
        "CPU (lower period), so the recommendation stays at v2's value. "
        "A real board run might move it.\n"
    )

    lines.append("## Most actionable next step\n")
    lines.append(
        "Run **1× yolov8n + 12× dronet on QRB5165 with per-workload "
        "measurement breakdown**. Today the closed-loop trace records the "
        "yolov8n per-iter wall (the source of the 0.788 contention factor) "
        "but not the per-dronet-copy wall. Adding per-copy timing lets us "
        "fit ``contention[dronet][CPU]`` and ``contention[dronet][DSP]`` "
        "the same way v3 fits yolov8n's. Without that, half of v3's "
        "contention table is ``default_no_data`` and the multi-rate "
        "recommendation cannot benefit from contention information.\n"
    )

    SUMMARY_PATH.write_text("\n".join(lines) + "\n")


def main() -> int:
    _require_matplotlib()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = _gather_payload()
    print(json.dumps({k: v for k, v in payload.items() if k != "v3_contention_provenance"},
                     indent=2, default=str))
    _plot_composite(payload)
    _plot_trajectory(payload)
    _write_results_jsonl(payload)
    _write_summary(payload)
    print(f"\n[exp20] wrote {COMPOSITE_PATH}")
    print(f"[exp20] wrote {TRAJECTORY_PATH}")
    print(f"[exp20] wrote {SUMMARY_PATH}")
    print(f"[exp20] wrote {RESULTS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
