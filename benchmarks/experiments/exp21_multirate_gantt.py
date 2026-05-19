"""Exp 21 — Multi-rate Gantt comparison: v1 (per-backend) vs v2 (per-workload).

Builds a side-by-side Gantt for one yolov8n period under each
calibration:

  * Left panel: v1 per-backend calibration → 7× dronet on CPU.
  * Right panel: v2 per-workload calibration → N× dronet on the lane
    its analyser picked.

The panels share x-axis units (microseconds) and y-axis device labels.
The N for v2 is read live from ``build/experiments/exp19_multi_rate/results.json``;
the v1 snapshot is read from the archived
``results.v1_per_backend.json`` (saved before the per-workload re-run).

Outputs:
    build/experiments/exp21_multirate_gantt/multirate_v1_vs_v2.png
    build/experiments/exp21_multirate_gantt/summary.md

Usage:
    uv run python scripts/experiments/exp21_multirate_gantt.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
EXP19_DIR = REPO_ROOT / "build" / "experiments" / "exp19_multi_rate"
OUT_DIR = REPO_ROOT / "build" / "experiments" / "exp21_multirate_gantt"
GANTT_PATH = OUT_DIR / "multirate_v1_vs_v2.png"
SUMMARY_PATH = OUT_DIR / "summary.md"

V1_RESULTS = EXP19_DIR / "results.v1_per_backend.json"
V2_RESULTS = EXP19_DIR / "results.json"

BACKENDS: tuple[str, ...] = ("CPU", "GPU", "DSP")


def _require_matplotlib() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: F401

    return matplotlib


def _load_recommendation(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"missing multi-rate result {path}; re-run exp19 first"
        )
    return json.loads(path.read_text())


def _draw_panel(ax: Any, payload: dict[str, Any], title: str) -> None:
    plt = sys.modules["matplotlib.pyplot"]
    from matplotlib.patches import Patch

    dominant_id = payload["dominant_workload_id"]
    dominant_period = float(payload["dominant_period_us"])
    rates = payload["rates"]

    dominant = next(r for r in rates if r["workload_id"] == dominant_id)
    secondaries = [r for r in rates if r["workload_id"] != dominant_id]

    backends = list(BACKENDS)
    y_for = {b: i for i, b in enumerate(backends)}

    dom_color = (0.86, 0.20, 0.20, 0.92)
    dom_busy = float(dominant.get("primary_lane_busy_us", 0.0))
    ax.broken_barh(
        [(0.0, max(dom_busy, 1.0))],
        (y_for[dominant["preferred_lane"]] - 0.4, 0.8),
        facecolors=dom_color,
        edgecolors="black",
        linewidth=0.4,
    )

    sec_palette = plt.get_cmap("viridis")
    handles = [Patch(facecolor=dom_color, edgecolor="black", label=dominant_id)]
    for s_idx, sec in enumerate(secondaries):
        mult = int(sec.get("multiplicity", 0))
        if mult <= 0:
            continue
        primary_busy = float(sec.get("primary_lane_busy_us", 0.0))
        per_cycle = primary_busy / mult if mult > 0 else 0.0
        color_t = (s_idx + 1) / max(len(secondaries), 1)
        sec_color = sec_palette(color_t)
        for i in range(mult):
            ax.broken_barh(
                [(i * per_cycle, max(per_cycle, 1.0))],
                (y_for[sec["preferred_lane"]] - 0.4, 0.8),
                facecolors=sec_color,
                edgecolors="black",
                linewidth=0.25,
            )
        handles.append(
            Patch(
                facecolor=sec_color,
                edgecolor="black",
                label=f"{sec['workload_id']} ×{mult}",
            )
        )

    ax.set_yticks(list(range(len(backends))))
    ax.set_yticklabels(backends)
    ax.set_ylim(-0.7, len(backends) - 0.3)
    ax.set_xlim(0, dominant_period * 1.04)
    ax.set_xlabel("time (us)")
    ax.axvline(
        dominant_period,
        color="black",
        linestyle="--",
        alpha=0.55,
        linewidth=1.0,
        label="dominant period",
    )
    ax.set_title(title, fontsize=11)
    ax.grid(axis="x", linestyle=":", alpha=0.4)
    ax.legend(handles=handles, loc="upper right", fontsize=8, frameon=True)


def _plot_side_by_side(v1: dict[str, Any], v2: dict[str, Any]) -> None:
    plt = sys.modules["matplotlib.pyplot"]

    fig, (ax_v1, ax_v2) = plt.subplots(2, 1, figsize=(14.0, 9.0), sharex=False)

    v1_dronet = next(r for r in v1["rates"] if r["workload_id"] == "dronet")
    v2_dronet = next(r for r in v2["rates"] if r["workload_id"] == "dronet")

    _draw_panel(
        ax_v1,
        v1,
        f"v1 per-backend calibration — dominant {v1['dominant_workload_id']} "
        f"on {next(r for r in v1['rates'] if r['workload_id'] == v1['dominant_workload_id'])['preferred_lane']}, "
        f"period {v1['dominant_period_us'] / 1000.0:.1f} ms; "
        f"dronet ×{v1_dronet['multiplicity']} on {v1_dronet['preferred_lane']}",
    )
    _draw_panel(
        ax_v2,
        v2,
        f"v2 per-workload calibration — dominant {v2['dominant_workload_id']} "
        f"on {next(r for r in v2['rates'] if r['workload_id'] == v2['dominant_workload_id'])['preferred_lane']}, "
        f"period {v2['dominant_period_us'] / 1000.0:.1f} ms; "
        f"dronet ×{v2_dronet['multiplicity']} on {v2_dronet['preferred_lane']}",
    )
    ax_v1.set_ylabel("device (v1)")
    ax_v2.set_ylabel("device (v2)")

    fig.suptitle(
        "Multi-rate Gantt: per-backend (v1) vs per-workload (v2) calibration",
        fontsize=13,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(GANTT_PATH, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _write_summary(v1: dict[str, Any], v2: dict[str, Any]) -> None:
    v1_dronet = next(r for r in v1["rates"] if r["workload_id"] == "dronet")
    v2_dronet = next(r for r in v2["rates"] if r["workload_id"] == "dronet")
    v1_dom = next(r for r in v1["rates"] if r["workload_id"] == v1["dominant_workload_id"])
    v2_dom = next(r for r in v2["rates"] if r["workload_id"] == v2["dominant_workload_id"])

    lines: list[str] = []
    lines.append("# Exp 21 — Multi-rate Gantt v1 vs v2\n")
    lines.append(
        "Side-by-side Gantt for one yolov8n period under each calibration. "
        "Closed-loop hand-tuned baseline: 12× dronet on CPU.\n"
    )

    lines.append("## Recommendation table\n")
    lines.append(
        "| Calibration | dominant | dominant lane | dominant period (ms) | dronet × | dronet lane |\n"
        "|---|---|---|---:|---:|---|"
    )
    lines.append(
        f"| v1 per-backend | {v1['dominant_workload_id']} | "
        f"{v1_dom['preferred_lane']} | {v1['dominant_period_us'] / 1000.0:.1f} | "
        f"{v1_dronet['multiplicity']} | {v1_dronet['preferred_lane']} |"
    )
    lines.append(
        f"| v2 per-workload | {v2['dominant_workload_id']} | "
        f"{v2_dom['preferred_lane']} | {v2['dominant_period_us'] / 1000.0:.1f} | "
        f"{v2_dronet['multiplicity']} | {v2_dronet['preferred_lane']} |"
    )
    lines.append(
        f"| closed-loop hand-tuned | yolov8n | DSP | n/a | 12 | CPU/DSP split |\n"
    )

    lines.append("## Reading the Gantt\n")
    lines.append(
        "* Each row of bars = one device (CPU / GPU / DSP).\n"
        "* The wide red bar is the dominant workload (yolov8n) on its preferred lane.\n"
        "* The viridis-coloured smaller bars are dronet cycles packed into the "
        "lane the analyser picked.\n"
        "* The dashed black vertical line marks the end of one dominant period.\n"
        "* Gantt bars do not include any per-cycle gap — the busy fraction is "
        "the static analyser's upper bound.\n"
    )

    if v2_dronet["multiplicity"] >= 12:
        verdict = (
            f"v2 matches/exceeds the closed-loop ({v2_dronet['multiplicity']} ≥ 12)."
        )
    elif v2_dronet["multiplicity"] > v1_dronet["multiplicity"]:
        verdict = (
            f"v2 ({v2_dronet['multiplicity']}) is more aggressive than v1 "
            f"({v1_dronet['multiplicity']}) but still below the closed-loop "
            f"hand-tuned {12}."
        )
    else:
        verdict = (
            f"v2 ({v2_dronet['multiplicity']}) is *more conservative* than v1 "
            f"({v1_dronet['multiplicity']}) — the per-workload overhead "
            "correctly punishes yolov8n's high GPU/DSP setup cost, shifting the "
            "dominant lane choice and reducing the secondary's lane budget."
        )
    lines.append("## Verdict\n")
    lines.append(verdict + "\n")

    SUMMARY_PATH.write_text("\n".join(lines) + "\n")


def main() -> int:
    _require_matplotlib()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    v1 = _load_recommendation(V1_RESULTS)
    v2 = _load_recommendation(V2_RESULTS)
    _plot_side_by_side(v1, v2)
    _write_summary(v1, v2)
    print(f"[exp21] wrote {GANTT_PATH}")
    print(f"[exp21] wrote {SUMMARY_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
