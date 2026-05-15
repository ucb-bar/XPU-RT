"""real PNG figures for the extension/provider evidence pack.

Reads the same underlying JSON/CSV data the pack writes
(``provider_status.json``, ``provider_target_matrix.csv``,
``provider_contract_matrix.csv``, ``dialect_status.json``) and
renders five real PNGs via matplotlib.

All renderers must:
* import matplotlib *lazily* so this module's import cost stays
  zero when figures aren't needed;
* skip a figure gracefully when its source data is missing (emit a
  typed ``FigureSkipResult`` rather than crash the pack build);
* produce non-empty PNGs (≥ 1 KB) when data is present.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

FIGURE_DPI = 110


@dataclass(frozen=True)
class FigureResult:
    name: str
    path: Path
    skipped: bool
    reason: str = ""


def _matplotlib():
    """Lazy matplotlib import with non-interactive backend."""

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return matplotlib, plt


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def _provider_status_rows(pack_dir: Path) -> list[dict[str, Any]] | None:
    body = _read_json(pack_dir / "provider_status.json")
    if not body:
        return None
    return body.get("providers", [])


def render_provider_target_heatmap(pack_dir: Path, out: Path) -> FigureResult:
    """Heatmap of provider × target_family showing probe status."""

    rows = _provider_status_rows(pack_dir)
    if not rows:
        return FigureResult(out.name, out, skipped=True, reason="no provider_status.json")
    matplotlib, plt = _matplotlib()
    families: list[str] = sorted(
        {f for r in rows for f in r.get("target_families", [])}
    )
    providers = [r["provider_id"] for r in rows]
    if not families or not providers:
        return FigureResult(out.name, out, skipped=True, reason="empty matrix")
    # Encode: 1=available, 0.5=blocked, 0=unsupported/no-claim
    cell = []
    for r in rows:
        row_cells = []
        for f in families:
            if f in r.get("target_families", []):
                if r["status"] == "available":
                    row_cells.append(1.0)
                else:
                    row_cells.append(0.5)
            else:
                row_cells.append(0.0)
        cell.append(row_cells)
    fig, ax = plt.subplots(
        figsize=(max(6, len(families) * 0.8), max(4, len(providers) * 0.35)),
        dpi=FIGURE_DPI,
    )
    im = ax.imshow(cell, aspect="auto", cmap="RdYlGn", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(families)))
    ax.set_xticklabels(families, rotation=45, ha="right")
    ax.set_yticks(range(len(providers)))
    ax.set_yticklabels(providers)
    ax.set_title("Provider × target_family — green=available, yellow=blocked, red=no-claim")
    cbar = fig.colorbar(im, ax=ax, ticks=[0.0, 0.5, 1.0])
    cbar.ax.set_yticklabels(["no-claim", "blocked", "available"])
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return FigureResult(out.name, out, skipped=False)


def render_provider_status_by_family(pack_dir: Path, out: Path) -> FigureResult:
    """Stacked bar: target family → counts of {available, blocked, …}."""

    rows = _provider_status_rows(pack_dir)
    if not rows:
        return FigureResult(out.name, out, skipped=True, reason="no provider_status.json")
    matplotlib, plt = _matplotlib()
    by_family: dict[str, Counter] = {}
    for r in rows:
        status = r["status"]
        for f in r.get("target_families", []) or ["(none)"]:
            by_family.setdefault(f, Counter())[status] += 1
    families = sorted(by_family.keys())
    all_statuses = sorted({s for c in by_family.values() for s in c})
    if not families or not all_statuses:
        return FigureResult(out.name, out, skipped=True, reason="empty data")
    fig, ax = plt.subplots(figsize=(max(7, len(families) * 0.7), 5), dpi=FIGURE_DPI)
    bottom = [0] * len(families)
    palette = {
        "available": "#3a8a3a",
        "blocked": "#c98a4b",
        "unsupported": "#5a5a5a",
        "probe_error": "#a02020",
        "not_installed": "#888888",
    }
    for status in all_statuses:
        vals = [by_family[f].get(status, 0) for f in families]
        ax.bar(families, vals, bottom=bottom, label=status, color=palette.get(status))
        bottom = [b + v for b, v in zip(bottom, vals)]
    ax.set_ylabel("# providers")
    ax.set_title("Provider status by target family")
    ax.legend()
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return FigureResult(out.name, out, skipped=False)


def render_blocked_reason_breakdown(pack_dir: Path, out: Path) -> FigureResult:
    """Bar chart of blocked_reason → # providers."""

    rows = _provider_status_rows(pack_dir)
    if not rows:
        return FigureResult(out.name, out, skipped=True, reason="no provider_status.json")
    matplotlib, plt = _matplotlib()
    counts = Counter(
        r.get("blocked_reason") or "(none)" for r in rows if r["status"] != "available"
    )
    if not counts:
        return FigureResult(out.name, out, skipped=True, reason="no blocked providers")
    labels, values = zip(*sorted(counts.items(), key=lambda kv: -kv[1]))
    fig, ax = plt.subplots(figsize=(max(7, len(labels) * 0.9), 5), dpi=FIGURE_DPI)
    ax.bar(labels, values, color="#c98a4b")
    ax.set_ylabel("# providers")
    ax.set_title("Blocked-reason breakdown")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return FigureResult(out.name, out, skipped=False)


def render_ir_analysis_levels(pack_dir: Path, out: Path) -> FigureResult:
    """Bar of analysis-snapshot levels with regions count (when present)."""

    snapshots_dir = pack_dir / "analysis_snapshots"
    if not snapshots_dir.is_dir():
        return FigureResult(out.name, out, skipped=True, reason="no analysis_snapshots/")
    matplotlib, plt = _matplotlib()
    levels = []
    region_counts = []
    statuses = []
    for f in sorted(snapshots_dir.glob("*_analysis.json")):
        body = _read_json(f)
        if not body:
            continue
        levels.append(body.get("level", f.stem))
        region_counts.append(len(body.get("regions", [])))
        statuses.append(body.get("status", "unknown"))
    if not levels:
        return FigureResult(out.name, out, skipped=True, reason="no snapshots")
    fig, ax = plt.subplots(figsize=(max(7, len(levels) * 0.9), 5), dpi=FIGURE_DPI)
    palette = {"available": "#3a8a3a", "not_available": "#888888"}
    colors = [palette.get(s, "#777777") for s in statuses]
    ax.bar(levels, region_counts, color=colors)
    ax.set_ylabel("# regions")
    ax.set_title("IR analysis snapshots by level")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return FigureResult(out.name, out, skipped=False)


def render_extension_lifecycle(pack_dir: Path, out: Path) -> FigureResult:
    """Simple flow diagram of the extension lifecycle: card → probe →
    propose → verify → certificate → register."""

    matplotlib, plt = _matplotlib()
    fig, ax = plt.subplots(figsize=(11, 3), dpi=FIGURE_DPI)
    boxes = [
        "Card",
        "Probe",
        "Propose",
        "Verifier",
        "Certificate",
        "Register",
    ]
    x = list(range(len(boxes)))
    for i, label in enumerate(boxes):
        ax.add_patch(
            plt.Rectangle(
                (i - 0.4, 0.3), 0.8, 0.4, edgecolor="#333", facecolor="#cfe2f3"
            )
        )
        ax.text(i, 0.5, label, ha="center", va="center", fontsize=10)
        if i < len(boxes) - 1:
            ax.annotate(
                "",
                xy=(i + 0.5, 0.5),
                xytext=(i + 0.4, 0.5),
                arrowprops={"arrowstyle": "->", "color": "#555"},
            )
    ax.set_xlim(-0.6, len(boxes) - 0.4)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Extension lifecycle — agent proposes; CompGen verifies + certifies")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return FigureResult(out.name, out, skipped=False)


def render_all_figures(pack_dir: Path | str) -> list[FigureResult]:
    """Render the 5 spec'd figures, replacing the markdown placeholders.

    Returns one :class:`FigureResult` per figure (some may be
    ``skipped`` if their source data is absent).
    """

    pack = Path(pack_dir)
    figures_dir = pack / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Clean out the markdown placeholders so the dir contains
    # only real renderable artifacts.
    for md in figures_dir.glob("*.md"):
        md.unlink()

    results = [
        render_provider_target_heatmap(pack, figures_dir / "provider_target_heatmap.png"),
        render_provider_status_by_family(pack, figures_dir / "provider_status_by_family.png"),
        render_extension_lifecycle(pack, figures_dir / "extension_lifecycle.png"),
        render_ir_analysis_levels(pack, figures_dir / "ir_analysis_levels.png"),
        render_blocked_reason_breakdown(pack, figures_dir / "blocked_reason_breakdown.png"),
    ]
    return results


__all__ = [
    "FigureResult",
    "render_all_figures",
    "render_blocked_reason_breakdown",
    "render_extension_lifecycle",
    "render_ir_analysis_levels",
    "render_provider_status_by_family",
    "render_provider_target_heatmap",
]
