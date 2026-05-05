"""M-17 figures — matplotlib only, no seaborn.

Each figure is rendered from the in-memory rows + aggregates that the
evidence pack has already computed. PNGs are written under
``<out>/figures/``. The figures are paper-facing summaries — keep them
clean: discrete labels, no decorative gridlines, deterministic ordering.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _setup() -> None:
    import matplotlib
    matplotlib.use("Agg")  # headless


def render_all(rows, agg, figures_dir: Path) -> None:  # type: ignore[no-untyped-def]
    """Render the 7 required figures + the M-18 calibration figure
    (rendered when at least one model has calibration evidence)."""
    _setup()
    figures_dir.mkdir(parents=True, exist_ok=True)
    _payload_coverage(rows, figures_dir / "payload_coverage_by_model.png")
    _candidate_family(rows, figures_dir / "candidate_family_by_model.png")
    _selected_action_family(rows, figures_dir / "selected_action_family_by_model.png")
    _real_verification_status(rows, figures_dir / "real_verification_status_by_model.png")
    _retry_flow_counts(rows, figures_dir / "retry_flow_counts.png")
    _greedy_vs_agent(rows, figures_dir / "greedy_vs_agent_candidate_change.png")
    _transform_family_discharge(rows, figures_dir / "transform_family_discharge_matrix.png")
    if any(r.calibration_overall in ("calibrated", "partial") for r in rows):
        _calibration_coverage(rows, figures_dir / "calibration_coverage_by_model.png")
        _calibration_suite_scale(rows, figures_dir / "calibration_suite_scale_by_model.png")


def _save(fig, path: Path) -> None:  # type: ignore[no-untyped-def]
    fig.tight_layout()
    fig.savefig(path, format="png", dpi=110)
    import matplotlib.pyplot as plt

    plt.close(fig)


def _payload_coverage(rows, path: Path) -> None:  # type: ignore[no-untyped-def]
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(max(6, len(rows) * 0.55), 4.5))
    if not rows:
        ax.text(0.5, 0.5, "no models", ha="center", va="center")
        _save(fig, path); return
    labels = [r.model_id for r in rows]
    decomposed = [r.decomposed_structured for r in rows]
    opaque = [r.opaque_fallback for r in rows]
    unaccounted = [r.unaccounted_fx_nodes for r in rows]
    ax.bar(labels, decomposed, label="decomposed_structured", color="#2b8a3e")
    ax.bar(labels, opaque, bottom=decomposed,
           label="opaque_fallback", color="#f08c00")
    bottom2 = [a + b for a, b in zip(decomposed, opaque)]
    ax.bar(labels, unaccounted, bottom=bottom2,
           label="unaccounted", color="#c92a2a")
    ax.set_ylabel("FX call_function nodes")
    ax.set_title("Payload coverage by model")
    ax.tick_params(axis="x", rotation=45)
    for lab in ax.get_xticklabels():
        lab.set_horizontalalignment("right")
    ax.legend()
    _save(fig, path)


def _candidate_family(rows, path: Path) -> None:  # type: ignore[no-untyped-def]
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(max(6, len(rows) * 0.55), 4.5))
    if not rows:
        ax.text(0.5, 0.5, "no models", ha="center", va="center")
        _save(fig, path); return
    labels = [r.model_id for r in rows]
    families: list[str] = []
    for r in rows:
        for k in r.candidate_families:
            if k not in families:
                families.append(k)
    families.sort()
    palette = ["#1971c2", "#e67700", "#5f3dc4", "#0c8599", "#ae3ec9", "#2f9e44"]
    bottom = [0] * len(rows)
    for i, fam in enumerate(families):
        vals = [r.candidate_families.get(fam, 0) for r in rows]
        ax.bar(labels, vals, bottom=bottom,
               label=fam, color=palette[i % len(palette)])
        bottom = [a + b for a, b in zip(bottom, vals)]
    ax.set_ylabel("candidate count (legal + illegal)")
    ax.set_title("Candidate families by model")
    ax.tick_params(axis="x", rotation=45)
    for lab in ax.get_xticklabels():
        lab.set_horizontalalignment("right")
    ax.legend(fontsize="small", loc="upper left")
    _save(fig, path)


def _selected_action_family(rows, path: Path) -> None:  # type: ignore[no-untyped-def]
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(max(6, len(rows) * 0.55), 4.5))
    if not rows:
        ax.text(0.5, 0.5, "no models", ha="center", va="center")
        _save(fig, path); return
    labels = [r.model_id for r in rows]
    families = [r.selected_candidate_kind or "(none)" for r in rows]
    unique = sorted(set(families))
    color_map = {
        f: c for f, c in zip(
            unique,
            ["#1971c2", "#e67700", "#5f3dc4", "#0c8599",
             "#2f9e44", "#ae3ec9", "#868e96"],
        )
    }
    colors = [color_map[f] for f in families]
    ax.bar(labels, [1] * len(labels), color=colors)
    ax.set_yticks([])
    ax.set_title("Selected action family per model")
    ax.tick_params(axis="x", rotation=45)
    for lab in ax.get_xticklabels():
        lab.set_horizontalalignment("right")
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=color_map[f]) for f in unique
    ]
    ax.legend(handles, unique, fontsize="small", loc="upper right")
    _save(fig, path)


def _real_verification_status(rows, path: Path) -> None:  # type: ignore[no-untyped-def]
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(max(6, len(rows) * 0.55), 4.5))
    if not rows:
        ax.text(0.5, 0.5, "no models", ha="center", va="center")
        _save(fig, path); return
    labels = [r.model_id for r in rows]
    status_color = {
        "pass": "#2b8a3e", "fail": "#c92a2a",
        "blocked": "#f08c00", "n/a": "#adb5bd",
    }
    tile_colors = [status_color.get(r.real_set_tile_status, "#adb5bd") for r in rows]
    fusion_colors = [status_color.get(r.real_fusion_status, "#adb5bd") for r in rows]
    x = list(range(len(rows)))
    width = 0.4
    ax.bar([xi - width / 2 for xi in x], [1] * len(rows),
           width=width, color=tile_colors, label="set_tile")
    ax.bar([xi + width / 2 for xi in x], [1] * len(rows),
           width=width, color=fusion_colors, label="fusion")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks([])
    ax.set_title("Real differential verification status (set_tile vs fusion)")
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=c) for c in status_color.values()
    ]
    ax.legend(handles, list(status_color.keys()),
              fontsize="small", loc="upper right")
    _save(fig, path)


def _retry_flow_counts(rows, path: Path) -> None:  # type: ignore[no-untyped-def]
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(max(5, len(rows) * 0.55), 4.5))
    labels = [r.model_id for r in rows]
    val_retries = [r.retry_attempts for r in rows]
    down_retries = [r.downstream_retry_events for r in rows]
    x = list(range(len(rows)))
    width = 0.4
    ax.bar([xi - width / 2 for xi in x], val_retries,
           width=width, label="agent-decision retries", color="#1971c2")
    ax.bar([xi + width / 2 for xi in x], down_retries,
           width=width, label="downstream retries (M-15B)", color="#e67700")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("count")
    ax.set_title("Retry flow counts")
    ax.legend(fontsize="small")
    _save(fig, path)


def _greedy_vs_agent(rows, path: Path) -> None:  # type: ignore[no-untyped-def]
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(max(5, len(rows) * 0.55), 4.5))
    labels = [r.model_id for r in rows]
    changed = [1 if r.agent_changed_from_greedy else 0 for r in rows]
    unchanged = [1 - c for c in changed]
    ax.bar(labels, unchanged, label="ratified greedy", color="#74b816")
    ax.bar(labels, changed, bottom=unchanged,
           label="agent changed pick", color="#5f3dc4")
    ax.set_yticks([])
    ax.set_title("Greedy vs agent candidate change")
    ax.tick_params(axis="x", rotation=45)
    for lab in ax.get_xticklabels():
        lab.set_horizontalalignment("right")
    ax.legend(fontsize="small")
    _save(fig, path)


def _calibration_coverage(rows, path: Path) -> None:  # type: ignore[no-untyped-def]
    import matplotlib.pyplot as plt

    cal_rows = [
        r for r in rows
        if r.calibration_overall in ("calibrated", "partial")
    ]
    fig, ax = plt.subplots(figsize=(max(6, len(cal_rows) * 0.45), 4.5))
    if not cal_rows:
        ax.text(0.5, 0.5, "no calibration", ha="center", va="center")
        _save(fig, path); return
    labels = [r.model_id for r in cal_rows]
    matched_pct = [r.calibration_match_fraction * 100.0 for r in cal_rows]
    colors = [
        "#2b8a3e" if r.calibration_overall == "calibrated" else "#f08c00"
        for r in cal_rows
    ]
    ax.bar(labels, matched_pct, color=colors)
    ax.axhline(y=50.0, color="black", linestyle="--", linewidth=0.8,
               label="50% threshold")
    ax.set_ylabel("matched region %")
    ax.set_title("M-18 calibration coverage per model")
    ax.tick_params(axis="x", rotation=45)
    for lab in ax.get_xticklabels():
        lab.set_horizontalalignment("right")
    ax.legend(fontsize="small")
    _save(fig, path)


def _calibration_suite_scale(rows, path: Path) -> None:  # type: ignore[no-untyped-def]
    import matplotlib.pyplot as plt

    cal_rows = [
        r for r in rows
        if r.calibration_overall in ("calibrated", "partial")
        and r.calibration_suite_scale is not None
    ]
    fig, ax = plt.subplots(figsize=(max(6, len(cal_rows) * 0.45), 4.5))
    if not cal_rows:
        ax.text(0.5, 0.5, "no calibration data", ha="center", va="center")
        _save(fig, path); return
    labels = [r.model_id for r in cal_rows]
    scales = [r.calibration_suite_scale for r in cal_rows]
    ax.bar(labels, scales, color="#5f3dc4")
    ax.axhline(y=1.0, color="black", linestyle="--", linewidth=0.8,
               label="scale=1 (perfect)")
    ax.set_ylabel("suite_scale = measured / predicted")
    ax.set_title(
        "M-18 calibration suite_scale per model "
        "(>1 → roofline underestimates)"
    )
    ax.set_yscale("log")
    ax.tick_params(axis="x", rotation=45)
    for lab in ax.get_xticklabels():
        lab.set_horizontalalignment("right")
    ax.legend(fontsize="small")
    _save(fig, path)


def _transform_family_discharge(rows, path: Path) -> None:  # type: ignore[no-untyped-def]
    import matplotlib.pyplot as plt
    import numpy as np

    families = ["set_tile_params", "fuse_producer_consumer"]
    status_to_int = {
        "not_selected": 0, "n/a": 0,
        "blocked": 1, "fail": 2, "pass": 3,
    }
    rev = {0: "not selected", 1: "blocked", 2: "fail", 3: "pass"}

    fig, ax = plt.subplots(figsize=(max(5, len(rows) * 0.55), 4.5))
    if not rows:
        ax.text(0.5, 0.5, "no models", ha="center", va="center")
        _save(fig, path); return

    Z = np.zeros((len(rows), len(families)), dtype=int)
    for i, r in enumerate(rows):
        for j, fam in enumerate(families):
            if r.selected_candidate_kind != fam:
                Z[i, j] = 0
                continue
            if fam == "set_tile_params":
                st = r.real_set_tile_status
            else:
                st = r.real_fusion_status
            Z[i, j] = status_to_int.get(st, 0)
    cmap = ["#dee2e6", "#f08c00", "#c92a2a", "#2b8a3e"]
    from matplotlib.colors import ListedColormap
    cm = ListedColormap(cmap)
    im = ax.imshow(Z, cmap=cm, aspect="auto", vmin=0, vmax=3)
    ax.set_xticks(range(len(families)))
    ax.set_xticklabels(families)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r.model_id for r in rows])
    for i in range(len(rows)):
        for j in range(len(families)):
            ax.text(j, i, rev[int(Z[i, j])], ha="center", va="center",
                    fontsize=8, color="black")
    ax.set_title("Transform family discharge matrix")
    _save(fig, path)
