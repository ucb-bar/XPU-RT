"""M-17.1 readiness figures — matplotlib only, no seaborn.

Five PNGs derived from the readiness reports. Each figure is a
self-contained summary; render fails are caught at the caller level so
they never break the pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _setup() -> None:
    import matplotlib
    matplotlib.use("Agg")


def _save(fig, path: Path) -> None:  # type: ignore[no-untyped-def]
    fig.tight_layout()
    fig.savefig(path, format="png", dpi=110)
    import matplotlib.pyplot as plt
    plt.close(fig)


def render_all(
    *, figures_dir: Path,
    precision: dict[str, Any],
    working_set: dict[str, Any],
    reuse: dict[str, Any],
    counterfactual: dict[str, Any],
    hw: dict[str, Any],
) -> None:
    _setup()
    figures_dir.mkdir(parents=True, exist_ok=True)
    _precision_budget(precision, figures_dir / "precision_budget_by_region.png")
    _working_set_fit(working_set, figures_dir / "working_set_fit_by_tile.png")
    _reuse_lifetime(reuse, figures_dir / "reuse_lifetime_histogram.png")
    _counterfactual(counterfactual, figures_dir / "candidate_counterfactual_coverage.png")
    _bottleneck(hw, figures_dir / "bottleneck_by_region.png")


def _precision_budget(report: dict, path: Path) -> None:
    import matplotlib.pyplot as plt

    regions = report.get("regions") or []
    fig, ax = plt.subplots(figsize=(max(6, len(regions) * 0.3), 4.5))
    if not regions:
        ax.text(0.5, 0.5, "no regions", ha="center", va="center")
        _save(fig, path); return

    dtypes = ("fp32", "fast_math", "fp16_accum", "fp8_e4m3")
    colors = {"fp32": "#2b8a3e", "fast_math": "#74b816",
              "fp16_accum": "#f08c00", "fp8_e4m3": "#c92a2a"}
    labels = [r["region_id"][:18] for r in regions]
    x = list(range(len(regions)))
    width = 0.2
    for i, dt in enumerate(dtypes):
        vals = []
        for r in regions:
            ds = (r.get("dtype_sensitivity") or {}).get(dt) or {}
            v = ds.get("budget_used_fraction")
            vals.append(min(v, 5.0) if isinstance(v, (int, float)) else 0.0)
        ax.bar(
            [xi + (i - 1.5) * width for xi in x], vals,
            width=width, color=colors[dt], label=dt,
        )
    ax.axhline(y=1.0, color="black", linestyle="--", linewidth=0.8,
               label="budget=1.0")
    ax.set_ylabel("budget_used_fraction (capped at 5)")
    ax.set_title("Precision-budget usage per region (per dtype)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.legend(fontsize="small", ncol=2)
    _save(fig, path)


def _working_set_fit(report: dict, path: Path) -> None:
    import matplotlib.pyplot as plt

    regions = report.get("regions") or []
    flat: list[tuple[str, str, int, bool, bool]] = []
    for r in regions:
        for t in r.get("candidate_tiles", []):
            label = f"{r['region_id'][:14]}/{t.get('label','')[:14]}"
            flat.append((
                r["region_id"], label,
                int(t.get("live_bytes") or 0),
                bool(t.get("fits_scratchpad")),
                bool(t.get("fits_l2")),
            ))

    fig, ax = plt.subplots(figsize=(max(6, len(flat) * 0.18), 4.5))
    if not flat:
        ax.text(0.5, 0.5, "no tile candidates", ha="center", va="center")
        _save(fig, path); return

    flat = sorted(flat, key=lambda t: t[2])
    labels = [f[1] for f in flat]
    bytes_ = [f[2] for f in flat]
    colors = []
    for f in flat:
        if f[3]:    colors.append("#2b8a3e")  # fits scratchpad
        elif f[4]:  colors.append("#f08c00")  # fits L2 only
        else:       colors.append("#c92a2a")  # exceeds L2
    ax.bar(range(len(flat)), bytes_, color=colors)
    tiers = report.get("memory_tiers", {}) or {}
    if tiers.get("scratchpad_bytes"):
        ax.axhline(y=tiers["scratchpad_bytes"], color="green",
                   linestyle="--", linewidth=0.8, label="scratchpad")
    if tiers.get("l2_bytes"):
        ax.axhline(y=tiers["l2_bytes"], color="orange",
                   linestyle="--", linewidth=0.8, label="L2")
    ax.set_yscale("log")
    ax.set_ylabel("live_bytes (log)")
    ax.set_title("Working-set fit per tile candidate")
    ax.set_xticks(range(len(flat)))
    ax.set_xticklabels(labels, rotation=70, ha="right", fontsize=6)
    ax.legend(fontsize="small")
    _save(fig, path)


def _reuse_lifetime(report: dict, path: Path) -> None:
    import matplotlib.pyplot as plt

    horizons: list[int] = []
    classes: dict[str, int] = {}
    for region in report.get("regions") or []:
        for t in region.get("outputs") or []:
            h = t.get("reuse_horizon")
            if isinstance(h, int):
                horizons.append(h)
            cls = t.get("producer_lifetime_class") or "unknown"
            classes[cls] = classes.get(cls, 0) + 1

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ax1, ax2 = axes
    if horizons:
        ax1.hist(horizons, bins=range(0, max(horizons) + 2), color="#1971c2")
        ax1.set_xlabel("reuse_horizon")
        ax1.set_ylabel("tensor count")
        ax1.set_title("Reuse-horizon distribution")
    else:
        ax1.text(0.5, 0.5, "no reuse_horizon data", ha="center", va="center")
    if classes:
        ax2.bar(list(classes.keys()), list(classes.values()), color="#5f3dc4")
        ax2.set_title("Producer lifetime classes")
        ax2.tick_params(axis="x", rotation=30)
        for lab in ax2.get_xticklabels():
            lab.set_horizontalalignment("right")
    else:
        ax2.text(0.5, 0.5, "no lifetime data", ha="center", va="center")
    _save(fig, path)


def _counterfactual(report: dict, path: Path) -> None:
    import matplotlib.pyplot as plt

    s = report.get("summary") or {}
    fig, ax = plt.subplots(figsize=(7, 4.5))
    keys = ("with_recipe_delta", "with_action_space_ir_block",
            "with_cost_preview", "with_legality")
    vals = [int(s.get(k, 0)) for k in keys]
    total = int(s.get("candidate_count", 0))
    ax.bar(keys, vals, color="#0c8599")
    if total > 0:
        ax.axhline(y=total, color="black", linestyle="--", linewidth=0.8,
                   label=f"total={total}")
    ax.set_ylabel("candidate count")
    ax.set_title("Counterfactual coverage per candidate")
    ax.tick_params(axis="x", rotation=20)
    for lab in ax.get_xticklabels():
        lab.set_horizontalalignment("right")
    ax.legend(fontsize="small")
    _save(fig, path)


def _bottleneck(report: dict, path: Path) -> None:
    import matplotlib.pyplot as plt

    regions = report.get("regions") or []
    fig, ax = plt.subplots(figsize=(max(6, len(regions) * 0.3), 4.5))
    if not regions:
        ax.text(0.5, 0.5, "no regions", ha="center", va="center")
        _save(fig, path); return

    color_map = {
        "compute": "#1971c2", "memory": "#e67700",
        "opaque": "#adb5bd", "unknown": "#adb5bd",
    }
    colors = [color_map.get(r["bottleneck_resource"], "#adb5bd") for r in regions]
    latencies = [
        float(r.get("estimated_latency_us") or 0) for r in regions
    ]
    labels = [r["region_id"][:18] for r in regions]
    ax.bar(labels, latencies, color=colors)
    ax.set_ylabel("estimated_latency_us")
    ax.set_title("Bottleneck resource per region (color)")
    ax.tick_params(axis="x", rotation=45)
    for lab in ax.get_xticklabels():
        lab.set_horizontalalignment("right")
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=c) for c in color_map.values()
    ]
    ax.legend(handles, list(color_map.keys()), fontsize="small")
    _save(fig, path)
