"""Paper-quality codegen plot generation for CompGen's MLSys submission.

Eight plot functions targeting code-generation-specific metrics: pipeline
coverage, strategy mix, roofline gaps, search cost-benefit, multi-device
planning, recipe scale payoff, agentic trajectories, and ablation heatmaps.

All plots save to PNG files. matplotlib is lazy-imported.
"""

from __future__ import annotations

import logging
from pathlib import Path

from benchmarks.record import RunRecord

log = logging.getLogger(__name__)


def _require_matplotlib():
    """Lazy import matplotlib."""
    try:
        import matplotlib

        matplotlib.use("Agg")  # non-interactive backend
        import matplotlib.pyplot as plt

        return plt
    except ImportError as e:
        raise ImportError("matplotlib required for plots. Install with: pip install matplotlib") from e


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


# ---- Plot 1: Coverage waterfall ----


def plot_coverage_waterfall(records: list[RunRecord], output_dir: str | Path) -> Path:
    """Stacked horizontal bar chart showing pipeline coverage per model.

    Each model gets a row of stacked segments representing how far it
    progressed through the CompGen pipeline: Captured, Imported, Contracts,
    Strategies, Verified, Bundled.
    """
    plt = _require_matplotlib()
    import numpy as np

    path = _ensure_dir(Path(output_dir)) / "coverage_waterfall.png"

    if not records:
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.text(0.5, 0.5, "No records", transform=ax.transAxes, ha="center")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    stage_names = ["Captured", "Imported", "Contracts", "Strategies", "Verified", "Bundled"]
    greens = ["#d5f5e3", "#a9dfbf", "#7dcea0", "#52be80", "#27ae60", "#1e8449"]

    labels = [r.model_name for r in records]
    data = []
    for r in records:
        row = [
            1 if r.capture.export_success else 0,
            1 if r.ir.total_ops > 0 else 0,
            1 if r.kernels.total_kernel_specs > 0 else 0,
            1 if bool(r.kernels.strategy_histogram) else 0,
            1 if r.verification.overall_status == "pass" else 0,
            1 if r.artifacts.completeness_score > 0.5 else 0,
        ]
        data.append(row)

    data_arr = np.array(data, dtype=float)
    y = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(10, max(3, len(labels) * 0.5 + 1)))
    left = np.zeros(len(labels))
    for col_idx, (stage, color) in enumerate(zip(stage_names, greens)):
        widths = data_arr[:, col_idx]
        ax.barh(y, widths, left=left, height=0.6, label=stage, color=color, edgecolor="white", linewidth=0.5)
        left += widths

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Pipeline Stages Completed")
    ax.set_title("Pipeline Coverage Waterfall")
    ax.set_xlim(0, len(stage_names) + 0.2)
    ax.legend(loc="lower right", fontsize=7, ncol=2)
    ax.grid(True, axis="x", alpha=0.3)
    ax.invert_yaxis()

    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ---- Plot 2: Strategy mix ----


def plot_strategy_mix(records: list[RunRecord], output_dir: str | Path) -> Path:
    """Stacked percentage bar chart showing kernel strategy distribution per model.

    Each bar is normalized to 100% and coloured by strategy lane:
    native, library, ukernel, autocomp, exo, fallback.
    """
    plt = _require_matplotlib()
    import numpy as np

    path = _ensure_dir(Path(output_dir)) / "strategy_mix.png"

    filtered = [r for r in records if r.kernels.strategy_histogram]
    if not filtered:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.text(0.5, 0.5, "No strategy data", transform=ax.transAxes, ha="center")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    strategy_order = ["native", "library", "ukernel", "autocomp", "exo", "fallback"]
    strategy_colors = {
        "native": "#2ecc71",
        "library": "#3498db",
        "ukernel": "#1abc9c",
        "autocomp": "#e67e22",
        "exo": "#9b59b6",
        "fallback": "#e74c3c",
    }

    labels = [r.model_name for r in filtered]
    x = np.arange(len(labels))

    # Build normalized fractions per strategy
    fractions: dict[str, list[float]] = {s: [] for s in strategy_order}
    for r in filtered:
        total = sum(r.kernels.strategy_histogram.values()) or 1
        for s in strategy_order:
            fractions[s].append(r.kernels.strategy_histogram.get(s, 0) / total * 100)

    fig, ax = plt.subplots(figsize=(10, 5))
    bottom = np.zeros(len(labels))
    for strategy in strategy_order:
        vals = np.array(fractions[strategy])
        ax.bar(
            x,
            vals,
            bottom=bottom,
            width=0.6,
            label=strategy,
            color=strategy_colors[strategy],
            edgecolor="white",
            linewidth=0.5,
        )
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8, rotation=30, ha="right")
    ax.set_ylabel("Strategy Share (%)")
    ax.set_title("Kernel Strategy Mix per Model")
    ax.set_ylim(0, 105)
    ax.legend(loc="upper right", fontsize=7, ncol=2)
    ax.grid(True, axis="y", alpha=0.3)

    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ---- Plot 3: Roofline gap ----


def plot_roofline_gap(records: list[RunRecord], output_dir: str | Path) -> Path:
    """Bar chart showing how close generated code is to roofline targets.

    Only includes records with a non-zero roofline_gap. Bars are coloured
    green/yellow/red based on the gap magnitude.
    """
    plt = _require_matplotlib()
    import numpy as np

    path = _ensure_dir(Path(output_dir)) / "roofline_gap.png"

    filtered = [r for r in records if r.kernels.roofline_gap > 0]
    if not filtered:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.text(0.5, 0.5, "No roofline data", transform=ax.transAxes, ha="center")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    labels = [r.model_name for r in filtered]
    gaps = [r.kernels.roofline_gap for r in filtered]
    colors = []
    for g in gaps:
        if g < 1.5:
            colors.append("#2ecc71")
        elif g < 3.0:
            colors.append("#f1c40f")
        else:
            colors.append("#e74c3c")

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x, gaps, width=0.6, color=colors, edgecolor="white", linewidth=0.5)
    ax.axhline(y=1.0, color="black", linestyle="--", linewidth=1, label="Optimal (1.0x)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8, rotation=30, ha="right")
    ax.set_ylabel("Roofline Gap (measured / target)")
    ax.set_title("Roofline Gap per Model")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)

    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ---- Plot 4: Speedup vs search cost ----


def plot_speedup_vs_search_cost(records: list[RunRecord], output_dir: str | Path) -> Path:
    """Scatter plot showing cost-benefit of code generation search.

    Each point is a per-region detail with search iterations on X and
    speedup vs reference on Y, coloured by the selected strategy.
    """
    plt = _require_matplotlib()

    path = _ensure_dir(Path(output_dir)) / "speedup_vs_search_cost.png"

    strategy_colors = {
        "native": "#2ecc71",
        "library": "#3498db",
        "ukernel": "#1abc9c",
        "autocomp": "#e67e22",
        "exo": "#9b59b6",
        "fallback": "#e74c3c",
    }

    xs: list[float] = []
    ys: list[float] = []
    cs: list[str] = []

    for r in records:
        for detail in r.kernels.region_details:
            speedup = detail.get("speedup_vs_reference", 0)
            if speedup <= 0:
                continue
            iters = detail.get("search_iterations_used", 0)
            if iters <= 0:
                # Fall back to record-level total_search_time_ms as proxy
                iters = r.kernels.total_search_time_ms
            if iters <= 0:
                continue
            strategy = detail.get("selected_strategy", "unknown")
            xs.append(float(iters))
            ys.append(float(speedup))
            cs.append(strategy_colors.get(strategy, "#95a5a6"))

    fig, ax = plt.subplots(figsize=(8, 6))
    if xs:
        ax.scatter(xs, ys, c=cs, alpha=0.7, edgecolors="white", linewidth=0.5, s=50)
        ax.axhline(y=1.0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)

        # Legend from strategy colors
        from matplotlib.lines import Line2D

        handles = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=8, label=s)
            for s, c in strategy_colors.items()
        ]
        ax.legend(handles=handles, fontsize=7, loc="upper left")
    else:
        ax.text(0.5, 0.5, "No region details with speedup data", transform=ax.transAxes, ha="center")

    ax.set_xlabel("Search Iterations / Time (ms)")
    ax.set_ylabel("Speedup vs Reference")
    ax.set_title("Speedup vs Search Cost (per Region)")
    ax.grid(True, alpha=0.3)

    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ---- Plot 5: Multi-device planning ----


def plot_multidevice_planning(records: list[RunRecord], output_dir: str | Path) -> Path:
    """Grouped bar chart comparing schedule makespan and solver overhead.

    Only includes records where placement is feasible. Shows makespan as
    main bar with hatched overlay for total solver wall-clock time.
    """
    plt = _require_matplotlib()
    import numpy as np

    path = _ensure_dir(Path(output_dir)) / "multidevice_planning.png"

    filtered = [r for r in records if r.solver.placement_feasible]
    if not filtered:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.text(0.5, 0.5, "No feasible placements", transform=ax.transAxes, ha="center")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    labels = [r.model_name for r in filtered]
    makespans = [r.solver.schedule_makespan_us for r in filtered]
    solver_overheads = [
        r.solver.placement_time_ms + r.solver.schedule_time_ms + r.solver.memory_time_ms for r in filtered
    ]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(10, 5))

    bars_makespan = ax1.bar(x - width / 2, makespans, width, label="Schedule Makespan", color="#3498db", alpha=0.85)
    ax1.set_ylabel("Schedule Makespan (us)", color="#3498db")
    ax1.tick_params(axis="y", labelcolor="#3498db")

    ax2 = ax1.twinx()
    bars_solver = ax2.bar(
        x + width / 2, solver_overheads, width, label="Solver Overhead", color="#e67e22", alpha=0.7, hatch="//"
    )
    ax2.set_ylabel("Solver Overhead (ms)", color="#e67e22")
    ax2.tick_params(axis="y", labelcolor="#e67e22")

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=8, rotation=30, ha="right")
    ax1.set_title("Multi-Device Planning: Makespan vs Solver Overhead")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)
    ax1.grid(True, axis="y", alpha=0.3)

    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ---- Plot 6: Recipe scale payoff ----


def plot_recipe_scale_payoff(records: list[RunRecord], output_dir: str | Path) -> Path:
    """Scatter plot showing recipe complexity vs speedup payoff.

    X = total recipe ops, Y = speedup vs compiled baseline, point size
    proportional to model complexity (total payload IR ops).
    """
    plt = _require_matplotlib()

    path = _ensure_dir(Path(output_dir)) / "recipe_scale_payoff.png"

    filtered = [r for r in records if r.recipe.total_recipe_ops > 0]
    if not filtered:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, "No recipe data", transform=ax.transAxes, ha="center")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    recipe_ops = [r.recipe.total_recipe_ops for r in filtered]
    speedups = [r.baselines.speedup_vs_compiled for r in filtered]
    ir_ops = [max(r.ir.total_ops, 1) for r in filtered]
    names = [r.model_name for r in filtered]

    # Normalize point sizes: 30..300 range
    max_ir = max(ir_ops) or 1
    sizes = [30 + 270 * (o / max_ir) for o in ir_ops]

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(recipe_ops, speedups, s=sizes, c="#3498db", alpha=0.7, edgecolors="white", linewidth=0.5)

    for i, name in enumerate(names):
        ax.annotate(
            name,
            (recipe_ops[i], speedups[i]),
            fontsize=6,
            ha="left",
            va="bottom",
            xytext=(4, 4),
            textcoords="offset points",
        )

    ax.axhline(y=1.0, color="black", linestyle="--", linewidth=0.8, alpha=0.5, label="Breakeven")
    ax.set_xlabel("Recipe IR Ops")
    ax.set_ylabel("Speedup vs torch.compile")
    ax.set_title("Recipe Complexity vs Speedup Payoff")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ---- Plot 7: Agentic outcome trajectories ----


def plot_agentic_outcome(records: list[RunRecord], output_dir: str | Path) -> Path:
    """Line plot showing agentic loop improvement trajectories.

    One line per model that has iteration_improvements populated. The
    final total_improvement_pct is annotated at the last point.
    """
    plt = _require_matplotlib()
    import numpy as np

    path = _ensure_dir(Path(output_dir)) / "agentic_outcome.png"

    filtered = [r for r in records if r.agentic.iteration_improvements]
    if not filtered:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "No agentic iteration data", transform=ax.transAxes, ha="center")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    fig, ax = plt.subplots(figsize=(10, 6))
    for r in filtered:
        improvements = r.agentic.iteration_improvements
        cumulative = list(np.cumsum(improvements))
        iters = list(range(1, len(cumulative) + 1))
        label = r.model_name
        (line,) = ax.plot(iters, cumulative, marker="o", markersize=4, linewidth=1.5, label=label)

        # Annotate final improvement
        final_pct = r.agentic.total_improvement_pct
        if final_pct > 0 and cumulative:
            ax.annotate(
                f"{final_pct:.1f}%",
                (iters[-1], cumulative[-1]),
                fontsize=7,
                fontweight="bold",
                xytext=(6, 4),
                textcoords="offset points",
                color=line.get_color(),
            )

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Cumulative Improvement (%)")
    ax.set_title("Agentic Loop Improvement Trajectories")
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5, alpha=0.3)

    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ---- Plot 8: Ablation heatmap ----


def plot_ablation_heatmap(records: list[RunRecord], output_dir: str | Path) -> Path:
    """Heatmap showing ablation study results.

    Rows = unique model names, columns = unique ablation configs. Cell
    values are speedup_vs_compiled (or total_compile_time_ms as fallback).
    Uses a diverging RdYlGn colormap centered at 1.0.
    """
    plt = _require_matplotlib()
    import numpy as np

    path = _ensure_dir(Path(output_dir)) / "ablation_heatmap.png"

    if not records:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.text(0.5, 0.5, "No records", transform=ax.transAxes, ha="center")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    # Collect unique models and ablation configs (preserving order)
    model_order: list[str] = []
    config_order: list[str] = []
    seen_models: set[str] = set()
    seen_configs: set[str] = set()
    for r in records:
        if r.model_name not in seen_models:
            model_order.append(r.model_name)
            seen_models.add(r.model_name)
        ablation = r.config.get("ablation", "full")
        if ablation not in seen_configs:
            config_order.append(ablation)
            seen_configs.add(ablation)

    if not model_order or not config_order:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.text(0.5, 0.5, "No ablation data", transform=ax.transAxes, ha="center")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    # Build the value matrix (NaN where no data)
    matrix = np.full((len(model_order), len(config_order)), np.nan)
    for r in records:
        row = model_order.index(r.model_name)
        col = config_order.index(r.config.get("ablation", "full"))
        val = r.baselines.speedup_vs_compiled
        if val <= 0:
            val = r.total_compile_time_ms
        matrix[row, col] = val

    fig, ax = plt.subplots(figsize=(max(6, len(config_order) * 1.2 + 2), max(4, len(model_order) * 0.6 + 1.5)))

    # Determine vmin/vmax centered on 1.0
    valid = matrix[~np.isnan(matrix)]
    if len(valid) > 0:
        max_dev = max(abs(valid.max() - 1.0), abs(valid.min() - 1.0), 0.5)
        vmin, vmax = 1.0 - max_dev, 1.0 + max_dev
    else:
        vmin, vmax = 0.0, 2.0

    im = ax.imshow(matrix, cmap="RdYlGn", vmin=vmin, vmax=vmax, aspect="auto")
    fig.colorbar(im, ax=ax, label="Speedup vs torch.compile", shrink=0.8)

    ax.set_xticks(range(len(config_order)))
    ax.set_xticklabels(config_order, fontsize=8, rotation=30, ha="right")
    ax.set_yticks(range(len(model_order)))
    ax.set_yticklabels(model_order, fontsize=8)
    ax.set_title("Ablation Study: Speedup by Configuration")

    # Annotate cells
    for i in range(len(model_order)):
        for j in range(len(config_order)):
            val = matrix[i, j]
            if np.isnan(val):
                ax.text(j, i, "--", ha="center", va="center", fontsize=7, color="grey")
            else:
                text_color = "white" if abs(val - 1.0) > max_dev * 0.6 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7, fontweight="bold", color=text_color)

    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ---- Master function ----


def generate_all_codegen_plots(records: list[RunRecord], output_dir: str | Path) -> list[Path]:
    """Generate all eight codegen paper plots.

    Each plot is wrapped in try/except so a single failure does not
    block the remaining plots. Returns list of successfully saved paths.
    """
    output_dir = Path(output_dir)
    paths: list[Path] = []

    plot_fns = [
        plot_coverage_waterfall,
        plot_strategy_mix,
        plot_roofline_gap,
        plot_speedup_vs_search_cost,
        plot_multidevice_planning,
        plot_recipe_scale_payoff,
        plot_agentic_outcome,
        plot_ablation_heatmap,
    ]

    for fn in plot_fns:
        try:
            p = fn(records, output_dir)
            paths.append(p)
        except Exception as e:
            log.warning("plot.failed %s: %s", fn.__name__, e)

    return paths


__all__ = [
    "generate_all_codegen_plots",
    "plot_ablation_heatmap",
    "plot_agentic_outcome",
    "plot_coverage_waterfall",
    "plot_multidevice_planning",
    "plot_recipe_scale_payoff",
    "plot_roofline_gap",
    "plot_speedup_vs_search_cost",
    "plot_strategy_mix",
]
