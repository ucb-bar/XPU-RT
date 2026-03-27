"""Benchmark plot generation — one function per plot type.

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
        raise ImportError(
            "matplotlib required for plots. Install with: pip install matplotlib"
        ) from e


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


# ---- Plot 1: Agentic convergence curve ----

def plot_convergence(records: list[RunRecord], output_dir: str | Path) -> Path:
    """Plot cost vs iteration for each run (agentic convergence curve).

    Shows how the optimization cost decreases over agent iterations.
    """
    plt = _require_matplotlib()
    fig, ax = plt.subplots(figsize=(10, 6))

    for record in records:
        if not record.agentic.iteration_costs:
            continue
        costs = record.agentic.iteration_costs
        label = f"{record.model_name}/{record.target_name}"
        if record.config.get("ablation"):
            label += f" ({record.config['ablation']})"
        ax.plot(range(len(costs)), costs, marker="o", markersize=4, label=label)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Estimated Cost (μs)")
    ax.set_title("Agentic Optimization Convergence")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    path = _ensure_dir(Path(output_dir)) / "convergence.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ---- Plot 2: Baseline comparison bar chart ----

def plot_baseline_comparison(records: list[RunRecord], output_dir: str | Path) -> Path:
    """Bar chart comparing CompGen vs baselines (eager CPU, eager GPU, compiled GPU)."""
    plt = _require_matplotlib()
    import numpy as np

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(records))
    width = 0.2
    labels = [f"{r.model_name}\n{r.target_name}" for r in records]

    eager_cpu = [r.baselines.eager_cpu_latency_us for r in records]
    eager_gpu = [r.baselines.eager_gpu_latency_us for r in records]
    compiled = [r.baselines.compiled_gpu_latency_us for r in records]
    compgen = [r.baselines.compgen_latency_us for r in records]

    ax.bar(x - 1.5 * width, eager_cpu, width, label="Eager CPU", color="#e74c3c", alpha=0.8)
    ax.bar(x - 0.5 * width, eager_gpu, width, label="Eager GPU", color="#3498db", alpha=0.8)
    ax.bar(x + 0.5 * width, compiled, width, label="torch.compile", color="#2ecc71", alpha=0.8)
    ax.bar(x + 1.5 * width, compgen, width, label="CompGen", color="#9b59b6", alpha=0.8)

    ax.set_xlabel("Model / Target")
    ax.set_ylabel("Latency (μs)")
    ax.set_title("End-to-End Latency: CompGen vs Baselines")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    path = _ensure_dir(Path(output_dir)) / "baseline_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ---- Plot 3: EqSat impact ----

def plot_eqsat_impact(records: list[RunRecord], output_dir: str | Path) -> Path:
    """Bar chart showing ops before/after eqsat and reduction percentage."""
    plt = _require_matplotlib()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    labels = [f"{r.model_name}" for r in records]
    before = [r.eqsat.ops_before for r in records]
    after = [r.eqsat.ops_after for r in records]
    reduction = [r.eqsat.ops_reduction_pct for r in records]

    x = range(len(records))
    ax1.bar([i - 0.2 for i in x], before, 0.4, label="Before", color="#e74c3c", alpha=0.8)
    ax1.bar([i + 0.2 for i in x], after, 0.4, label="After", color="#2ecc71", alpha=0.8)
    ax1.set_xlabel("Model")
    ax1.set_ylabel("Op Count")
    ax1.set_title("Op Count Before/After EqSat")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(labels, fontsize=8)
    ax1.legend()
    ax1.grid(True, axis="y", alpha=0.3)

    ax2.bar(x, reduction, color="#3498db", alpha=0.8)
    ax2.set_xlabel("Model")
    ax2.set_ylabel("Reduction (%)")
    ax2.set_title("EqSat Op Reduction")
    ax2.set_xticks(list(x))
    ax2.set_xticklabels(labels, fontsize=8)
    ax2.grid(True, axis="y", alpha=0.3)

    path = _ensure_dir(Path(output_dir)) / "eqsat_impact.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ---- Plot 4: Solver performance ----

def plot_solver_metrics(records: list[RunRecord], output_dir: str | Path) -> Path:
    """Solver time and optimality gap per run."""
    plt = _require_matplotlib()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    labels = [f"{r.model_name}" for r in records]
    placement_time = [r.solver.placement_time_ms for r in records]
    schedule_time = [r.solver.schedule_time_ms for r in records]
    memory_time = [r.solver.memory_time_ms for r in records]
    gap = [r.solver.placement_gap for r in records]

    x = range(len(records))
    ax1.bar([i - 0.25 for i in x], placement_time, 0.25, label="Placement", color="#e74c3c")
    ax1.bar(list(x), schedule_time, 0.25, label="Schedule", color="#3498db")
    ax1.bar([i + 0.25 for i in x], memory_time, 0.25, label="Memory", color="#2ecc71")
    ax1.set_xlabel("Model")
    ax1.set_ylabel("Solve Time (ms)")
    ax1.set_title("Solver Wall-Clock Time")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(labels, fontsize=8)
    ax1.legend()
    ax1.grid(True, axis="y", alpha=0.3)

    ax2.bar(x, gap, color="#9b59b6", alpha=0.8)
    ax2.set_xlabel("Model")
    ax2.set_ylabel("Optimality Gap")
    ax2.set_title("Placement Optimality Gap (0 = proven optimal)")
    ax2.set_xticks(list(x))
    ax2.set_xticklabels(labels, fontsize=8)
    ax2.axhline(y=0, color="black", linestyle="--", alpha=0.3)
    ax2.grid(True, axis="y", alpha=0.3)

    path = _ensure_dir(Path(output_dir)) / "solver_metrics.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ---- Plot 5: Recipe IR composition ----

def plot_recipe_composition(records: list[RunRecord], output_dir: str | Path) -> Path:
    """Stacked bar showing Recipe IR op family distribution."""
    plt = _require_matplotlib()
    import numpy as np

    fig, ax = plt.subplots(figsize=(12, 6))
    labels = [f"{r.model_name}\n{r.target_name}" for r in records]
    x = np.arange(len(records))

    families = ["scope_ops", "fact_ops", "candidate_ops", "choice_ops", "verify_ops", "provenance_ops"]
    colors = ["#1abc9c", "#3498db", "#9b59b6", "#e67e22", "#e74c3c", "#95a5a6"]
    family_labels = ["Scope", "Facts", "Candidates", "Choice", "Verification", "Provenance"]

    bottom = np.zeros(len(records))
    for family, color, label in zip(families, colors, family_labels):
        values = [getattr(r.recipe, family) for r in records]
        ax.bar(x, values, 0.6, bottom=bottom, label=label, color=color, alpha=0.85)
        bottom += np.array(values, dtype=float)

    ax.set_xlabel("Model / Target")
    ax.set_ylabel("Op Count")
    ax.set_title("Recipe IR Composition by Family")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)

    path = _ensure_dir(Path(output_dir)) / "recipe_composition.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ---- Plot 6: LLM cost breakdown ----

def plot_llm_cost(records: list[RunRecord], output_dir: str | Path) -> Path:
    """Token usage and cost per run."""
    plt = _require_matplotlib()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    labels = [f"{r.model_name}" for r in records]
    prompt_tokens = [r.llm.total_prompt_tokens for r in records]
    completion_tokens = [r.llm.total_completion_tokens for r in records]
    cost = [r.llm.total_cost_usd for r in records]

    x = range(len(records))
    ax1.bar([i - 0.2 for i in x], prompt_tokens, 0.4, label="Prompt", color="#3498db")
    ax1.bar([i + 0.2 for i in x], completion_tokens, 0.4, label="Completion", color="#e74c3c")
    ax1.set_xlabel("Model")
    ax1.set_ylabel("Tokens")
    ax1.set_title("LLM Token Usage")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(labels, fontsize=8)
    ax1.legend()
    ax1.grid(True, axis="y", alpha=0.3)

    ax2.bar(x, cost, color="#2ecc71", alpha=0.8)
    ax2.set_xlabel("Model")
    ax2.set_ylabel("Cost (USD)")
    ax2.set_title("LLM API Cost per Run")
    ax2.set_xticks(list(x))
    ax2.set_xticklabels(labels, fontsize=8)
    ax2.grid(True, axis="y", alpha=0.3)

    path = _ensure_dir(Path(output_dir)) / "llm_cost.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ---- Plot 7: Compile time breakdown ----

def plot_compile_time(records: list[RunRecord], output_dir: str | Path) -> Path:
    """Stacked bar showing time spent in each pipeline stage."""
    plt = _require_matplotlib()
    import numpy as np

    fig, ax = plt.subplots(figsize=(12, 6))
    labels = [f"{r.model_name}" for r in records]
    x = np.arange(len(records))

    capture = [r.capture.export_time_ms for r in records]
    eqsat = [r.eqsat.eqsat_time_ms for r in records]
    recipe = [r.recipe.seed_generation_time_ms for r in records]
    solver = [r.solver.placement_time_ms + r.solver.schedule_time_ms + r.solver.memory_time_ms for r in records]
    other = [max(r.total_compile_time_ms - c - e - re - s, 0)
             for r, c, e, re, s in zip(records, capture, eqsat, recipe, solver)]

    bottom = np.zeros(len(records))
    for values, label, color in [
        (capture, "Capture", "#e74c3c"),
        (eqsat, "EqSat", "#3498db"),
        (recipe, "Recipe", "#9b59b6"),
        (solver, "Solver", "#2ecc71"),
        (other, "Other", "#95a5a6"),
    ]:
        ax.bar(x, values, 0.6, bottom=bottom, label=label, color=color, alpha=0.85)
        bottom += np.array(values, dtype=float)

    ax.set_xlabel("Model")
    ax.set_ylabel("Time (ms)")
    ax.set_title("Compile Time Breakdown")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.legend(loc="upper right")
    ax.grid(True, axis="y", alpha=0.3)

    path = _ensure_dir(Path(output_dir)) / "compile_time.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ---- Plot 8: Verification ladder ----

def plot_verification_ladder(records: list[RunRecord], output_dir: str | Path) -> Path:
    """Heatmap-style grid showing pass/fail at each verification level per run."""
    plt = _require_matplotlib()

    fig, ax = plt.subplots(figsize=(10, max(3, len(records) * 0.6 + 1)))

    levels = ["Structural", "CHECK", "Differential", "Translation\nValidation"]
    data = []
    labels = []
    for r in records:
        v = r.verification
        row = [
            1 if v.structural_pass else 0,
            1 if v.check_assertions_pass else 0,
            1 if v.differential_pass else 0,
            1 if v.translation_validation_pass else (0.5 if v.translation_validation_pass is None else 0),
        ]
        data.append(row)
        labels.append(f"{r.model_name}/{r.target_name}")

    if not data:
        data = [[0, 0, 0, 0]]
        labels = ["(no data)"]

    import numpy as np
    data_arr = np.array(data)

    # Color: green=pass, red=fail, yellow=skip/None
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(["#e74c3c", "#f39c12", "#2ecc71"])
    ax.imshow(data_arr, cmap=cmap, vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(len(levels)))
    ax.set_xticklabels(levels, fontsize=9)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_title("Verification Ladder Results")

    # Add text annotations
    for i in range(len(labels)):
        for j in range(len(levels)):
            val = data_arr[i, j]
            text = "PASS" if val == 1 else ("SKIP" if val == 0.5 else "FAIL")
            ax.text(j, i, text, ha="center", va="center", fontsize=8, fontweight="bold",
                    color="white" if val != 0.5 else "black")

    path = _ensure_dir(Path(output_dir)) / "verification_ladder.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ---- Plot 9: Kernel strategy distribution ----

def plot_kernel_strategies(records: list[RunRecord], output_dir: str | Path) -> Path:
    """Pie/bar chart showing kernel strategy distribution across all runs."""
    plt = _require_matplotlib()

    # Aggregate strategy counts across all records
    total_hist: dict[str, int] = {}
    for r in records:
        for strategy, count in r.kernels.strategy_histogram.items():
            total_hist[strategy] = total_hist.get(strategy, 0) + count

    if not total_hist:
        total_hist = {"no_data": 1}

    fig, ax = plt.subplots(figsize=(8, 8))
    colors = {"native": "#2ecc71", "library": "#3498db", "autocomp": "#9b59b6",
              "exo": "#e67e22", "fallback": "#95a5a6", "unsupported": "#e74c3c"}
    pie_colors = [colors.get(k, "#bdc3c7") for k in total_hist]
    ax.pie(total_hist.values(), labels=total_hist.keys(), colors=pie_colors,
           autopct="%1.1f%%", startangle=90)
    ax.set_title("Kernel Strategy Distribution (All Runs)")

    path = _ensure_dir(Path(output_dir)) / "kernel_strategies.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ---- Plot 10: Latency distribution (box plot) ----

def plot_latency_distribution(records: list[RunRecord], output_dir: str | Path) -> Path:
    """Box plot of per-run latency distributions."""
    plt = _require_matplotlib()

    fig, ax = plt.subplots(figsize=(12, 6))
    data = []
    labels = []
    for r in records:
        if r.performance.per_run_us:
            data.append(r.performance.per_run_us)
            labels.append(f"{r.model_name}\n{r.target_name}")

    if data:
        bp = ax.boxplot(data, labels=labels, patch_artist=True)
        for patch in bp["boxes"]:
            patch.set_facecolor("#3498db")
            patch.set_alpha(0.7)
    else:
        ax.text(0.5, 0.5, "No latency data", transform=ax.transAxes, ha="center")

    ax.set_ylabel("Latency (μs)")
    ax.set_title("Runtime Latency Distribution")
    ax.grid(True, axis="y", alpha=0.3)

    path = _ensure_dir(Path(output_dir)) / "latency_distribution.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ---- Plot 11: Speedup CDF ----

def plot_speedup_cdf(records: list[RunRecord], output_dir: str | Path) -> Path:
    """CDF of speedup vs compiled baseline for comparable records."""
    plt = _require_matplotlib()

    speedups = sorted(
        r.baselines.speedup_vs_compiled
        for r in records
        if r.baselines.speedup_vs_compiled > 0
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    if speedups:
        yvals = [(i + 1) / len(speedups) for i in range(len(speedups))]
        ax.step(speedups, yvals, where="post", color="#1f77b4", linewidth=2)
    else:
        ax.text(0.5, 0.5, "No speedup data", transform=ax.transAxes, ha="center")

    ax.set_xlabel("Speedup vs torch.compile")
    ax.set_ylabel("CDF")
    ax.set_title("Speedup Distribution")
    ax.grid(True, alpha=0.3)

    path = _ensure_dir(Path(output_dir)) / "speedup_cdf.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ---- Plot 12: Bring-up effort ----

def plot_bringup_effort(records: list[RunRecord], output_dir: str | Path) -> Path:
    """Bring-up effort comparison across systems."""
    plt = _require_matplotlib()
    import numpy as np

    fig, ax = plt.subplots(figsize=(12, 6))
    labels = [f"{r.system_name}\n{r.model_name}/{r.target_name}" for r in records]
    x = np.arange(len(records))
    first_correct = [r.productivity.person_hours_to_first_correct for r in records]
    to_80 = [r.productivity.person_hours_to_80pct_expert for r in records]

    ax.bar(x - 0.2, first_correct, 0.4, label="First Correct", color="#34495e")
    ax.bar(x + 0.2, to_80, 0.4, label="80% Expert", color="#e67e22")
    ax.set_ylabel("Person Hours")
    ax.set_title("Bring-up Effort")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    path = _ensure_dir(Path(output_dir)) / "bringup_effort.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ---- Plot 13: Artifact completeness ----

def plot_artifact_completeness(records: list[RunRecord], output_dir: str | Path) -> Path:
    """Bar chart of bundle artifact completeness."""
    plt = _require_matplotlib()

    fig, ax = plt.subplots(figsize=(12, 6))
    labels = [f"{r.system_name}\n{r.model_name}/{r.target_name}" for r in records]
    completeness = [r.artifacts.completeness_score * 100 for r in records]
    ax.bar(range(len(records)), completeness, color="#2ecc71", alpha=0.85)
    ax.set_ylabel("Completeness (%)")
    ax.set_title("Artifact Completeness")
    ax.set_xticks(range(len(records)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, 100)
    ax.grid(True, axis="y", alpha=0.3)

    path = _ensure_dir(Path(output_dir)) / "artifact_completeness.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ---- Plot 14: Verification catch matrix ----

def plot_verification_catch_matrix(records: list[RunRecord], output_dir: str | Path) -> Path:
    """Heatmap of caught defects by verification layer."""
    plt = _require_matplotlib()
    import numpy as np

    levels = sorted({level for record in records for level in record.verification.caught_by_level} or {"none"})
    fig, ax = plt.subplots(figsize=(10, max(3, len(records) * 0.6 + 1)))
    matrix = []
    labels = []
    for record in records:
        matrix.append([record.verification.caught_by_level.get(level, 0) for level in levels])
        labels.append(f"{record.system_name}/{record.model_name}")
    matrix_arr = np.array(matrix or [[0]])
    ax.imshow(matrix_arr, aspect="auto", cmap="Blues")
    ax.set_xticks(range(len(levels)))
    ax.set_xticklabels(levels, fontsize=8, rotation=30, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_title("Verification Catch Matrix")

    path = _ensure_dir(Path(output_dir)) / "verification_catch_matrix.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ---- Plot 15: Proposal funnel ----

def plot_proposal_funnel(records: list[RunRecord], output_dir: str | Path) -> Path:
    """Proposal-to-promotion funnel for the CompGen runs."""
    plt = _require_matplotlib()
    import numpy as np

    compgen_records = [r for r in records if r.system_name == "compgen"]
    explored = sum(r.generation.candidate_recipes_explored for r in compgen_records)
    transforms = sum(r.generation.candidate_transforms for r in compgen_records)
    rejected = sum(r.generation.rejected_by_verification for r in compgen_records)
    promoted = sum(r.generation.promoted_candidates for r in compgen_records)

    values = [explored, transforms, max(explored - rejected, 0), promoted]
    labels = ["Explored", "Transforms", "Accepted", "Promoted"]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(np.arange(len(values)), values, color=["#95a5a6", "#3498db", "#2ecc71", "#9b59b6"])
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels(labels)
    ax.set_title("Proposal-to-Promotion Funnel")
    ax.grid(True, axis="y", alpha=0.3)

    path = _ensure_dir(Path(output_dir)) / "proposal_funnel.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ---- Master plot function ----

def generate_all_plots(records: list[RunRecord], output_dir: str | Path) -> list[Path]:
    """Generate all benchmark plots from a list of RunRecords.

    Returns list of generated plot file paths.
    """
    output_dir = Path(output_dir)
    paths = []

    plot_fns = [
        plot_convergence,
        plot_baseline_comparison,
        plot_eqsat_impact,
        plot_solver_metrics,
        plot_recipe_composition,
        plot_llm_cost,
        plot_compile_time,
        plot_verification_ladder,
        plot_kernel_strategies,
        plot_latency_distribution,
        plot_speedup_cdf,
        plot_bringup_effort,
        plot_artifact_completeness,
        plot_verification_catch_matrix,
        plot_proposal_funnel,
    ]

    for fn in plot_fns:
        try:
            path = fn(records, output_dir)
            paths.append(path)
        except Exception as e:
            log.warning("plot.failed %s: %s", fn.__name__, e)

    return paths


__all__ = [
    "generate_all_plots",
    "plot_baseline_comparison",
    "plot_compile_time",
    "plot_convergence",
    "plot_eqsat_impact",
    "plot_kernel_strategies",
    "plot_latency_distribution",
    "plot_llm_cost",
    "plot_bringup_effort",
    "plot_artifact_completeness",
    "plot_proposal_funnel",
    "plot_recipe_composition",
    "plot_solver_metrics",
    "plot_speedup_cdf",
    "plot_verification_catch_matrix",
    "plot_verification_ladder",
]
