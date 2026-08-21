"""
Generate two plots about LP relaxation:
  1) plots/lp_relaxation_concept.png   - descriptive diagram of what LP relaxation is
  2) plots/lp_relaxation_speedup.png   - empirical solve-time comparison of MILP vs LP

The benchmark drives the same code paths the scheduler uses in production:
the MILP variant from scheduler.schedule_window (binary alpha/beta), and the
LP-relaxed variant from packing.lp_schedule (continuous alpha/beta in [0,1]).
"""

import os
import sys
import time
import random

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.patches import FancyArrowPatch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
XPURT_DIR = os.path.join(REPO_ROOT, "xpu-rt")
PLOTS_DIR = os.path.join(REPO_ROOT, "plots")
sys.path.insert(0, XPURT_DIR)

from workload import Operation, Workload, Window
from scheduler import schedule_window
from packing import lp_schedule


# ----------------------------------------------------------------------------
# 1) Conceptual LP relaxation diagram
# ----------------------------------------------------------------------------
def make_concept_plot(out_path: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))

    # LEFT: feasible regions
    ax = axes[0]
    polytope = np.array([
        [0.5, 0.5],
        [5.5, 0.5],
        [5.5, 4.3],
        [4.3, 5.5],
        [0.5, 5.5],
    ])
    ax.add_patch(MplPolygon(polytope, closed=True, facecolor="#cfe2ff",
                             edgecolor="#1f4ea8", lw=2, alpha=0.85, zorder=1))

    # Integer lattice points inside the polytope
    inside_pts = []
    for x in range(0, 7):
        for y in range(0, 7):
            if _point_in_polygon((x, y), polytope):
                inside_pts.append((x, y))
    inside_pts = np.array(inside_pts)
    ax.scatter(inside_pts[:, 0], inside_pts[:, 1],
               s=70, color="#1f4ea8", zorder=3,
               label="MILP feasible (integer points)")

    # Objective: maximize x + y (equivalently minimize -x - y).
    # LP optimum at the vertex (5.5, 4.3) with x+y = 9.8 (fractional).
    # MILP optimum at (5, 4) with x+y = 9 (best integer point).
    lp_opt = np.array([5.5, 4.3])
    milp_opt = np.array([5, 4])

    # Objective contour lines x + y = c
    for c, lbl_y_off in [(5, 0.18), (7, 0.18), (9, 0.18), (9.8, -0.30)]:
        xs = np.array([-0.5, 7.5])
        ys = c - xs
        style = "-" if abs(c - 9.8) < 1e-6 else "--"
        col = "#d62728" if abs(c - 9.8) < 1e-6 else "#666"
        lw = 1.4 if abs(c - 9.8) < 1e-6 else 0.9
        ax.plot(xs, ys, style, color=col, lw=lw, alpha=0.85, zorder=2)
        # place label inside the visible window
        x_lbl = 6.6
        y_lbl = c - x_lbl + lbl_y_off
        if -0.5 < y_lbl < 6.8:
            ax.text(x_lbl, y_lbl, f"x+y={c}", color=col, fontsize=8,
                    ha="left", va="bottom")

    ax.scatter(*lp_opt, s=300, marker="*", color="#d62728", zorder=5,
               edgecolor="black", linewidth=0.8,
               label=f"LP optimum  ({lp_opt[0]}, {lp_opt[1]})  — fractional")
    ax.scatter(*milp_opt, s=180, marker="D", color="#2ca02c", zorder=5,
               edgecolor="black", linewidth=0.8,
               label=f"MILP optimum  ({milp_opt[0]}, {milp_opt[1]})  — integer")

    arrow = FancyArrowPatch(lp_opt, milp_opt + np.array([0.05, 0.05]),
                            arrowstyle="->", mutation_scale=15,
                            color="black", lw=1.2, zorder=4)
    ax.add_patch(arrow)
    ax.text(4.7, 4.85, "rounding /\nbranch & bound",
            fontsize=9, ha="right", va="bottom")

    ax.text(0.6, 5.95, "LP-relaxed feasible region\n(continuous polytope)",
            fontsize=10, color="#1f4ea8", ha="left", va="top", fontweight="bold")

    ax.set_xlim(-0.5, 7.5)
    ax.set_ylim(-0.5, 7.0)
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_aspect("equal")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13),
              ncol=1, fontsize=9, framealpha=0.95)
    ax.set_title("What LP relaxation does\nDrop integrality; solve over the convex hull")

    # RIGHT: the variable-by-variable picture (binary -> [0,1])
    ax = axes[1]
    ax.set_xlim(-0.2, 5.2)
    ax.set_ylim(-0.2, 4.4)
    ax.axis("off")
    ax.set_title("In this scheduler", pad=14)

    # MILP box
    _draw_box(ax, x=0.2, y=2.55, w=4.6, h=1.6,
              face="#fde7e7", edge="#b22222",
              title="MILP (scheduler.schedule_window)",
              body=[
                  r"$\alpha_{i,k} \in \{0,1\}$   op $i$ on combo $k$",
                  r"$\beta_{i,j} \in \{0,1\}$    ordering of $i,j$",
                  "branch-and-bound over an exponential tree",
              ])

    # LP box
    _draw_box(ax, x=0.2, y=0.35, w=4.6, h=1.6,
              face="#e7f0fd", edge="#1f4ea8",
              title="LP relaxation (packing.lp_schedule)",
              body=[
                  r"$0 \leq \alpha_{i,k} \leq 1$   (continuous)",
                  r"$0 \leq \beta_{i,j} \leq 1$    (continuous)",
                  "interior-point on a convex polytope",
              ])

    arr = FancyArrowPatch((2.5, 2.55), (2.5, 1.95), arrowstyle="->",
                          mutation_scale=18, color="black", lw=1.5)
    ax.add_patch(arr)
    ax.text(2.6, 2.25, "relax integrality", fontsize=9, va="center")

    fig.suptitle("LP relaxation of the scheduling MILP", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def _point_in_polygon(point, polygon) -> bool:
    x, y = point
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi):
            inside = not inside
        j = i
    return inside


def _draw_box(ax, x, y, w, h, face, edge, title, body):
    ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=face, edgecolor=edge,
                                lw=2, zorder=1))
    ax.text(x + w / 2, y + h - 0.22, title, fontsize=11, fontweight="bold",
            ha="center", va="top", color=edge)
    for k, line in enumerate(body):
        ax.text(x + 0.2, y + h - 0.55 - 0.32 * k, line, fontsize=10,
                ha="left", va="top")


# ----------------------------------------------------------------------------
# 2) Empirical benchmark of MILP vs LP solve time
# ----------------------------------------------------------------------------
def _build_window(n_ops: int, n_machines: int, seed: int) -> Window:
    rng = random.Random(seed)
    machines = [f"M{i}" for i in range(n_machines)]
    machine_combinations = [[m] for m in machines]
    transfer_times = np.zeros((n_machines, n_machines))

    ops = []
    for i in range(n_ops):
        proc = [round(rng.uniform(1.0, 10.0), 2) for _ in range(n_machines)]
        preds = []
        # chain dependency to keep the DAG connected
        if i > 0 and rng.random() < 0.7:
            preds.append(ops[i - 1])
        # one extra random predecessor for some operations
        if i > 1 and rng.random() < 0.3:
            preds.append(ops[rng.randint(0, i - 2)])
        ops.append(Operation(processing_times=proc, predecessors=preds,
                             operation_id=i, operation_name=f"op{i}", job_id=0))
    return Window(time_frame=1000.0, operations=ops, machines=machines,
                  transfer_times=transfer_times,
                  machine_combinations=machine_combinations)


def _window_to_workload(window: Window) -> Workload:
    return Workload(
        operations=window.operations,
        machines=window.machines,
        transfer_times=window.transfer_times,
        job_names=["bench"],
        machine_combinations=window.machine_combinations,
    )


def run_benchmark(sizes, n_machines: int, n_trials: int):
    results = {n: {"milp": [], "lp": []} for n in sizes}
    for n in sizes:
        for trial in range(n_trials):
            seed = 1000 * n + trial
            print(f"[benchmark] n_ops={n} trial={trial} seed={seed}")
            window = _build_window(n, n_machines, seed)

            t0 = time.perf_counter()
            schedule_window(window, debug_constraints=False)
            t_milp = time.perf_counter() - t0

            workload = _window_to_workload(_build_window(n, n_machines, seed))
            t0 = time.perf_counter()
            lp_schedule(workload)
            t_lp = time.perf_counter() - t0

            results[n]["milp"].append(t_milp)
            results[n]["lp"].append(t_lp)
            print(f"  MILP={t_milp:.3f}s   LP={t_lp:.3f}s   speedup={t_milp/max(t_lp,1e-6):.1f}x")
    return results


def make_speedup_plot(results, out_path: str) -> None:
    sizes = sorted(results.keys())
    milp_med = np.array([np.median(results[n]["milp"]) for n in sizes])
    lp_med = np.array([np.median(results[n]["lp"]) for n in sizes])
    milp_min = np.array([np.min(results[n]["milp"]) for n in sizes])
    milp_max = np.array([np.max(results[n]["milp"]) for n in sizes])
    lp_min = np.array([np.min(results[n]["lp"]) for n in sizes])
    lp_max = np.array([np.max(results[n]["lp"]) for n in sizes])
    speedup = milp_med / np.maximum(lp_med, 1e-6)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))

    ax = axes[0]
    ax.fill_between(sizes, milp_min, milp_max, color="#b22222", alpha=0.18)
    ax.fill_between(sizes, lp_min, lp_max, color="#1f4ea8", alpha=0.18)
    ax.plot(sizes, milp_med, "o-", color="#b22222", lw=2, label="MILP (boolean αβ)")
    ax.plot(sizes, lp_med, "s-", color="#1f4ea8", lw=2, label="LP relaxation (αβ ∈ [0,1])")
    ax.set_yscale("log")
    ax.set_xlabel("number of operations in window")
    ax.set_ylabel("solver wall time (s, log scale)")
    ax.set_title("Solve time: MILP vs LP relaxation")
    ax.grid(alpha=0.3, which="both")
    ax.legend()

    ax = axes[1]
    bars = ax.bar(sizes, speedup, color="#2ca02c", edgecolor="black", lw=0.6)
    for x, v in zip(sizes, speedup):
        ax.text(x, v, f"{v:.0f}×", ha="center", va="bottom", fontsize=9)
    ax.set_xlabel("number of operations in window")
    ax.set_ylabel("MILP time / LP time")
    ax.set_title("Speed-up from LP relaxation")
    ax.grid(alpha=0.3, axis="y")
    ax.margins(y=0.18)

    fig.suptitle("Why LP relaxation matters for this scheduler", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    make_concept_plot(os.path.join(PLOTS_DIR, "lp_relaxation_concept.png"))

    sizes = [5, 8, 11, 14, 17]
    results = run_benchmark(sizes=sizes, n_machines=3, n_trials=2)
    make_speedup_plot(results, os.path.join(PLOTS_DIR, "lp_relaxation_speedup.png"))


if __name__ == "__main__":
    main()
