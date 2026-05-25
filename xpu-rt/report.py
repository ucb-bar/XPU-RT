"""
Comparison-report writer for XPU-RT benchmark sweeps.

Takes a list of per-scheduler results for one benchmark cell and produces:

  1. A Markdown summary table covering every metric in ``metrics.py``.
  2. A Gantt PNG per scheduler (delegated to ``plot.plot_optimization_schedule``).
  3. A side-by-side Gantt composite (matplotlib subplot grid, one row per scheduler).

Memory ``feedback_gantt_per_cell`` is load-bearing here: every sweep MUST emit
a Gantt per cell; infeasible cells are logged in the markdown rather than skipped.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import plot as xpurt_plot
from workload import Workload


@dataclass
class SchedulerResult:
    """One scheduler's outcome on a single benchmark cell."""
    scheduler_name: str
    workload: Optional[Workload] = None
    t: Optional[np.ndarray] = None
    alpha: Optional[np.ndarray] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    schedule_json_path: Optional[str] = None
    gantt_png_path: Optional[str] = None
    feasible: bool = True
    note: str = ""


# ----- Gantt emission ---------------------------------------------------------


def render_gantt(result: SchedulerResult, save_path: str, title: Optional[str] = None) -> Optional[str]:
    """Render a single Gantt PNG for ``result`` via the existing plot module.

    Returns the saved path, or ``None`` if the cell was infeasible.
    """
    if not result.feasible or result.workload is None or result.t is None or result.alpha is None:
        return None

    workload = result.workload
    num_jobs = sum(1 for op in workload.operations if not op.predecessors) or 1
    xpurt_plot.plot_optimization_schedule(
        workload.get_durations(),
        result.t,
        result.alpha,
        num_jobs,
        len(workload.machines),
        workload.machines,
        workload.get_transfer_times(),
        save_path=save_path,
        plot_title=title or f"{result.scheduler_name} schedule",
        workload=workload,
    )
    result.gantt_png_path = save_path
    return save_path


def _render_gantt_to_ax(ax, result: SchedulerResult) -> None:
    """Draw a compact Gantt for ``result`` directly onto ``ax``.

    Each machine becomes one row; bars are coloured by job_id; small black
    rectangles on the right edge mark intra-row transfer time. Designed for
    side-by-side composites so every panel reads at full resolution.
    """
    wl = result.workload
    t = result.t
    alpha = result.alpha
    combos = wl.get_machine_combinations()
    machines = list(wl.machines)
    n = len(wl.operations)
    if t is None or alpha is None or n == 0:
        ax.text(0.5, 0.5, "infeasible", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return

    n_jobs = max((op.job_id or 0 for op in wl.operations), default=0) + 1
    cmap = plt.get_cmap("tab20")
    y_of = {m: i for i, m in enumerate(machines)}

    finish = []
    for i, op in enumerate(wl.operations):
        k = int(np.argmax(alpha[i]))
        dur = float(op.get_duration_for_combination(k, combos, machines))
        start = float(t[i])
        finish.append(start + dur)
        job = op.job_id or 0
        color = cmap(job % 20)
        for m in combos[k]:
            yi = y_of.get(m)
            if yi is None:
                continue
            ax.barh(yi, dur, left=start, color=color, edgecolor="black",
                    linewidth=0.4, alpha=0.85)
            # Tiny label for op-index when bars are wide enough.
            if dur > 0:
                ax.text(start + dur / 2, yi, str(i), fontsize=6, ha="center",
                        va="center", color="white" if dur > 30 else "black")

    ax.set_yticks(range(len(machines)))
    ax.set_yticklabels(machines, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, max(finish) * 1.02 if finish else 1)
    ax.grid(True, alpha=0.2, axis="x")

    # Title shows scheduler + key metrics.
    m = result.metrics or {}
    title = f"{result.scheduler_name}"
    bits = []
    if "makespan_us" in m:
        bits.append(f"ms={m['makespan_us']:.0f}us")
    if m.get("deadline_miss_count") is not None:
        bits.append(f"misses={m['deadline_miss_count']}")
    if m.get("cross_device_transitions") is not None:
        bits.append(f"xfers={m['cross_device_transitions']}")
    if bits:
        title += f" | {'  '.join(bits)}"
    ax.set_title(title, fontsize=10)


def render_side_by_side(
    results: List[SchedulerResult],
    save_path: str,
    title: str = "Scheduler comparison",
    *,
    panel_height_in: float = 2.0,
    fig_width_in: float = 14.0,
) -> Optional[str]:
    """Composite of per-scheduler Gantts as a vertical stack of subplots.

    Each panel is rendered DIRECTLY (not by re-loading a saved PNG) so the
    bars and machine labels remain readable regardless of the per-scheduler
    Gantt's original resolution.
    """
    rendered = [r for r in results if r.feasible and r.workload is not None
                and r.t is not None and r.alpha is not None]
    if not rendered:
        return None

    rows = len(rendered)
    fig, axes = plt.subplots(rows, 1, figsize=(fig_width_in, panel_height_in * rows),
                             constrained_layout=True, sharex=False)
    if rows == 1:
        axes = [axes]
    for ax, r in zip(axes, rendered):
        _render_gantt_to_ax(ax, r)
    # Only the bottom axis shows the time label.
    axes[-1].set_xlabel("time (us)")
    fig.suptitle(title, fontsize=12)
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=130)
    plt.close(fig)
    return save_path


# ----- Markdown table ---------------------------------------------------------


_TABLE_COLUMNS: List[tuple] = [
    ("scheduler", "Scheduler"),
    ("num_operations", "Ops"),
    ("makespan_us", "Makespan (us)"),
    ("nonperiodic_makespan_us", "Non-periodic makespan (us)"),
    ("p95_op_duration_us", "p95 op (us)"),
    ("p99_op_duration_us", "p99 op (us)"),
    ("deadline_miss_count", "Deadline misses"),
    ("deadline_miss_ratio", "Miss ratio"),
    ("total_lateness_us", "Total lateness (us)"),
    ("max_lateness_us", "Max lateness (us)"),
    ("cross_device_transitions", "Cross-device transitions"),
    ("critical_path_us", "Critical path (us)"),
    ("peak_dram_bytes", "Peak DRAM (B)"),
    ("peak_scratchpad_bytes", "Peak scratchpad (B)"),
    ("solver_wall_time_s", "Solver time (s)"),
]


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        if abs(value) >= 1000:
            return f"{value:,.1f}"
        return f"{value:.3f}"
    if isinstance(value, dict):
        return ", ".join(f"{k}={_fmt(v)}" for k, v in value.items())
    return str(value)


def write_markdown_report(
    results: List[SchedulerResult],
    save_path: str,
    *,
    title: str = "XPU-RT scheduler comparison",
    side_by_side_png: Optional[str] = None,
) -> str:
    """Write a markdown summary table + Gantt links for a single benchmark cell."""
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    lines: List[str] = [f"# {title}", ""]

    if side_by_side_png:
        rel = os.path.relpath(side_by_side_png, start=os.path.dirname(save_path) or ".")
        lines.append(f"![side-by-side]({rel})")
        lines.append("")

    headers = [label for _, label in _TABLE_COLUMNS]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join("---" for _ in headers) + "|")
    for r in results:
        row: List[str] = []
        for key, _ in _TABLE_COLUMNS:
            if key == "scheduler":
                row.append(r.scheduler_name + ("" if r.feasible else " *(infeasible)*"))
            else:
                row.append(_fmt(r.metrics.get(key)))
        lines.append("| " + " | ".join(row) + " |")

    lines.append("")
    lines.append("## Per-scheduler Gantts")
    lines.append("")
    for r in results:
        if r.gantt_png_path and os.path.exists(r.gantt_png_path):
            rel = os.path.relpath(r.gantt_png_path, start=os.path.dirname(save_path) or ".")
            lines.append(f"### {r.scheduler_name}")
            lines.append(f"![{r.scheduler_name}]({rel})")
            if r.schedule_json_path:
                json_rel = os.path.relpath(r.schedule_json_path, start=os.path.dirname(save_path) or ".")
                lines.append(f"[schedule json]({json_rel})")
            if r.note:
                lines.append(f"\n_note: {r.note}_")
            lines.append("")
        else:
            lines.append(f"### {r.scheduler_name}")
            lines.append(f"_infeasible or no Gantt available_ ({r.note or 'n/a'})")
            lines.append("")

    with open(save_path, "w") as f:
        f.write("\n".join(lines))
    return save_path
