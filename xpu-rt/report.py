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


def render_side_by_side(
    results: List[SchedulerResult],
    save_path: str,
    title: str = "Scheduler comparison",
) -> Optional[str]:
    """Composite of per-scheduler Gantts as a vertical stack of subplots.

    Reuses the already-rendered individual PNGs to avoid double work.
    """
    rendered = [r for r in results if r.gantt_png_path and os.path.exists(r.gantt_png_path)]
    if not rendered:
        return None

    rows = len(rendered)
    fig, axes = plt.subplots(rows, 1, figsize=(14, 3.5 * rows), constrained_layout=True)
    if rows == 1:
        axes = [axes]
    for ax, r in zip(axes, rendered):
        ax.imshow(mpimg.imread(r.gantt_png_path))
        ax.set_title(r.scheduler_name)
        ax.axis("off")
    fig.suptitle(title)
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=120)
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
