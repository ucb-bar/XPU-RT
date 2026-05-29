"""xpurt — installable facade for the XPU-RT scheduler.

Provides clean top-level imports for external consumers:

    from xpurt import schedule, Workload, compute_metrics

Internally still uses the existing flat module layout — no file moves.
Internal modules (scheduler.py, workload.py, ...) live as top-level
modules thanks to ``py-modules`` in pyproject.toml. This package only
re-exports their stable API surface.
"""

from scheduler import (
    schedule,
    schedule_window,
    schedule_additional_objectives,
    schedule_with_greedy_packing,
    schedule_with_convex_packing,
)
from workload import Workload, Operation, Job, Window
from workload_factory import (
    create_workload_from_dependencies,
    create_sequential_job,
    generate_syn_workload,
)
from metrics import compute_metrics
from profiling import SchedulerReport
from plot_gantt import render_gantt

__all__ = [
    "SchedulerReport",
    "schedule",
    "schedule_window",
    "schedule_additional_objectives",
    "schedule_with_greedy_packing",
    "schedule_with_convex_packing",
    "Workload",
    "Operation",
    "Job",
    "Window",
    "create_workload_from_dependencies",
    "create_sequential_job",
    "generate_syn_workload",
    "compute_metrics",
]

__version__ = "0.1.0"
