"""
Scheduler registry for XPU-RT.

All registered schedulers share the signature:

    def <scheduler>(workload, **kwargs) -> (t, alpha, fused_workload, fusion_map)

where (t, alpha) feed straight into ``postprocessing.output_scheduled_json`` and
``schedule_validation.validate_schedule``, so any scheduler in the registry can
be substituted into the existing entry-point scripts without further plumbing.

This file is intentionally a flat module (no subpackage) per repo convention.
"""

from __future__ import annotations

import os
import sys
from typing import Callable, Dict, List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


SchedulerFn = Callable[..., tuple]


def _mosek(workload, **kwargs):
    """Forward to the existing CVXPY/MOSEK MILP scheduler verbatim."""
    from scheduler import schedule as mosek_schedule
    return mosek_schedule(workload, **kwargs)


_REGISTRY: Dict[str, SchedulerFn] = {
    "mosek": _mosek,
}


def register(name: str, fn: SchedulerFn) -> None:
    """Register a scheduler by name. Later milestones populate this."""
    _REGISTRY[name] = fn


def get_scheduler(name: str) -> SchedulerFn:
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY.keys()))
        raise ValueError(f"Unknown scheduler '{name}'. Available: {available}")
    return _REGISTRY[name]


def available_schedulers() -> List[str]:
    return sorted(_REGISTRY.keys())
