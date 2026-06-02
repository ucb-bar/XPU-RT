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
    from scheduler import schedule as milp_schedule
    return milp_schedule(workload, **kwargs)


def _milp_with(solver: str):
    """Build a registry entry that calls the existing MILP formulation with a
    different CVXPY-supported solver backend. Same MILP, different vendor —
    useful for solver-comparison studies."""
    def _fn(workload, **kwargs):
        from scheduler import schedule as milp_schedule
        kwargs["cvxpy_solver"] = solver
        return milp_schedule(workload, **kwargs)
    _fn.__name__ = f"_milp_{solver.lower()}"
    return _fn


def _heft(workload, **kwargs):
    from scheduler_heft import heft
    return heft(workload, **kwargs)


def _critical_path(workload, **kwargs):
    from scheduler_heft import critical_path
    return critical_path(workload, **kwargs)


def _edf(workload, **kwargs):
    from scheduler_heft import edf
    return edf(workload, **kwargs)


def _fastest_device(workload, **kwargs):
    from scheduler_heft import fastest_device
    return fastest_device(workload, **kwargs)


def _fifo(workload, **kwargs):
    from scheduler_heft import fifo
    return fifo(workload, **kwargs)


def _random_list(workload, **kwargs):
    from scheduler_heft import random_list
    return random_list(workload, **kwargs)


def _cpsat(workload, **kwargs):
    from scheduler_cpsat import cpsat_with_heft_warm_start
    return cpsat_with_heft_warm_start(workload, **kwargs)


def _cpsat_memory(workload, **kwargs):
    from scheduler_cpsat import cpsat_memory_aware
    return cpsat_memory_aware(workload, **kwargs)


def _round_robin(workload, **kwargs):
    from scheduler_heft import round_robin
    return round_robin(workload, **kwargs)


def _peft(workload, **kwargs):
    from scheduler_heft import peft
    return peft(workload, **kwargs)


def _min_min(workload, **kwargs):
    from scheduler_heft import min_min
    return min_min(workload, **kwargs)


def _max_min(workload, **kwargs):
    from scheduler_heft import max_min
    return max_min(workload, **kwargs)


def _sa(workload, **kwargs):
    from scheduler_heft import simulated_annealing
    return simulated_annealing(workload, **kwargs)


def _cost_model(workload, **kwargs):
    from scheduler_ml import cost_model_scheduler
    return cost_model_scheduler(workload, **kwargs)


def _gnn_placement(workload, **kwargs):
    from scheduler_gnn import gnn_placement_scheduler
    return gnn_placement_scheduler(workload, **kwargs)


def _rl_policy(workload, **kwargs):
    from scheduler_rl import rl_policy_scheduler
    return rl_policy_scheduler(workload, **kwargs)


def _llm_ranker(workload, **kwargs):
    """LLM ranker is a meta-scheduler — it ranks candidates produced by
    the rewrite loop, not a scheduler per se. As a scheduler entry it
    just runs HEFT (so it can be slotted into existing eval pipelines)."""
    from scheduler_heft import heft
    return heft(workload, **kwargs)


_REGISTRY: Dict[str, SchedulerFn] = {
    "mosek": _mosek,
    "heft": _heft,
    "critical_path": _critical_path,
    "edf": _edf,
    "fastest_device": _fastest_device,
    "fifo": _fifo,
    "random_list": _random_list,
    "cpsat": _cpsat,
    "cpsat_memory": _cpsat_memory,
    "round_robin": _round_robin,
    "peft": _peft,
    "min_min": _min_min,
    "max_min": _max_min,
    "simulated_annealing": _sa,
    "cost_model": _cost_model,
    "gnn_placement": _gnn_placement,
    "rl_policy": _rl_policy,
    "llm_ranker": _llm_ranker,
    # Same MILP formulation as ``mosek``, different CVXPY backend. Used to
    # disambiguate "is this win driven by the formulation or by MOSEK?".
    # If the backend isn't installed CVXPY raises a clear error at solve time.
    "milp_gurobi": _milp_with("GUROBI"),
    "milp_highs":  _milp_with("HIGHS"),
    "milp_scip":   _milp_with("SCIP"),
    "milp_cbc":    _milp_with("CBC"),
}


def register(name: str, fn: SchedulerFn) -> None:
    """Register a scheduler by name. Later milestones populate this."""
    _REGISTRY[name] = fn


def _wrap_with_compaction(fn: SchedulerFn) -> SchedulerFn:
    """Wrap a scheduler so its (t, alpha) return value passes through the
    left-shift compaction post-pass. Idempotent on tight solvers
    (MOSEK/CPSAT) — slack-eliminating on list schedulers (HEFT/PEFT/EDF/...).

    Disabled by setting env XPURT_NO_COMPACT=1 (useful when comparing a
    solver's raw vs compacted makespan)."""
    def _compacted(workload, **kwargs):
        result = fn(workload, **kwargs)
        if os.environ.get("XPURT_NO_COMPACT", "0") in ("1", "true", "True"):
            return result
        if result is None:
            return result
        # Schedulers should return (t, alpha, fused_workload, fusion_map),
        # but the registry also accepts opaque return values for testing —
        # pass anything that doesn't fit the contract through unchanged.
        if not isinstance(result, tuple) or len(result) < 2:
            return result
        t, alpha = result[0], result[1]
        fused = result[2] if len(result) >= 3 else None
        fmap = result[3] if len(result) >= 4 else None
        if t is None or alpha is None:
            return result
        try:
            from compaction import left_shift_compact
            wl_for_compact = fused if fused is not None else workload
            t2, alpha2 = left_shift_compact(t, alpha, wl_for_compact)
            return (t2, alpha2, fused, fmap) if len(result) >= 4 else result
        except Exception as exc:
            # Compaction must never break a working schedule. If anything
            # goes wrong we fall back to the un-compacted output and warn.
            print(f"warning: compaction post-pass skipped ({exc})")
            return result
    _compacted.__name__ = f"{fn.__name__}_compacted"
    return _compacted


def get_scheduler(name: str) -> SchedulerFn:
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY.keys()))
        raise ValueError(f"Unknown scheduler '{name}'. Available: {available}")
    return _wrap_with_compaction(_REGISTRY[name])


def available_schedulers() -> List[str]:
    return sorted(_REGISTRY.keys())
