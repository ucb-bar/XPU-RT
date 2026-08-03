"""Iterative scheduling policies (Phase C).

Each policy wraps an underlying solver from the schedulers registry but
with a documented mapping to the structural intent the policy embodies.
That structure shows up in two places:

  1. The choice of underlying solver (e.g. periodic_anchor uses
     `decomposed`, which is a period-first list scheduler).
  2. The pre-pass applied to the workload before solving (currently a
     no-op for C4; for C1-C3 it's documented inline).

This module is intentionally thin: policies do NOT reinvent scheduling.
They compose existing solvers with analytical formulas from
`decision_formulas.py` to produce a fixture + a policy log.

Each policy returns:
    {
      "fixture_path":  path to the scheduled JSON fixture,
      "makespan_us":   float,
      "n_deadline_miss": int,
      "n_shards_applied": int,
      "n_fuses_applied": int,
      "solve_wall_s":  float,
      "policy_log":    list of {action, op, reason, delta} dicts
    }
"""

from __future__ import annotations

from .yolo_anchor import yolo_anchor
from .periodic_anchor import periodic_anchor
from .critical_path_first import critical_path_first
from .cpsat_unconstrained import cpsat_unconstrained
from .mosek_decomposed import mosek_decomposed
from .hybrid_periodic_mosek_yolo import hybrid_periodic_mosek_yolo

POLICIES = {
    "yolo_anchor": yolo_anchor,
    "periodic_anchor": periodic_anchor,
    "critical_path_first": critical_path_first,
    "cpsat_unconstrained": cpsat_unconstrained,
    "mosek_decomposed": mosek_decomposed,                # Phase F2g
    "hybrid_periodic_mosek_yolo": hybrid_periodic_mosek_yolo,  # hybrid
}

__all__ = [
    "POLICIES",
    "yolo_anchor",
    "periodic_anchor",
    "critical_path_first",
    "cpsat_unconstrained",
    "mosek_decomposed",
    "hybrid_periodic_mosek_yolo",
]
