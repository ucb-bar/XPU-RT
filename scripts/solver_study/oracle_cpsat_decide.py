"""CP-SAT as a DECISION procedure: "is there a schedule with objective <= T?"

CP-SAT's own `best_bound` after a long optimisation is a valid lower bound, but
on these instances it is a terrible one -- 4.86 ms against a 56.8 ms incumbent
on control_mix_gempair after 1800 s, because the search spends its budget
improving the incumbent and its relaxation never tightens. Asked the strictly
easier question "does anything reach T?", the same solver can often return
INFEASIBLE, and an INFEASIBLE answer is a *proof* that the optimum exceeds T.
Bisecting on T turns CP-SAT into a lower-bound engine.

MODEL PROVENANCE.  The model must be the repo's model, not a re-implementation,
or the bound is about a different problem. So nothing is rebuilt here:
`cpsat_scheduler.cpsat_schedule` constructs its payload exactly as it always
does, and only the path it hands to the subprocess is redirected (module
attribute, at runtime -- no solver file is edited). The subprocess then runs
`_cpsat_solve.main` verbatim, with one interception: `CpModel.Minimize` is
wrapped so that the objective variable it is handed also gets
`objective <= T`. The model CP-SAT sees is therefore the production model plus
that single row.

VALIDITY.  An INFEASIBLE verdict proves no schedule satisfies (precedence,
per-machine no-overlap, releases, every periodic window, objective <= T) *on
the integer microsecond grid the model uses*. Durations are rounded to the
nearest microsecond, so the statement transfers to the real-valued problem up
to that rounding -- at most 0.5 us per operation on the binding chain, i.e.
well under 0.1 ms at these depths, and negligible against the gaps reported.
The start-time domain is capped at the payload's `horizon`; the driver checks
T <= horizon so no schedule achieving T is excluded by that cap.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

_HELPER = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "_oracle_cpsat_decide_child.py")


def decide(workload, T: float, time_limit: float = 300.0, workers: int = 4,
           random_seed: int = 0) -> dict:
    """Ask CP-SAT whether objective <= T is reachable.

    Returns the child's verdict dict: status is INFEASIBLE (T is refuted --
    a proven lower bound), OPTIMAL/FEASIBLE (T is reachable), or UNKNOWN
    (budget exhausted, nothing proven either way).
    """
    import cpsat_scheduler as cps
    from schedule_decoder import DecoderContext

    ctx = DecoderContext(workload)
    with tempfile.TemporaryDirectory() as td:
        verdict = os.path.join(td, "verdict.json")
        old_script, old_env = cps._SOLVER_SCRIPT, dict(os.environ)
        cps._SOLVER_SCRIPT = _HELPER
        os.environ["XPURT_ORACLE_CMAX_US"] = str(int(np.floor(T * cps._SCALE)))
        os.environ["XPURT_ORACLE_VERDICT"] = verdict
        t0 = time.perf_counter()
        why = None
        try:
            cps.cpsat_schedule(workload, time_limit=time_limit, workers=workers,
                               random_seed=random_seed, warm_start=None)
        except RuntimeError as e:
            why = str(e)              # INFEASIBLE/UNKNOWN raise; the verdict file has it
        finally:
            cps._SOLVER_SCRIPT = old_script
            os.environ.clear()
            os.environ.update(old_env)
        out = (json.load(open(verdict)) if os.path.exists(verdict)
               else {"status": "NO_VERDICT", "why": why})
    out["wall_s"] = round(time.perf_counter() - t0, 2)
    out["T"] = T
    return out


def horizon_of(workload) -> float:
    """The payload's own horizon, in ms -- the cap on every start variable.

    Recomputed exactly as `_cpsat_solve` does, so the driver can assert that a
    candidate T sits inside it and the cap therefore excludes nothing.
    """
    from schedule_decoder import DecoderContext
    ctx = DecoderContext(workload)
    d = np.where(np.isfinite(ctx.dur), ctx.dur, -1.0)
    rows = [int(round(x * 1000)) for row in d for x in row]
    per_op = [max((int(round(x * 1000)) for x in row if x >= 0), default=0)
              for row in d]
    hz = sum(per_op)
    me = [int(round(v * 1000)) for v in ctx.max_end if np.isfinite(v)]
    return (max(hz, max(me, default=0)) + 1) / 1000.0
