"""Subprocess side of the CP-SAT decision bound. Runs under the ortools venv.

`_cpsat_solve.main` is executed verbatim -- the model it builds is the model the
production solver builds -- with `CpModel.Minimize` wrapped so the objective
variable also carries `objective <= XPURT_ORACLE_CMAX_US`. Wrapping Minimize is
what makes this work without touching the solver: it is the one call that hands
the objective variable out, and it happens after every other constraint is
already posted.

The verdict is written to XPURT_ORACLE_VERDICT because the caller's own error
path swallows INFEASIBLE, which is precisely the answer being sought.
"""
import json
import os
import sys
import time

from ortools.sat.python import cp_model

_HERE = os.path.dirname(os.path.abspath(__file__))
_SOLVE = os.path.abspath(os.path.join(_HERE, "..", "..", "xpu-rt", "_cpsat_solve.py"))

LIMIT = int(os.environ["XPURT_ORACLE_CMAX_US"])
VERDICT = os.environ["XPURT_ORACLE_VERDICT"]

# ortools renamed the camel-case API to snake_case and reaches the old names
# through a compatibility shim, so patch whichever the installed build defines.
_MIN = "Minimize" if "Minimize" in vars(cp_model.CpModel) else "minimize"
_SOL = "Solve" if "Solve" in vars(cp_model.CpSolver) else "solve"
_orig_minimize = getattr(cp_model.CpModel, _MIN)
_orig_solve = getattr(cp_model.CpSolver, _SOL)
_state = {}


def _minimize(self, obj):
    self.add(obj <= LIMIT)
    _state["bounded"] = True
    return _orig_minimize(self, obj)


def _solve(self, model, *a, **kw):
    t0 = time.time()
    status = _orig_solve(self, model, *a, **kw)
    _state["status"] = self.status_name(status)
    _state["wall"] = time.time() - t0
    return status


setattr(cp_model.CpModel, _MIN, _minimize)
setattr(cp_model.CpSolver, _SOL, _solve)

import importlib.util                                        # noqa: E402
spec = importlib.util.spec_from_file_location("_cpsat_solve", _SOLVE)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
try:
    mod.main(sys.argv[1], sys.argv[2])
finally:
    json.dump({"status": _state.get("status", "NO_SOLVE"),
               "bounded": bool(_state.get("bounded")),
               "limit_us": LIMIT,
               "solver_wall": _state.get("wall")},
              open(VERDICT, "w"))
