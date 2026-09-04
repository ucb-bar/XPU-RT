"""CP-SAT backend: the same scheduling problem as a constraint model.

Why bother when there is already a MILP: this is a disjunctive no-overlap
problem, and the MILP encodes "these two ops cannot overlap" with a big-M
plus an ordering boolean *per pair* — the thing that makes the model
1.46M constraints on a 1751-op workload. CP-SAT has interval variables and a
native `AddNoOverlap` propagator, so the same statement costs one interval per
op and one constraint per machine rather than O(N^2) rows.

OR-Tools is not installed in the scheduler's own environment (it needs a newer
numpy and protobuf than that environment pins), so this module talks to it out
of process: dump the workload as JSON, run `_cpsat_solve.py` under whatever
interpreter has ortools, read the assignment back. `XPURT_CPSAT_PYTHON` names
that interpreter.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import tempfile

import numpy as np

from schedule_decoder import DecoderContext

_SOLVER_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "_cpsat_solve.py")
# CP-SAT is integral: durations are milliseconds as floats, so scale to
# integer microseconds. Anything finer buys nothing at these magnitudes and
# costs the solver domain size.
_SCALE = 1000

# Details of the most recent solve (status, objective, proven bound, wall).
# `cpsat_schedule` has to keep returning bare (t, alpha) — that is the contract
# every other solver in this repo uses — so the provenance goes here instead of
# into the return value. Without it a time-limited FEASIBLE answer at a 4x
# optimality gap is indistinguishable from a proven OPTIMAL one.
LAST_SOLVE: dict = {}

# Milliseconds -> integer microseconds, rounded outwards. The epsilon absorbs
# binary-representation noise so an exact 3.0 ms does not become 3001 us.
_EPS = 1e-9


def _ceil(v: float) -> int:
    return int(math.ceil(float(v) * _SCALE - _EPS))


def _floor(v: float) -> int:
    return int(math.floor(float(v) * _SCALE + _EPS))


def _integerize(ctx, t, alpha, dur_int) -> tuple[list[int], list[int]] | None:
    """Re-lay a float schedule on the integer microsecond grid CP-SAT uses.

    Rounding each start and each duration independently is not safe: two
    operations that abut exactly in float can end up overlapping by one
    microsecond once rounded, which makes the whole hint infeasible — and
    CP-SAT reports a complete-but-infeasible hint and falls back to its own
    search. This replays the schedule in the solver's own arithmetic, keeping
    the machine assignment and the relative order and pushing each operation to
    the first integer instant where its predecessors are done and its machine
    is free. Returns None if an operation cannot be placed inside its window.
    """
    n, n_combos = ctx.n, ctx.n_combos
    combos = [int(np.argmax(row)) for row in alpha]
    order = sorted(range(n), key=lambda i: (float(t[i]), i))
    machine_free: dict = {}
    starts = [0] * n
    ends = [0] * n
    for i in order:
        c = combos[i]
        d = dur_int[i][c]
        if d < 0:
            return None
        floor = _ceil(t[i])
        floor = max(floor, _ceil(ctx.min_start[i]))
        for p in ctx.pred[i]:
            floor = max(floor, ends[p])
        # A combination occupies EVERY machine in it, so availability has to be
        # tracked per machine. Keying it by the FIRST machine of each
        # conflicting combination cannot express "CPU_P#1 is busy": once
        # sibling-core combinations exist -- ['CPU_P#0'], ['CPU_P#0','CPU_P#1'],
        # ['CPU_P#1'] -- the two-core group reports P#0 while the singleton
        # reports P#1, so the group's occupancy of P#1 goes unrecorded. The
        # replay then packs operations that genuinely collide, drifts from the
        # schedule it is meant to reproduce, and overruns a periodic window
        # (4.76 ms on the 295-op control_mix gempair workload), at which point
        # the whole hint is discarded and the "warm" solve is a cold one.
        for mm in ctx.combos[c]:
            floor = max(floor, machine_free.get(mm, 0))
        starts[i], ends[i] = floor, floor + d
        if np.isfinite(ctx.max_end[i]) and ends[i] > _floor(ctx.max_end[i]):
            return None
        for mm in ctx.combos[c]:
            machine_free[mm] = max(machine_free.get(mm, 0), ends[i])
    return starts, ends


def build_payload(ctx, time_limit: float = 60.0,
                  restrict_to_nonperiodic: bool = True, workers: int = 8,
                  random_seed: int = 0, warm_start=None,
                  verbose: bool = False) -> dict:
    """The JSON model handed to `_cpsat_solve.py`.

    Split out from `cpsat_schedule` so the model can be inspected without an
    ortools interpreter: the two bugs fixed in 95db5778 were both in this
    payload's account of which machines a combination occupies, and neither
    was reachable by a test that could only look at an objective value.
    """
    dur = np.where(np.isfinite(ctx.dur), ctx.dur, -1.0)
    model = {
        "n": ctx.n,
        "n_combos": ctx.n_combos,
        "scale": _SCALE,
        # Round the model's numbers OUTWARDS, never to nearest.
        #
        # `end <= max_end` is enforced on the integer grid, but `evaluate`
        # scores the schedule in the original floats. Rounding a duration to
        # nearest understates it by up to half a microsecond, and minimising
        # makespan pushes periodic operations to end exactly on their deadline,
        # so the integer model was satisfied while the float schedule overran:
        # on control_mix_quad, op34 has duration 0.001251 ms, stored as 1 us,
        # and finished 0.000251 ms past its window -- a real, if tiny, missed
        # deadline that the solver reported as OPTIMAL and valid.
        #
        # Ceiling durations and min_start, and flooring max_end, makes the
        # integer model strictly conservative: satisfying it implies satisfying
        # the float constraint it stands for. Costs at most 1 us per operation.
        "dur": [[_ceil(d) if d >= 0 else -1 for d in row] for row in dur],
        "pred": ctx.pred,
        # Transfer cost is indexed by the *first machine* of each combination,
        # matching how the MILP and the greedy pickers charge it.
        "transfer": [[int(round(float(ctx.transfer[a][b]) * _SCALE))
                      for b in range(len(ctx.machines))]
                     for a in range(len(ctx.machines))],
        "first_machine": ctx.first_machine,
        # Every machine each combination occupies, not just its first: the
        # per-machine no-overlap constraints need the whole set, or a
        # multi-core combination's hold on its non-first cores is invisible.
        "combo_machines": [[ctx.machines.index(mname) for mname in combo]
                           for combo in ctx.combos],
        "conflict": [[bool(x) for x in row] for row in ctx.conflict],
        "min_start": [_ceil(v) for v in ctx.min_start],
        "max_end": [_floor(v) if np.isfinite(v) else -1 for v in ctx.max_end],
        "periodic": [bool(v) for v in ctx.periodic],
        "restrict_to_nonperiodic": bool(restrict_to_nonperiodic),
        "time_limit": float(time_limit),
        "workers": int(workers),
        "random_seed": int(random_seed),
    }
    if warm_start is not None:
        ws_t, ws_alpha = warm_start
        # A hint has to be *complete and self-consistent* to be usable: CP-SAT
        # completes a partial assignment itself, and if that completion is
        # infeasible it drops the hint silently. Hinting only start + the
        # combination booleans left duration and end unhinted and broke
        # end == start + duration, so the hint was discarded and the "warm"
        # arm was just a cold solve. Send all four, from the same schedule.
        combos = [int(np.argmax(row)) for row in ws_alpha]
        dur_int = model["dur"]
        placed = _integerize(ctx, ws_t, ws_alpha, dur_int)
        if placed is None:
            if verbose:
                print("  cpsat: warm start does not fit the integer model; "
                      "solving cold")
        else:
            starts, ends = placed
            model["hint_start"] = starts
            model["hint_combo"] = combos
            model["hint_dur"] = [dur_int[i][combos[i]] for i in range(ctx.n)]
            model["hint_end"] = ends
    return model


def cpsat_available() -> str | None:
    """Path to an interpreter that can import ortools, or None."""
    for cand in (os.environ.get("XPURT_CPSAT_PYTHON"), "python3"):
        if not cand:
            continue
        try:
            r = subprocess.run([cand, "-c", "import ortools"],
                               capture_output=True, timeout=60)
            if r.returncode == 0:
                return cand
        except Exception:
            continue
    return None


def cpsat_schedule(workload, time_limit: float = 60.0,
                   restrict_to_nonperiodic: bool = True,
                   workers: int = 8, verbose: bool = False,
                   warm_start=None, random_seed: int = 0,
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Solve with CP-SAT. Raises RuntimeError if no ortools interpreter exists.

    Determinism: CP-SAT is only reproducible with `workers=1`. With several
    search workers the result depends on thread interleaving, and repeated
    runs of an identical configuration differ — measured spread on a 242-op
    instance is about +/-1.5 ms on a 46 ms schedule. `random_seed` is passed
    through either way, but it does not make a multi-worker solve
    deterministic.

    `warm_start` is an existing (t, alpha) — typically a greedy schedule — fed
    to CP-SAT as a solution hint. CP-SAT does not have to respect a hint, but
    a feasible one gives it an incumbent immediately instead of spending the
    first part of its budget finding any solution at all, and it bounds the
    search from the start.
    """
    python = cpsat_available()
    if python is None:
        raise RuntimeError(
            "no interpreter with ortools found; set XPURT_CPSAT_PYTHON to one "
            "(e.g. a venv created with `python -m venv && pip install ortools`)")

    ctx = DecoderContext(workload)
    model = build_payload(ctx, time_limit=time_limit,
                          restrict_to_nonperiodic=restrict_to_nonperiodic,
                          workers=workers, random_seed=random_seed,
                          warm_start=warm_start, verbose=verbose)

    with tempfile.TemporaryDirectory() as td:
        inp, outp = os.path.join(td, "model.json"), os.path.join(td, "sol.json")
        with open(inp, "w") as fh:
            json.dump(model, fh)
        proc = subprocess.run([python, _SOLVER_SCRIPT, inp, outp],
                              capture_output=True, text=True,
                              timeout=time_limit + 300)
        if verbose and proc.stdout:
            print(proc.stdout.rstrip())
        if proc.returncode != 0 or not os.path.exists(outp):
            raise RuntimeError(
                f"cpsat solver failed (rc={proc.returncode}): "
                f"{(proc.stderr or '')[-500:]}")
        sol = json.load(open(outp))

    if sol.get("status") in ("INFEASIBLE", "UNKNOWN", "MODEL_INVALID"):
        raise RuntimeError(f"cpsat returned {sol['status']} with no solution")

    t = np.array(sol["start"], dtype=float) / _SCALE
    alpha = np.zeros((ctx.n, ctx.n_combos))
    for i, c in enumerate(sol["combo"]):
        alpha[i, int(c)] = 1.0
    LAST_SOLVE.clear()
    LAST_SOLVE.update(
        status=sol["status"],
        objective=sol["objective"] / _SCALE,
        best_bound=sol["best_bound"] / _SCALE,
        gap=((sol["objective"] - sol["best_bound"]) / sol["objective"]
             if sol["objective"] else None),
        wall=sol["wall"],
        time_limit=float(time_limit),
    )
    if verbose:
        print(f"  cpsat: status={sol['status']} objective={sol['objective'] / _SCALE:.2f} ms "
              f"bound={sol['best_bound'] / _SCALE:.2f} ms wall={sol['wall']:.1f}s")
    return t, alpha
