"""Oracle gap study, lower-bound half: bounds that are PROVEN, not observed.

A "best known" objective says nothing about how much is left on the table. To
turn a comparison between solvers into a statement about optimality we need a
number no schedule can beat. Two are computed here, both from the instance
alone, so neither depends on any solver being run.

WHAT IS BEING BOUNDED.  `schedule_decoder.evaluate` minimises the makespan over
the NON-PERIODIC operations when any exist, and over all operations otherwise;
call that set the *targets*. Separately, a schedule is only a solution at all
when `misses == 0`, i.e. every periodic operation ends inside its own window.
So the quantity bounded below is

    obj = max over target operations of their completion time,
    over schedules that also respect every periodic window.

Both bounds below drop the periodic-window constraint wherever it is awkward,
which only enlarges the feasible set, so they stay valid for the constrained
problem. Nothing here is a bound on `all_ops` (the makespan over every
operation) -- that is a different and larger quantity.

BOUND 1, CRITICAL PATH.  Give every operation its fastest feasible machine
combination and give every transfer zero cost. Then

    head[i] = max(min_start[i], max over predecessors p of head[p] + mindur[p])

is a valid lower bound on i's start in any schedule, because precedence and the
release time hold whatever the assignment, and no operation can run faster than
its fastest combination. `max over targets of head[i] + mindur[i]` is therefore
a lower bound on obj. This bound ignores every resource conflict.

BOUND 2, ENERGETIC / AREA LP.  This is the one that matters here, because in
these workloads EVERY operation can run on EVERY combination -- so the textbook
"work that can only go on machine m" is the empty set and that bound is
identically zero. The generalisation: an operation running on combination c
occupies *every* machine in c for its whole duration (that is exactly what the
per-machine `AddNoOverlap` in the CP-SAT model says), so machine m can absorb at
most (b - a) machine-time inside any window [a, b].

For a window [a, b], the operations that must lie entirely inside it are those
with head[i] >= a and deadline[i] <= b, where

    deadline[i] = min( max_end[i] if i is periodic,
                       T - tail[i] if i has a target descendant or is a target )

and tail[i] is the longest min-duration path from i's completion to the
completion of some target. `T - tail[i]` is a deadline because a schedule with
objective T finishes every target by T, and everything on a path into a target
must clear out ahead of it.

Given a candidate T that produces those deadlines, the LP

    min 0  s.t.  sum_c x[i,c] = 1,  x >= 0,
                 for every machine m and window [a, b]:
                     sum over i inside [a,b] of sum over c containing m of
                         x[i,c] * dur[i,c]   <=   b - a

is a relaxation of "a schedule with objective T exists" -- every real schedule
induces an integral x satisfying all of it. So if the LP is INFEASIBLE, no
schedule achieves T. Bisecting on T gives the largest T that can be refuted;
that value plus one grid step is the reported lower bound.

The windows are drawn from the observed head and deadline values (capped to a
grid, then refined by a cutting-plane pass that hunts the most violated window
against the current LP solution), plus [0, T] and every [0, max_end] level.

A useful side effect: the window constraints with b < T do not involve T at all.
If the LP is infeasible for those alone, the instance admits NO schedule that
meets all its periodic windows -- a proof of infeasibility rather than a report
that our solvers failed to find one.
"""
from __future__ import annotations

import os
import sys

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix, vstack

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

_INF = float("inf")


# --------------------------------------------------------------------------
# structural quantities
# --------------------------------------------------------------------------
def min_dur(ctx) -> np.ndarray:
    """Fastest feasible duration per operation.

    Not `ctx.min_dur`: that one takes the minimum over *strictly positive*
    durations, so an operation with both a zero-cost and a positive combination
    would report the positive one. Every head and tail below is a longest path
    over these values, and an overstated duration would overstate the bound --
    the one direction a lower bound may not err in. These instances happen to
    have no such operation (the zero-duration ones are zero on every
    combination), but the bound should not depend on that.
    """
    d = np.where(np.isfinite(ctx.dur), ctx.dur, np.inf)
    m = d.min(axis=1)
    return np.where(np.isfinite(m), m, 0.0)


def target_mask(ctx) -> np.ndarray:
    """Operations the objective maximises over, mirroring `evaluate`."""
    if np.any(~ctx.periodic):
        return ~ctx.periodic
    return np.ones(ctx.n, dtype=bool)


def heads(ctx) -> np.ndarray:
    """Lower bound on each operation's start: longest min-duration path in,
    with `min_start` releases folded in along the way."""
    md = min_dur(ctx)
    h = np.array(ctx.min_start, dtype=float)
    for i in ctx.topo:
        for p in ctx.pred[i]:
            v = h[p] + md[p]
            if v > h[i]:
                h[i] = v
    return h


def tails(ctx, targets) -> np.ndarray:
    """Longest min-duration path from an operation's completion to the
    completion of some target. -inf when no target is reachable."""
    md = min_dur(ctx)
    tl = np.where(targets, 0.0, -np.inf)
    for i in reversed(ctx.topo):
        for s in ctx.succ[i]:
            v = md[s] + tl[s]
            if v > tl[i]:
                tl[i] = v
    return tl


def cp_bound(ctx) -> float:
    """Critical-path lower bound on the objective."""
    tg = target_mask(ctx)
    h = heads(ctx)
    return float(np.max((h + min_dur(ctx))[tg])) if tg.any() else 0.0


# --------------------------------------------------------------------------
# area / energetic LP
# --------------------------------------------------------------------------
def _lp_pieces(ctx):
    """Column layout (i, c) over feasible pairs, plus the assignment rows."""
    cols, col_of = [], {}
    for i in range(ctx.n):
        for c in range(ctx.n_combos):
            if np.isfinite(ctx.dur[i, c]):
                col_of[(i, c)] = len(cols)
                cols.append((i, c))
    ncol = len(cols)
    rows, ci, dat = [], [], []
    for i in range(ctx.n):
        for c in range(ctx.n_combos):
            j = col_of.get((i, c))
            if j is not None:
                rows.append(i)
                ci.append(j)
                dat.append(1.0)
    A_eq = csr_matrix((dat, (rows, ci)), shape=(ctx.n, ncol))
    machine_of_col = [set(ctx.machines.index(m) for m in ctx.combos[c])
                      for (_i, c) in cols]
    dur_of_col = np.array([ctx.dur[i, c] for (i, c) in cols])
    op_of_col = np.array([i for (i, _c) in cols])
    return cols, col_of, A_eq, machine_of_col, dur_of_col, op_of_col


def _window_row(ctx, inside, m, machine_of_col, dur_of_col, op_of_col):
    """One `machine m load inside this window` row, as a sparse coefficient
    vector over the (i, c) columns."""
    sel = inside[op_of_col] & np.array([m in s for s in machine_of_col])
    coef = np.where(sel, dur_of_col, 0.0)
    return coef


def area_infeasible(ctx, T: float, n_grid: int = 32, cuts: int = 12,
                    verbose: bool = False):
    """True when no schedule can have objective <= T, by the area LP.

    Returns (infeasible, detail). `detail` names the binding window when the
    refutation is a single machine/window pair, which is what makes the number
    checkable by hand.
    """
    tg = target_mask(ctx)
    h = heads(ctx)
    tl = tails(ctx, tg)
    n_machines = len(ctx.machines)

    dl = np.full(ctx.n, _INF)
    with np.errstate(invalid="ignore"):
        by_tail = np.where(np.isfinite(tl), T - tl, _INF)
    dl = np.minimum(dl, by_tail)
    dl = np.minimum(dl, np.where(ctx.periodic, ctx.max_end, _INF))

    md = min_dur(ctx)
    if np.any(dl < h + md - 1e-9):
        bad = int(np.argmax((h + md) - dl))
        return True, (f"op {bad}: head {h[bad]:.3f} + mindur {md[bad]:.3f} "
                      f"> deadline {dl[bad]:.3f}")

    cols, col_of, A_eq, machine_of_col, dur_of_col, op_of_col = _lp_pieces(ctx)
    ncol = A_eq.shape[1]

    def grid(vals, k):
        v = np.unique(vals[np.isfinite(vals)])
        if len(v) <= k:
            return list(v)
        idx = np.unique(np.linspace(0, len(v) - 1, k).round().astype(int))
        return list(v[idx])

    starts = sorted(set([0.0] + grid(h, n_grid)))
    ends = sorted(set([T] + grid(dl, n_grid)))

    rows, rhs, tags = [], [], []

    def add_window(a, b):
        if b - a <= 1e-9:
            return
        inside = (h >= a - 1e-9) & (dl <= b + 1e-9)
        if not inside.any():
            return
        for m in range(n_machines):
            coef = _window_row(ctx, inside, m, machine_of_col, dur_of_col, op_of_col)
            if coef.sum() <= b - a + 1e-9:
                continue                      # cannot bind; skip the row
            rows.append(csr_matrix(coef))
            rhs.append(b - a)
            tags.append((a, b, m, int(inside.sum())))

    for a in starts:
        for b in ends:
            add_window(a, b)

    if not rows:
        return False, "no binding window"

    A_ub = vstack(rows, format="csr")
    res = linprog(np.zeros(ncol), A_ub=A_ub, b_ub=np.array(rhs),
                  A_eq=A_eq, b_eq=np.ones(ctx.n), bounds=(0, None),
                  method="highs")
    for _ in range(cuts):
        if not res.success:
            break
        # Cutting plane: with the current fractional assignment, find the
        # window whose machine load most exceeds its own length. A grid can
        # miss the binding window; this finds it exactly.
        x = res.x
        best = None
        load = dur_of_col * x
        for m in range(n_machines):
            sel = np.array([m in s for s in machine_of_col])
            per_op = np.zeros(ctx.n)
            np.add.at(per_op, op_of_col[sel], load[sel])
            order = np.argsort(h)
            for ai in range(0, len(order), max(1, len(order) // 64)):
                a = h[order[ai]]
                cand = order[ai:]
                cand = cand[np.isfinite(dl[cand])]
                if len(cand) == 0:
                    continue
                o2 = cand[np.argsort(dl[cand])]
                cum = np.cumsum(per_op[o2])
                viol = cum - (dl[o2] - a)
                k = int(np.argmax(viol))
                if viol[k] > 1e-7 and (best is None or viol[k] > best[0]):
                    best = (viol[k], a, dl[o2][k], m)
        if best is None:
            break
        _, a, b, _m = best
        before = len(rows)
        add_window(a, b)
        if len(rows) == before:
            break
        A_ub = vstack(rows, format="csr")
        res = linprog(np.zeros(ncol), A_ub=A_ub, b_ub=np.array(rhs),
                      A_eq=A_eq, b_eq=np.ones(ctx.n), bounds=(0, None),
                      method="highs")

    if res.success:
        return False, f"LP feasible with {len(rows)} window rows"
    if res.status != 2:                       # 2 == proven primal infeasible
        # Numerical trouble or an iteration limit is not a proof. Refusing to
        # call it one is the whole point of a lower bound.
        return False, (f"LP did not prove infeasibility (highs status "
                       f"{res.status}: {res.message}); T is not refuted")
    return _certificate(ctx, A_ub, np.array(rhs), tags, A_eq, ncol,
                        h, dl, starts, ends, len(rows))


def _certificate(ctx, A_ub, b_ub, tags, A_eq, ncol, h, dl, starts, ends, nrow):
    """Say WHICH windows are jointly over capacity, not merely that the LP failed.

    Re-solves elastically -- every window row may be exceeded by s >= 0, and the
    total excess is minimised. This is a second, numerically different LP, and
    it has to agree: a strictly positive optimum re-proves infeasibility, while
    an optimum of zero says the first solve's INFEASIBLE was numerical noise and
    the refutation is withdrawn. Returns (refuted, text), so a withdrawn
    refutation cannot be mistaken for a bound.

    The rows carrying the excess are the windows worth quoting. An all-machine
    window that is over on its own is quoted too when one exists, because that
    form is checkable without an LP at all.
    """
    from scipy.sparse import hstack, identity
    k = A_ub.shape[0]
    Ae = hstack([A_ub, -identity(k, format="csr")], format="csr")
    Aeq2 = hstack([A_eq, csr_matrix((A_eq.shape[0], k))], format="csr")
    cost = np.concatenate([np.zeros(ncol), np.ones(k)])
    r = linprog(cost, A_ub=Ae, b_ub=b_ub, A_eq=Aeq2, b_eq=np.ones(ctx.n),
                bounds=(0, None), method="highs")
    if not r.success:
        # The elastic problem is always feasible (make every s large enough),
        # so a failure here is solver trouble, not a proof.
        return False, (f"elastic recheck failed (highs status {r.status}: "
                       f"{r.message}); refutation withdrawn")
    s = r.x[ncol:]
    if s.sum() <= 1e-7:
        return False, ("elastic recheck puts the unavoidable overrun at "
                       f"{s.sum():.3e} ms; the INFEASIBLE verdict was numerical, "
                       "refutation withdrawn")
    idx = np.argsort(-s)[:3]
    quoted = [f"[{tags[i][0]:.2f},{tags[i][1]:.2f}] on "
              f"{ctx.machines[tags[i][2]]} short by {s[i]:.3f} ms "
              f"({tags[i][3]} ops pinned inside)"
              for i in idx if s[i] > 1e-7]
    parts = [f"LP over {nrow} window rows is infeasible"]
    if quoted:
        parts.append("binding windows: " + "; ".join(quoted))
    parts.append(f"total unavoidable overrun {s.sum():.3f} ms")
    agg = _tightest_aggregate_window(ctx, h, dl, starts, ends)
    parts.append(agg if agg.startswith("tightest all-machine window")
                 and "OVER" in agg else "no single all-machine window is "
                 "over on its own; the refutation needs the per-machine LP")
    return True, "; ".join(parts)


def _min_machine_time(ctx) -> np.ndarray:
    """Least machine-time each operation can consume, over its combinations.

    A combination holds every machine in it for the whole duration, so a
    two-core combination that halves the duration costs the same machine-time
    as the one-core one; picking the cheapest is what makes the aggregate bound
    below valid whatever the solver assigns.
    """
    w = np.full(ctx.n, np.inf)
    for i in range(ctx.n):
        for c in range(ctx.n_combos):
            d = ctx.dur[i, c]
            if np.isfinite(d):
                w[i] = min(w[i], len(ctx.combos[c]) * float(d))
    return np.where(np.isfinite(w), w, 0.0)


def _tightest_aggregate_window(ctx, h, dl, starts, ends) -> str:
    """The single most overloaded window, stated over ALL machines at once.

    Unlike a per-machine row this stands on its own -- every combination draws
    from the same pool of machine-time -- so it can be checked by hand:
    `n_machines * (b - a)` of machine-time exists in [a, b] and the operations
    pinned inside it need more.
    """
    mm = _min_machine_time(ctx)
    nm = len(ctx.machines)
    worst = None
    for a in starts:
        for b in ends:
            if not np.isfinite(b) or b - a <= 1e-9:
                continue
            inside = (h >= a - 1e-9) & (dl <= b + 1e-9)
            need = float(mm[inside].sum())
            slack = nm * (b - a) - need
            if worst is None or slack < worst[0]:
                worst = (slack, a, b, need, int(inside.sum()))
    if worst is None:
        return "no window"
    s, a, b, need, k = worst
    verdict = "OVER by" if s < 0 else "fits with"
    return (f"tightest all-machine window [{a:.3f},{b:.3f}]: {k} ops need "
            f"{need:.3f} ms of machine-time, {nm} machines supply "
            f"{nm * (b - a):.3f} -- {verdict} {abs(s):.3f} ms")


def aggregate_window_bound(ctx, T: float, n_grid: int = 64) -> tuple[bool, str]:
    """Standalone all-machine area check at objective T, no LP involved."""
    tg = target_mask(ctx)
    h = heads(ctx)
    tl = tails(ctx, tg)
    dl = np.minimum(np.where(np.isfinite(tl), T - tl, _INF),
                    np.where(ctx.periodic, ctx.max_end, _INF))
    mm = _min_machine_time(ctx)
    nm = len(ctx.machines)

    def grid(vals, k):
        v = np.unique(vals[np.isfinite(vals)])
        if len(v) <= k:
            return list(v)
        idx = np.unique(np.linspace(0, len(v) - 1, k).round().astype(int))
        return list(v[idx])

    starts = sorted(set([0.0] + grid(h, n_grid)))
    ends = sorted(set(([T] if np.isfinite(T) else []) + grid(dl, n_grid)))
    for a in starts:
        for b in ends:
            if b - a <= 1e-9:
                continue
            inside = (h >= a - 1e-9) & (dl <= b + 1e-9)
            if float(mm[inside].sum()) > nm * (b - a) + 1e-9:
                return True, _tightest_aggregate_window(ctx, h, dl, starts, ends)
    return False, _tightest_aggregate_window(ctx, h, dl, starts, ends)


def area_bound(ctx, hi: float, tol: float = 0.01, n_grid: int = 32,
               verbose: bool = False):
    """Largest refutable T, by bisection: the area lower bound on the objective.

    Bisects between the critical-path bound (always refutable below it, by
    construction) and `hi`, an objective some valid schedule actually achieves.
    Returns (bound, why). The bound is the last T the LP could refute, so the
    true optimum lies in (bound, hi]; the reported value is a valid lower bound
    up to the `tol` bisection step.
    """
    lo = cp_bound(ctx)
    if hi <= lo + tol:
        return lo, "critical path already meets the known solution"
    ok, detail = area_infeasible(ctx, hi, n_grid=n_grid)
    if ok:
        return _INF, (f"INCONSISTENT: T={hi}, achieved by a valid schedule, "
                      f"was refuted -- {detail}")
    bad, bad_detail, good = lo, "critical path", hi
    while good - bad > tol:
        mid = 0.5 * (good + bad)
        r, d = area_infeasible(ctx, mid, n_grid=n_grid)
        if verbose:
            print(f"    T={mid:9.3f} {'REFUTED' if r else 'feasible'}  {d}", flush=True)
        if r:
            bad, bad_detail = mid, d
        else:
            good = mid
    return bad, bad_detail


def window_feasibility(ctx, n_grid: int = 32):
    """Is the instance schedulable at all? Uses only the windows that do not
    involve T, so the answer is about the periodic constraints alone."""
    return area_infeasible(ctx, T=_INF, n_grid=n_grid)
