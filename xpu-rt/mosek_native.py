"""The scheduling MILP built directly against MOSEK's Optimizer API.

Why this exists: cvxpy cannot pass a MIP start. It compiles the problem through
a reduction chain, so the variables MOSEK receives are not `t`/`alpha`/`beta`
and there is no mapping to carry a user-set `.value` through. Supplying a
complete, feasible starting schedule via cvxpy is measurably a no-op — MOSEK
returns the identical cold answer to three decimals (see
docs/scheduler_solver_study.md 4.2b). Building the model here instead makes
`putxxslice` + MSK_IPAR_MIO_CONSTRUCT_SOL available, which is the supported way
to hand MOSEK an incumbent.

The formulation is the same one `scheduler.py` builds, kept deliberately
identical so the two are comparable:

  variables   t[i] >= 0                       start times, continuous
              C_max                           makespan, continuous
              alpha[i, c] in {0, 1}           op i runs on combination c
              beta[p] in {0, 1}               ordering bit for surviving pair p
  subject to  (2) sum_c alpha[i, c] == 1
              (3) precedence, with the predecessor's duration selected by alpha
              (W) t[i] >= min_start[i];  t[i] + dur <= max_end[i]
              (4)(5) big-M non-overlap for each surviving pair
              (6) C_max >= finish of every objective-set op
  minimise    C_max

Pair pruning matches `scheduler.py`: pairs whose windows cannot overlap, and
pairs already ordered by the precedence DAG, carry no ordering bit.
"""

from __future__ import annotations

import time

import numpy as np

from schedule_decoder import DecoderContext

try:
    import mosek
except ImportError:                                   # pragma: no cover
    mosek = None


def _surviving_pairs(ctx: DecoderContext) -> list[tuple[int, int]]:
    """Pairs that still need an ordering bit, using the same two tests as the
    cvxpy model: disjoint windows, and transitive precedence."""
    n = ctx.n
    # Transitive reachability by bitset, over the decoder's topological order.
    desc = [0] * n
    for u in reversed(ctx.topo):
        bits = 0
        for v in ctx.succ[u]:
            bits |= desc[v] | (1 << v)
        desc[u] = bits

    def windows_overlap(i, j):
        a0, a1 = ctx.min_start[i], ctx.max_end[i]
        b0, b1 = ctx.min_start[j], ctx.max_end[j]
        if not (np.isfinite(a1) and np.isfinite(b1)):
            return True                                # unbounded: assume yes
        return a0 < b1 and b0 < a1

    out = []
    for i in range(n):
        for j in range(i + 1, n):
            if not windows_overlap(i, j):
                continue
            if ((desc[i] >> j) & 1) or ((desc[j] >> i) & 1):
                continue
            out.append((i, j))
    return out


def _big_m(ctx: DecoderContext) -> float:
    """Same bound as scheduler._compute_big_m: everything serial on its slowest
    combination, doubled, floored at 5000."""
    slowest = np.where(np.isfinite(ctx.dur), ctx.dur, 0.0).max(axis=1)
    return max(5000.0, float(slowest.sum()) * 2.0)


def schedule_mosek_native(workload, time_limit: float = 120.0,
                          restrict_to_nonperiodic: bool = True,
                          warm_start=None, verbose: bool = False,
                          ) -> tuple[np.ndarray, np.ndarray]:
    if mosek is None:
        raise RuntimeError("mosek is not installed in this environment")
    ctx = DecoderContext(workload)
    n, C = ctx.n, ctx.n_combos
    pairs = _surviving_pairs(ctx)
    P = len(pairs)
    H = _big_m(ctx)

    # Variable layout: [ t(n) | C_max(1) | alpha(n*C) | beta(P) ]
    T0, CMAX, A0, B0 = 0, n, n + 1, n + 1 + n * C
    numvar = n + 1 + n * C + P
    ai = lambda i, c: A0 + i * C + c

    targets = [i for i in range(n)
               if not (restrict_to_nonperiodic and ctx.periodic[i])]
    if not targets:
        targets = list(range(n))

    rows: list[tuple[list[int], list[float], float, float]] = []   # (idx, val, lo, up)
    INF = 0.0                                          # MOSEK ignores it for ranged keys

    for i in range(n):                                 # (2) assignment
        rows.append(([ai(i, c) for c in range(C)], [1.0] * C, 1.0, 1.0))

    for i in range(n):                                 # (3) precedence
        for p in ctx.pred[i]:
            idx = [T0 + i, T0 + p] + [ai(p, c) for c in range(C)]
            val = [1.0, -1.0] + [-(ctx.dur[p, c] if np.isfinite(ctx.dur[p, c]) else H)
                                 for c in range(C)]
            rows.append((idx, val, 0.0, None))         # t_i - t_p - dur_p >= 0

    for i in range(n):                                 # (W) deadline
        if np.isfinite(ctx.max_end[i]):
            idx = [T0 + i] + [ai(i, c) for c in range(C)]
            val = [1.0] + [(ctx.dur[i, c] if np.isfinite(ctx.dur[i, c]) else H)
                           for c in range(C)]
            rows.append((idx, val, None, float(ctx.max_end[i])))

    for pidx, (i, j) in enumerate(pairs):              # (4)(5) non-overlap
        for k1 in range(C):
            if not np.isfinite(ctx.dur[i, k1]):
                continue
            for k2 in range(C):
                if not np.isfinite(ctx.dur[j, k2]) or not ctx.conflict[k1][k2]:
                    continue
                # t_i - t_j - H*a_ik1 - H*a_jk2 + H*b >= dur_j_k2 - 2H
                rows.append(([T0 + i, T0 + j, ai(i, k1), ai(j, k2), B0 + pidx],
                             [1.0, -1.0, -H, -H, H],
                             float(ctx.dur[j, k2]) - 2 * H, None))
                # t_j - t_i - H*a_ik1 - H*a_jk2 - H*b >= dur_i_k1 - 3H
                rows.append(([T0 + j, T0 + i, ai(i, k1), ai(j, k2), B0 + pidx],
                             [1.0, -1.0, -H, -H, -H],
                             float(ctx.dur[i, k1]) - 3 * H, None))

    for i in targets:                                  # (6) makespan
        idx = [CMAX, T0 + i] + [ai(i, c) for c in range(C)]
        val = [1.0, -1.0] + [-(ctx.dur[i, c] if np.isfinite(ctx.dur[i, c]) else H)
                             for c in range(C)]
        rows.append((idx, val, 0.0, None))

    with mosek.Env() as env, env.Task() as task:
        if verbose:
            task.set_Stream(mosek.streamtype.log, lambda msg: print(msg, end=""))
        task.appendvars(numvar)
        task.appendcons(len(rows))

        for i in range(n):                             # release times as bounds
            task.putvarbound(T0 + i, mosek.boundkey.lo, float(ctx.min_start[i]), 0.0)
        task.putvarbound(CMAX, mosek.boundkey.lo, 0.0, 0.0)
        for v in range(A0, numvar):
            task.putvarbound(v, mosek.boundkey.ra, 0.0, 1.0)
            task.putvartype(v, mosek.variabletype.type_int)

        for r, (idx, val, lo, up) in enumerate(rows):
            task.putarow(r, idx, val)
            if lo is not None and up is not None:
                key = mosek.boundkey.fx if lo == up else mosek.boundkey.ra
                task.putconbound(r, key, lo, up)
            elif lo is not None:
                task.putconbound(r, mosek.boundkey.lo, lo, 0.0)
            else:
                task.putconbound(r, mosek.boundkey.up, 0.0, up)

        task.putcj(CMAX, 1.0)
        task.putobjsense(mosek.objsense.minimize)
        task.putdouparam(mosek.dparam.optimizer_max_time, float(time_limit))

        if warm_start is not None:
            ws_t, ws_alpha = warm_start
            chosen = [int(np.argmax(r)) for r in np.asarray(ws_alpha)]
            ws_t = np.asarray(ws_t, dtype=float)
            finish = np.array([ws_t[i] + ctx.dur[i, chosen[i]] for i in range(n)])
            xx = np.zeros(numvar)
            xx[T0:T0 + n] = ws_t
            xx[CMAX] = finish[targets].max()
            for i, c in enumerate(chosen):
                xx[ai(i, c)] = 1.0
            for pidx, (i, j) in enumerate(pairs):
                xx[B0 + pidx] = 1.0 if finish[i] <= ws_t[j] + 1e-9 else 0.0
            # This is the call cvxpy has no route to: hand MOSEK the integer
            # solution and tell it to build an incumbent from it.
            task.putxxslice(mosek.soltype.itg, 0, numvar, xx)
            task.putintparam(mosek.iparam.mio_construct_sol, mosek.onoffkey.on)
            if verbose:
                print(f"native warm start: {n} starts, {n * C} alpha, {P} beta, "
                      f"C_max={xx[CMAX]:.2f}")

        t0 = time.perf_counter()
        task.optimize()
        wall = time.perf_counter() - t0
        solsta = task.getsolsta(mosek.soltype.itg)
        if solsta in (mosek.solsta.unknown,):
            raise RuntimeError(f"mosek returned {solsta} with no integer solution")
        xx = task.getxx(mosek.soltype.itg)

    t = np.array(xx[T0:T0 + n], dtype=float)
    alpha = np.zeros((n, C))
    for i in range(n):
        c = int(np.argmax([xx[ai(i, c)] for c in range(C)]))
        alpha[i, c] = 1.0
    if verbose:
        print(f"  mosek-native: solsta={solsta} C_max={xx[CMAX]:.2f} ms "
              f"vars={numvar} rows={len(rows)} pairs={P} wall={wall:.1f}s")
    return t, alpha
