"""Search-based schedulers built on the shared SGS decoder.

Three families, all driving `schedule_decoder.decode`:

  - :func:`heft_schedule` — HEFT (Topcuoglu, Hariri & Wu 2002). Order ops by
    *upward rank* (longest path to a sink) instead of by earliest completion,
    then give each one the combination that finishes it soonest, allowing
    insertion into idle gaps. The rank is what the greedy pickers lack: they
    pick whichever ready op finishes first, which repeatedly defers the long
    chain that actually sets the makespan.

  - :func:`heft_edf_schedule` — the same, plus a deadline band for the periodic
    ops that have no slack to give, chosen by decoding every gate the laxity
    distribution admits and keeping the best feasible one.

  - :func:`pso_schedule` — particle swarm over a random-key encoding
    (Bean 1994): each particle is 2N reals, N priority keys plus N
    combination keys, decoded by the SGS. PSO needs no gradient, no
    permutation-repair operator, and no external solver.

  - :func:`sa_schedule` — simulated annealing over the same encoding, as a
    cheap control for "is the swarm actually doing anything a random walk
    with a temperature wouldn't".

All three are seeded with the HEFT solution, which both guarantees they never
return something worse than HEFT and gives the search a sane starting basin.
Fitness is lexicographic (missed periodic windows first, then makespan),
flattened into a scalar with a penalty large enough that no makespan gain can
buy a missed deadline.
"""

from __future__ import annotations

import time

import numpy as np

from schedule_decoder import DecoderContext, decode, evaluate


def _fitness(ctx: DecoderContext, keys: np.ndarray, penalty: float,
             restrict: bool) -> tuple[float, np.ndarray, np.ndarray]:
    n = ctx.n
    priority = keys[:n]
    combo = np.floor(np.clip(keys[n:], 0.0, 0.999999) * ctx.n_combos).astype(int)
    t, alpha = decode(ctx, priority, combo)
    obj, misses, _all_end = evaluate(ctx, t, alpha, restrict)
    return obj + penalty * misses, t, alpha


def _heft_keys(ctx: DecoderContext) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """HEFT's own solution, plus the key vector that reproduces it."""
    rank = ctx.upward_rank()
    t, alpha = decode(ctx, rank, None)
    span = rank.max() - rank.min()
    prio = (rank - rank.min()) / span if span > 0 else np.full(ctx.n, 0.5)
    return t, alpha, np.concatenate([prio, _combo_keys(ctx, alpha)])


def _combo_keys(ctx: DecoderContext, alpha: np.ndarray) -> np.ndarray:
    return (np.argmax(alpha, axis=1) + 0.5) / max(1, ctx.n_combos)


def _keys_from_schedule(ctx: DecoderContext, t: np.ndarray,
                        alpha: np.ndarray) -> np.ndarray:
    """Recover the key vector that makes the SGS reproduce a given schedule.

    Start-time order is the schedule's own priority order, so ranking by
    descending start time and normalising gives keys that decode back to
    (approximately) the same sequence — approximately because the SGS still
    enforces precedence, which can only improve on an ordering that violated
    it. Combination keys are read straight off `alpha`.
    """
    order = np.argsort(np.argsort(-np.asarray(t, dtype=float)))
    prio = 1.0 - order / max(1, ctx.n - 1)
    return np.concatenate([prio, _combo_keys(ctx, alpha)])


def _true_fitness(ctx: DecoderContext, t, alpha, penalty, restrict) -> float:
    obj, misses, _ = evaluate(ctx, t, alpha, restrict)
    return obj + penalty * misses


def _heuristic_seeds(ctx: DecoderContext, workload):
    """Key vectors for every cheap heuristic, as starting points.

    Seeding only from HEFT was actively harmful: HEFT ignores periodic
    windows, so on a workload with tight periods its schedule carries a large
    miss penalty, and the search spends its whole budget climbing out of that
    basin instead of improving a schedule that was already deadline-feasible.
    The greedy pickers are cheap (tenths of a second) and land in the feasible
    region, so they belong in the initial population.

    Returns (keys, t, alpha) triples. The schedule is carried alongside the
    keys because the key round-trip is lossy: re-decoding a heuristic's keys
    can land somewhere slightly worse than the heuristic itself, and without
    the original schedule as the incumbent the search would report that
    regression as its answer.
    """
    import greedy_scheduler as _gs
    seeds = []
    ht, ha, hkeys = _heft_keys(ctx)
    seeds.append((hkeys, ht, ha))
    for fn in (_gs.greedy_reserved_schedule, _gs.greedy_periodic_schedule,
               _gs.greedy_schedule, _gs.decomposed_schedule):
        try:
            t, alpha = fn(workload)
            seeds.append((_keys_from_schedule(ctx, t, alpha), t, alpha))
        except Exception:
            continue
    return seeds


def _penalty_scale(ctx: DecoderContext) -> float:
    """One missed window must cost more than any achievable makespan gain."""
    finite = ctx.dur[np.isfinite(ctx.dur)]
    total = float(finite.sum()) if finite.size else 1.0
    return max(1.0, total)


def heft_schedule(workload) -> tuple[np.ndarray, np.ndarray]:
    """HEFT: upward-rank priority + insertion-based earliest-finish placement."""
    ctx = DecoderContext(workload)
    t, alpha, _keys = _heft_keys(ctx)
    return t, alpha


def total_float(ctx: DecoderContext) -> np.ndarray:
    """CPM total float per op: how long it can slip before a window is missed.

    Forward pass gives the earliest finish reachable at all,

        eft[i] = max(min_start[i], max over preds p of eft[p]) + min_dur[i]

    backward pass the latest finish that still leaves the chain feasible,

        lft[i] = min(max_end[i], min over succs s of (lft[s] - min_dur[s]))

    and `lft - eft` is the slack. Both use `min_dur` — the fastest combination
    the op could run on — so the float is a genuine upper bound on the
    breathing room, never an artefact of a slow combination we would not pick.

    The backward pass is what makes this usable as a laxity signal at all. An
    op's own `max_end_t` is the *instance* window (every op of a 30-op dronet
    instance carries the same 20 ms), so the raw `max_end - eft` of the head of
    a chain looks enormous even when the 29 ops behind it leave no room. The
    backward pass pushes the real deadline up the chain, and every op in a pure
    chain ends up with the same float — which is the right semantics, because
    an instance is tight or slack as a whole, not op by op.

    Non-periodic ops with no periodic successor get `+inf`, as they should.
    """
    eft = np.zeros(ctx.n)
    for u in ctx.topo:
        es = ctx.min_start[u]
        for p in ctx.pred[u]:
            if eft[p] > es:
                es = eft[p]
        eft[u] = es + ctx.min_dur[u]
    lft = np.array(ctx.max_end, dtype=float)
    for u in reversed(ctx.topo):
        for s in ctx.succ[u]:
            v = lft[s] - ctx.min_dur[s]
            if v < lft[u]:
                lft[u] = v
    return lft - eft


def _span_lower_bound(ctx: DecoderContext) -> float:
    """Lower bound on any schedule's length: critical path, or the work each
    machine must average, whichever binds. Dividing float by it makes laxity
    scale-free, so the same numbers mean the same thing on a 35 ms control
    workload and on a 3.7 s vision one."""
    rank = ctx.upward_rank()
    cp = float(rank.max()) if ctx.n else 0.0
    per_machine = float(ctx.min_dur.sum()) / max(len(ctx.machines), 1)
    return max(cp, per_machine, 1e-9)


def laxity(ctx: DecoderContext) -> np.ndarray:
    """`total_float` normalised by `_span_lower_bound` — "how much of a whole
    schedule's worth of slack does this op have"."""
    return total_float(ctx) / _span_lower_bound(ctx)


# Ceiling on how many laxity gates `heft_edf_schedule` decodes. A gate can only
# sit between two adjacent distinct laxity *levels*, and laxity is close to a
# property of a periodic network rather than of an op — the ops of an instance
# chain share a float — so the whole family is 7-8 gates wide on the wl_sweep
# corpus and this ceiling never binds there. It exists so a pathological
# workload with hundreds of levels cannot turn a tenth-of-a-second heuristic
# into a minute-long one; above it the levels are thinned by index, which always
# keeps the two endpoints.
_MAX_LAXITY_GATES = 12

# Laxity values closer together than this (relatively) are the same level. The
# float of a chain is a sum of the same durations in a different association
# order, so ops that are genuinely equally tight come back differing in the last
# couple of ulps: control_mix_hetero has 140 periodic ops at 32 "distinct"
# laxities but only 6 levels, the other 26 being spread of order 1e-16. Gating
# inside that spread splits an instance chain at a boundary decided by
# floating-point noise — it does move the makespan (up to 2% here), but not
# reproducibly, so the levels are merged first.
_LAXITY_RTOL = 1e-9


def _band_priority(ctx: DecoderContext, np_band: np.ndarray,
                   lift: np.ndarray) -> np.ndarray:
    """Rank priority in [0, 1], with `lift` raised into the EDF band [1, 2]."""
    priority = np.array(np_band, dtype=float)
    if np.any(lift):
        d = ctx.max_end[lift]
        finite = d[np.isfinite(d)]
        lo, hi = (finite.min(), finite.max()) if finite.size else (0.0, 1.0)
        rng = (hi - lo) or 1.0
        priority[lift] = 2.0 - np.clip((d - lo) / rng, 0.0, 1.0)
    return priority


def laxity_levels(lax: np.ndarray) -> np.ndarray:
    """Sorted laxity values with floating-point-identical ones merged.

    See `_LAXITY_RTOL`. Each surviving value is the *largest* of its level, so
    `lax <= level` selects the whole level and never a noise-decided part of it.
    """
    vals = np.unique(lax)
    if vals.size == 0:
        return vals
    keep = [vals[0]]
    for v in vals[1:]:
        if v - keep[-1] > _LAXITY_RTOL * max(1.0, abs(keep[-1])):
            keep.append(v)
        else:
            keep[-1] = v                   # same level, take its top edge
    return np.array(keep)


def laxity_gates(ctx: DecoderContext, gate=None) -> list[np.ndarray]:
    """The lift masks `heft_edf_schedule` will try, tightest first, deduped.

    With `gate=None` this enumerates the family: "lift nothing", then "lift
    every periodic op at or below the k-th laxity level" for each k. The last is
    "lift every periodic op" — the unconditional band — so both endpoints are
    always present however hard the list is thinned.

    An explicit `gate` (a float, or a sequence of floats) instead uses those as
    literal thresholds, `laxity < gate`, which is what pins the behaviour in
    tests and lets a caller ask for one fixed cut.
    """
    per = ctx.periodic
    lax = laxity(ctx)
    if gate is not None:
        thresholds = (float(gate),) if np.isscalar(gate) else tuple(gate)
        masks = [per & (lax < th) for th in thresholds]
    else:
        levels = laxity_levels(lax[per])
        if levels.size > _MAX_LAXITY_GATES - 1:
            keep = np.unique(np.linspace(0, levels.size - 1, _MAX_LAXITY_GATES - 1)
                             .round().astype(int))
            levels = levels[keep]
        masks = [np.zeros(ctx.n, dtype=bool)] + [per & (lax <= v) for v in levels]
    out, seen = [], set()
    for m in masks:
        key = np.packbits(m).tobytes()
        if key not in seen:
            seen.add(key)
            out.append(m)
    return out


def heft_edf_schedule(workload, gate=None,
                      restrict_to_nonperiodic: bool = True
                      ) -> tuple[np.ndarray, np.ndarray]:
    """Deadline-aware HEFT: EDF for the *tight* periodic ops, rank for the rest.

    Plain HEFT is deadline-blind — it orders purely by distance to a sink — so
    on any workload with periodic windows it packs the long non-periodic chain
    first and every periodic instance lands late. Measured over 30 generated
    spike workloads it produced *zero* valid schedules, missing windows on all
    30 with a median worst-lateness of 728 ms.

    The fix keeps HEFT's insight (order the makespan-critical chain by upward
    rank, not by who finishes soonest) but puts it in the band *below* the
    periodic work: periodic ops are ordered among themselves by deadline —
    earliest deadline first, the classic EDF rule — and outrank every
    non-periodic op, which then backfills the gaps the SGS leaves.

    Lifting *every* periodic op is what that costs. On configurations with
    enough machines the deadline band is pure loss: on the wl_sweep corpus's
    4-machine "quad" configs, where plain HEFT was already valid, the band cost
    +7.4% on control_mix, +14.7% on saturation and +5.0% on vint_multi, and
    bought nothing. So the lift is gated on `laxity`: only periodic ops with
    less than a threshold's worth of slack are promoted, and the roomy ones
    stay in the rank band where they backfill like ordinary work.

    No *fixed* threshold does that job, and the sweep behind this says so
    plainly. Laxity is very nearly configuration-invariant — the same ops are
    tight on two machines and on four, because a window and a chain of work
    don't change when you add a core — while the amount of crowding those ops
    must survive changes a great deal. On saturation the gate that recovers
    HEFT's makespan on quad misses ten windows on hetero, and every constant
    that is safe on hetero gives quad's 14.7% straight back. Normalising the
    slack by a span lower bound, by the window, or leaving it in milliseconds
    all hit the same wall.

    What rescues it is that the gate family is *tiny*. A gate can only fall
    between two adjacent laxity *levels* (`laxity_levels`), and laxity is close
    to a property of a periodic network rather than of an op — the ops of an
    instance chain share a float — so the corpus's 24 workloads admit 7 to 8
    gates each, endpoints included. There is nothing left to tune: enumerate
    them (see `laxity_gates`), decode each, and keep the best by (missed
    windows, makespan), lexicographic — exactly the fitness the searches in
    this module already use. Because "lift everything" is always in the list
    the result is never worse than the unconditional band, and because "lift
    nothing" is too it is never worse than plain HEFT either. That guard is
    what makes a gate this aggressive safe to ship: a gate that would drop a
    deadline is decoded, scored, and discarded before it can be returned.

    Cost is one decode per gate — tenths of a second on the largest workload
    here, against the 20 s the searches in this module take — and
    `_MAX_LAXITY_GATES` caps it.

    `gate` overrides the enumeration with literal thresholds: a float for one
    fixed cut, or `float("inf")` to restore the old unconditional band.
    """
    ctx = DecoderContext(workload)
    rank = ctx.upward_rank()
    span = rank.max() - rank.min()
    np_band = (rank - rank.min()) / span if span > 0 else np.full(ctx.n, 0.5)

    if not np.any(ctx.periodic):           # nothing to band: this is HEFT
        return decode(ctx, np.array(np_band, dtype=float), None)

    best = None                            # (misses, objective, -lifted, t, alpha)
    for lift in laxity_gates(ctx, gate):
        t, alpha = decode(ctx, _band_priority(ctx, np_band, lift), None)
        obj, misses, _ = evaluate(ctx, t, alpha, restrict_to_nonperiodic)
        # Ties go to the wider gate: when the objective cannot tell two
        # schedules apart, the one protecting more deadlines is the safer bet.
        cand = (misses, obj, -int(lift.sum()), t, alpha)
        if best is None or cand[:3] < best[:3]:
            best = cand
    return best[3], best[4]


def pso_schedule(workload, n_particles: int = 24, iters: int = 60,
                 seed: int = 0, time_budget: float | None = 20.0,
                 restrict_to_nonperiodic: bool = True,
                 w: float = 0.72, c1: float = 1.5, c2: float = 1.5,
                 verbose: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Particle swarm over random keys. Returns the best schedule found."""
    ctx = DecoderContext(workload)
    rng = np.random.default_rng(seed)
    dim = 2 * ctx.n
    penalty = _penalty_scale(ctx)
    started = time.perf_counter()

    seeds = _heuristic_seeds(ctx, workload)
    x = rng.random((n_particles, dim))
    gbest, gbest_fit, best_t, best_alpha = None, np.inf, None, None
    for k, (sd, st, sa) in enumerate(seeds[:n_particles]):
        x[k] = sd
        # Rank the swarm by the decoded key vector, but hold the incumbent
        # against the heuristic's own schedule, which may be strictly better.
        fit_keys, t, alpha = _fitness(ctx, sd, penalty, restrict_to_nonperiodic)
        if fit_keys < gbest_fit:
            gbest, gbest_fit, best_t, best_alpha = sd.copy(), fit_keys, t, alpha
        fit_true = _true_fitness(ctx, st, sa, penalty, restrict_to_nonperiodic)
        if fit_true < gbest_fit:
            gbest, gbest_fit, best_t, best_alpha = sd.copy(), fit_true, st, sa
    # A few jittered copies of the best seed, so the swarm explores around the
    # good basin rather than only from uniform noise.
    for k in range(len(seeds), min(len(seeds) + 3, n_particles)):
        x[k] = np.clip(gbest + rng.normal(0, 0.08, dim), 0, 1)
    v = rng.normal(0, 0.1, (n_particles, dim))

    pbest = x.copy()
    pbest_fit = np.full(n_particles, np.inf)
    seed_fit = gbest_fit

    for it in range(iters):
        for p in range(n_particles):
            fit, t, alpha = _fitness(ctx, x[p], penalty, restrict_to_nonperiodic)
            if fit < pbest_fit[p]:
                pbest_fit[p], pbest[p] = fit, x[p].copy()
            if fit < gbest_fit:
                gbest_fit, gbest = fit, x[p].copy()
                best_t, best_alpha = t, alpha
        if time_budget is not None and time.perf_counter() - started > time_budget:
            if verbose:
                print(f"  pso: stopped at iteration {it + 1}/{iters} on time budget")
            break
        r1 = rng.random((n_particles, dim))
        r2 = rng.random((n_particles, dim))
        v = w * v + c1 * r1 * (pbest - x) + c2 * r2 * (gbest - x)
        np.clip(v, -0.5, 0.5, out=v)
        x = np.clip(x + v, 0.0, 1.0)

    if verbose:
        print(f"  pso: best fitness {gbest_fit:.3f} (best seed {seed_fit:.3f})")
    return best_t, best_alpha


def sa_schedule(workload, iters: int = 4000, seed: int = 0,
                time_budget: float | None = 20.0,
                restrict_to_nonperiodic: bool = True,
                t0: float = 0.25, t1: float = 0.005,
                verbose: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Simulated annealing over the same random-key encoding."""
    ctx = DecoderContext(workload)
    rng = np.random.default_rng(seed)
    penalty = _penalty_scale(ctx)
    started = time.perf_counter()

    best, best_fit, best_t, best_alpha = None, np.inf, None, None
    cur, cur_fit = None, np.inf
    for sd, st, sa in _heuristic_seeds(ctx, workload):
        fit_keys, t, alpha = _fitness(ctx, sd, penalty, restrict_to_nonperiodic)
        if fit_keys < cur_fit:
            cur, cur_fit = sd.copy(), fit_keys
        if fit_keys < best_fit:
            best, best_fit, best_t, best_alpha = sd.copy(), fit_keys, t, alpha
        fit_true = _true_fitness(ctx, st, sa, penalty, restrict_to_nonperiodic)
        if fit_true < best_fit:
            best, best_fit, best_t, best_alpha = sd.copy(), fit_true, st, sa
    dim = cur.size
    # Perturb a handful of keys per step: a full-vector resample is just a
    # random restart, and a single key rarely changes the decoded order.
    n_mut = max(1, dim // 50)

    for it in range(iters):
        if time_budget is not None and time.perf_counter() - started > time_budget:
            if verbose:
                print(f"  sa: stopped at iteration {it}/{iters} on time budget")
            break
        temp = t0 * (t1 / t0) ** (it / max(1, iters - 1))
        cand = cur.copy()
        idx = rng.integers(0, dim, n_mut)
        cand[idx] = np.clip(cand[idx] + rng.normal(0, 0.2, n_mut), 0.0, 1.0)
        fit, t, alpha = _fitness(ctx, cand, penalty, restrict_to_nonperiodic)
        if fit < cur_fit or rng.random() < np.exp(-(fit - cur_fit) / max(temp, 1e-9)):
            cur, cur_fit = cand, fit
            if fit < best_fit:
                best, best_fit, best_t, best_alpha = cand.copy(), fit, t, alpha

    if verbose:
        print(f"  sa: best fitness {best_fit:.3f}")
    return best_t, best_alpha
