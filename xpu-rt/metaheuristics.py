"""Search-based schedulers built on the shared SGS decoder.

Three families, all driving `schedule_decoder.decode`:

  - :func:`heft_schedule` — HEFT (Topcuoglu, Hariri & Wu 2002). Order ops by
    *upward rank* (longest path to a sink) instead of by earliest completion,
    then give each one the combination that finishes it soonest, allowing
    insertion into idle gaps. The rank is what the greedy pickers lack: they
    pick whichever ready op finishes first, which repeatedly defers the long
    chain that actually sets the makespan.

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


def heft_edf_schedule(workload) -> tuple[np.ndarray, np.ndarray]:
    """Deadline-aware HEFT: EDF for the periodic ops, upward rank for the rest.

    Plain HEFT is deadline-blind — it orders purely by distance to a sink — so
    on any workload with periodic windows it packs the long non-periodic chain
    first and every periodic instance lands late. Measured over 30 generated
    spike workloads it produced *zero* valid schedules, missing windows on all
    30 with a median worst-lateness of 728 ms.

    The fix keeps HEFT's insight (order the makespan-critical chain by upward
    rank, not by who finishes soonest) but puts it in the band *below* the
    periodic work: periodic ops are ordered among themselves by deadline —
    earliest deadline first, the classic EDF rule — and all of them outrank
    every non-periodic op, which then backfills the gaps the SGS leaves.
    """
    ctx = DecoderContext(workload)
    rank = ctx.upward_rank()
    span = rank.max() - rank.min()
    np_band = (rank - rank.min()) / span if span > 0 else np.full(ctx.n, 0.5)

    priority = np.array(np_band, dtype=float)          # non-periodic: [0, 1]
    per = ctx.periodic
    if np.any(per):
        # Periodic ops occupy [1, 2], ranked by deadline (earliest first).
        d = ctx.max_end[per]
        finite = d[np.isfinite(d)]
        lo, hi = (finite.min(), finite.max()) if finite.size else (0.0, 1.0)
        rng = (hi - lo) or 1.0
        priority[per] = 2.0 - np.clip((d - lo) / rng, 0.0, 1.0)
    return decode(ctx, priority, None)


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
