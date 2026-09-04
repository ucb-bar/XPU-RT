"""Search-based schedulers built on the shared SGS decoder.

Three families, all driving `schedule_decoder.decode`:

  - :func:`heft_schedule` — HEFT (Topcuoglu, Hariri & Wu 2002). Order ops by
    *upward rank* (longest path to a sink) instead of by earliest completion,
    then give each one the combination that finishes it soonest, allowing
    insertion into idle gaps. The rank is what the greedy pickers lack: they
    pick whichever ready op finishes first, which repeatedly defers the long
    chain that actually sets the makespan.

  - :func:`pso_schedule` — particle swarm over a random-key encoding
    (Bean 1994): each particle is N priority keys decoded by the SGS, which
    places each op by its own earliest-finish rule. PSO needs no gradient, no
    permutation-repair operator, and no external solver. The encoding used to
    carry N more keys forcing a machine combination per op; measured against
    the wl_sweep corpus that half was worth 10-24% to *delete* (see
    :func:`_combo_from_keys`), and `defer_frac` is what brings it back.

  - :func:`sa_schedule` — simulated annealing over the same encoding, as a
    cheap control for "is the swarm actually doing anything a random walk
    with a temperature wouldn't".

The two searches are seeded with every cheap heuristic's schedule — HEFT,
deadline-aware HEFT and the greedy pickers — and return the best of those
unless the search strictly beats it, so they can never be worse than the
0-second constructive answer they started from. Fitness is lexicographic
(missed periodic windows first, then makespan), flattened into a scalar with a
penalty large enough that no makespan gain can buy a missed deadline.
"""

from __future__ import annotations

import time

import numpy as np

from schedule_decoder import DecoderContext, decode, evaluate


#: Share of the combination-key range that means "no forced combination", so
#: that 1.0 drops the combination half of the encoding entirely and searches
#: only the priority order. See :func:`_combo_from_keys` for why that is the
#: default and :func:`_key_dim` for what it costs.
DEFER_FRACTION = 1.0


def _feasible_combos(ctx: DecoderContext):
    """Per-op table of the combinations the op can actually run on.

    `(index, count, position)`: `index[i, :count[i]]` lists op *i*'s usable
    combinations and `position[i, c]` is where combination *c* sits in that
    list (-1 if unusable). Cached on the context, which is itself built once
    per solve and reused across thousands of decodes.
    """
    tbl = getattr(ctx, "_mh_feasible", None)
    if tbl is not None:
        return tbl
    usable = [np.flatnonzero(np.isfinite(ctx.dur[i])) for i in range(ctx.n)]
    width = max((len(u) for u in usable), default=1) or 1
    index = np.zeros((ctx.n, width), dtype=int)
    count = np.zeros(ctx.n, dtype=int)
    position = np.full((ctx.n, ctx.n_combos), -1, dtype=int)
    for i, u in enumerate(usable):
        count[i] = u.size
        if u.size:
            index[i, :u.size] = u
            position[i, u] = np.arange(u.size)
    tbl = (index, count, position)
    ctx._mh_feasible = tbl
    return tbl


def _combo_from_keys(ctx: DecoderContext, ckeys: np.ndarray,
                     defer_frac: float) -> np.ndarray:
    """Map the combination half of a key vector onto combination indices.

    Two things separate this from a flat `floor(key * n_combos)`:

      - Keys below `defer_frac` decode to -1, which `decode` reads as "use the
        earliest-finish rule for this op" — the same rule HEFT uses. Forcing a
        combination on *every* op is what made the second half of the vector
        harmful rather than merely useless: a random-key placement displaces
        the decoder's earliest-finish choice on every uncontended op too, and
        the search then spends its budget undoing that.
      - The rest of the range indexes each op's *usable* combinations rather
        than all of them. This one closes a latent hole rather than a measured
        one: forcing an op onto a combination it cannot run on is *rewarded*,
        not punished, because `decode` falls through to its "no usable
        combination" branch, parks the op at combination 0 with duration 0 and
        commits nothing, and `evaluate` then scores it as a free, instantaneous
        op. Every op in the wl_sweep corpus can run on every combination, so
        nothing there could reach it; a workload with a restricted op could.

    Swept over the wl_sweep corpus at 0, 0.25, 0.5, 0.75 and 1.0, only the
    endpoint pays: partial deferral is worth hundredths of a percent, and full
    deferral is worth 10-24% on every spec where two ops ever contend for a
    lane. Hence the 1.0 default — on this corpus the combination half of the
    encoding is not a search dimension worth having, and the search is better
    off spending its whole budget on the priority order.
    """
    keys = np.clip(np.asarray(ckeys, dtype=float), 0.0, 1.0 - 1e-12)
    index, count, _pos = _feasible_combos(ctx)
    combo = np.full(ctx.n, -1, dtype=int)
    if defer_frac >= 1.0:
        return combo
    forced = (keys >= defer_frac) & (count > 0)
    rows = np.flatnonzero(forced)
    if rows.size:
        scaled = (keys[rows] - defer_frac) / (1.0 - defer_frac)
        slot = np.floor(np.clip(scaled, 0.0, 1.0 - 1e-12) * count[rows]).astype(int)
        combo[rows] = index[rows, slot]
    return combo


def _combo_keys_for(ctx: DecoderContext, alpha: np.ndarray,
                    defer_frac: float) -> np.ndarray:
    """Inverse of :func:`_combo_from_keys`: keys that force `alpha`'s combos."""
    if defer_frac >= 1.0:
        return np.zeros(0)
    _index, count, position = _feasible_combos(ctx)
    slot = position[np.arange(ctx.n), np.argmax(alpha, axis=1)]
    frac = np.where(slot >= 0, (slot + 0.5) / np.maximum(count, 1), 0.5)
    return defer_frac + (1.0 - defer_frac) * frac


def _key_dim(ctx: DecoderContext, defer_frac: float) -> int:
    """Length of one particle: N priority keys, plus N combination keys unless
    placement is fully deferred, in which case there is nothing to encode and
    carrying the half would only dilute SA's mutations across dead keys."""
    return ctx.n if defer_frac >= 1.0 else 2 * ctx.n


def _fitness(ctx: DecoderContext, keys: np.ndarray, penalty: float,
             restrict: bool,
             defer_frac: float = DEFER_FRACTION) -> tuple[float, np.ndarray, np.ndarray]:
    n = ctx.n
    priority = keys[:n]
    combo = _combo_from_keys(ctx, keys[n:], defer_frac)
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


def _keys_from_schedule(ctx: DecoderContext, t: np.ndarray, alpha: np.ndarray,
                        defer_frac: float = DEFER_FRACTION) -> np.ndarray:
    """Recover the key vector that makes the SGS reproduce a given schedule.

    Start-time order is the schedule's own priority order, so ranking by
    descending start time and normalising gives keys that decode back to
    (approximately) the same sequence — approximately because the SGS still
    enforces precedence, which can only improve on an ordering that violated
    it. Combination keys are read straight off `alpha`, into the forced band
    so the seed reproduces the heuristic's placement rather than deferring it.
    """
    order = np.argsort(np.argsort(-np.asarray(t, dtype=float)))
    prio = 1.0 - order / max(1, ctx.n - 1)
    return np.concatenate([prio, _combo_keys_for(ctx, alpha, defer_frac)])


def _true_fitness(ctx: DecoderContext, t, alpha, penalty, restrict) -> float:
    obj, misses, _ = evaluate(ctx, t, alpha, restrict)
    return obj + penalty * misses


def _heuristic_seeds(ctx: DecoderContext, workload,
                     defer_frac: float = DEFER_FRACTION):
    """Key vectors for every cheap heuristic, as starting points.

    Seeding only from HEFT was actively harmful: HEFT ignores periodic
    windows, so on a workload with tight periods its schedule carries a large
    miss penalty, and the search spends its whole budget climbing out of that
    basin instead of improving a schedule that was already deadline-feasible.
    The greedy pickers are cheap (tenths of a second) and land in the feasible
    region, so they belong in the initial population.

    `heft_edf` belongs there for the same reason and was the costly omission:
    on a workload where plain HEFT misses windows it is usually the *only*
    deadline-feasible schedule anywhere near HEFT's makespan, so leaving it
    out meant the search started from a greedy picker several percent worse
    and — however well it then searched — could only be reported as a loss
    against a heuristic that cost 0.07 s.

    Returns (keys, t, alpha) triples. The schedule is carried alongside the
    keys because the key round-trip is lossy: re-decoding a heuristic's keys
    can land somewhere slightly worse than the heuristic itself, and without
    the original schedule as the incumbent the search would report that
    regression as its answer.
    """
    import greedy_scheduler as _gs
    seeds = []
    ht, ha, hkeys = _heft_keys(ctx)
    if defer_frac >= 1.0:
        hkeys = hkeys[:ctx.n]
    elif defer_frac > 0.0:
        # HEFT *is* the decoder's earliest-finish rule, so a deferred
        # combination half reproduces it exactly instead of approximately.
        hkeys = np.concatenate([hkeys[:ctx.n], np.full(ctx.n, defer_frac * 0.5)])
    else:
        hkeys = np.concatenate([hkeys[:ctx.n], _combo_keys_for(ctx, ha, 0.0)])
    seeds.append((hkeys, ht, ha))
    fns = [heft_edf_schedule]
    for name in ("greedy_reserved_schedule", "greedy_periodic_schedule",
                 "greedy_schedule", "decomposed_schedule"):
        fn = getattr(_gs, name, None)
        if fn is not None:
            fns.append(fn)
    for fn in fns:
        try:
            t, alpha = fn(workload)
            seeds.append((_keys_from_schedule(ctx, t, alpha, defer_frac), t, alpha))
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
                 stall_iters: int | None = 8,
                 defer_frac: float = DEFER_FRACTION,
                 verbose: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Particle swarm over random keys. Returns the best schedule found.

    Two things the swarm proper does not do:

      - The *incumbent* — what gets returned — is tracked separately from
        `gbest`, the swarm's leader. `gbest` is ranked on decoded key vectors,
        because that is the space the velocity update moves in; the incumbent
        is ranked over those *and* over each seed heuristic's own schedule.
        Ranking one thing on both made the leader a position whose recorded
        fitness no particle could reproduce, so the swarm had no gradient to
        follow; keeping them apart also makes "never worse than the best
        seed" a property of the return statement rather than a coincidence.
      - `stall_iters` stops the search once `stall_iters` consecutive
        iterations improve nothing. On the workloads where the seed is already
        the best thing the encoding can express, the swarm otherwise spends
        its entire budget rediscovering it.
    """
    ctx = DecoderContext(workload)
    rng = np.random.default_rng(seed)
    dim = _key_dim(ctx, defer_frac)
    penalty = _penalty_scale(ctx)
    started = time.perf_counter()

    seeds = _heuristic_seeds(ctx, workload, defer_frac)
    inc_fit, inc_t, inc_alpha = np.inf, None, None
    for _sd, st, sa in seeds:
        fit_true = _true_fitness(ctx, st, sa, penalty, restrict_to_nonperiodic)
        if fit_true < inc_fit:
            inc_fit, inc_t, inc_alpha = fit_true, st, sa
    seed_fit = inc_fit

    x = rng.random((n_particles, dim))
    gbest, gbest_fit = None, np.inf
    for k, (sd, _st, _sa) in enumerate(seeds[:n_particles]):
        x[k] = sd
        fit, t, alpha = _fitness(ctx, sd, penalty, restrict_to_nonperiodic, defer_frac)
        if fit < gbest_fit:
            gbest, gbest_fit = sd.copy(), fit
        if fit < inc_fit:
            inc_fit, inc_t, inc_alpha = fit, t, alpha
    if gbest is None:
        gbest = x[0].copy()
    # A few jittered copies of the best seed, so the swarm explores around the
    # good basin rather than only from uniform noise.
    for k in range(len(seeds), min(len(seeds) + 3, n_particles)):
        x[k] = np.clip(gbest + rng.normal(0, 0.08, dim), 0, 1)
    v = rng.normal(0, 0.1, (n_particles, dim))

    pbest = x.copy()
    pbest_fit = np.full(n_particles, np.inf)
    stall = 0
    stopped = "iteration limit"
    it = -1

    for it in range(iters):
        improved = False
        for p in range(n_particles):
            fit, t, alpha = _fitness(ctx, x[p], penalty, restrict_to_nonperiodic,
                                     defer_frac)
            if fit < pbest_fit[p]:
                pbest_fit[p], pbest[p] = fit, x[p].copy()
            if fit < gbest_fit:
                gbest_fit, gbest = fit, x[p].copy()
                improved = True
            if fit < inc_fit:
                inc_fit, inc_t, inc_alpha = fit, t, alpha
                improved = True
        stall = 0 if improved else stall + 1
        if stall_iters is not None and stall >= stall_iters:
            stopped = f"no gain for {stall} iterations"
            break
        if time_budget is not None and time.perf_counter() - started > time_budget:
            stopped = "time budget"
            break
        r1 = rng.random((n_particles, dim))
        r2 = rng.random((n_particles, dim))
        v = w * v + c1 * r1 * (pbest - x) + c2 * r2 * (gbest - x)
        np.clip(v, -0.5, 0.5, out=v)
        x = np.clip(x + v, 0.0, 1.0)

    if verbose:
        print(f"  pso: stopped at iteration {it + 1}/{iters} on {stopped}; "
              f"fitness {inc_fit:.3f} (best seed {seed_fit:.3f})")
    return inc_t, inc_alpha


def sa_schedule(workload, iters: int = 4000, seed: int = 0,
                time_budget: float | None = 20.0,
                restrict_to_nonperiodic: bool = True,
                t0: float = 0.25, t1: float = 0.005,
                stall_iters: int | None = 500,
                defer_frac: float = DEFER_FRACTION,
                verbose: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Simulated annealing over the same random-key encoding.

    Carries the same two corrections as :func:`pso_schedule`: the returned
    incumbent is ranked over the seed heuristics' own schedules as well as
    over decoded key vectors, and the walk gives up after `stall_iters`
    consecutive steps that improve nothing.
    """
    ctx = DecoderContext(workload)
    rng = np.random.default_rng(seed)
    penalty = _penalty_scale(ctx)
    started = time.perf_counter()

    seeds = _heuristic_seeds(ctx, workload, defer_frac)
    inc_fit, inc_t, inc_alpha = np.inf, None, None
    for _sd, st, sa in seeds:
        fit_true = _true_fitness(ctx, st, sa, penalty, restrict_to_nonperiodic)
        if fit_true < inc_fit:
            inc_fit, inc_t, inc_alpha = fit_true, st, sa
    seed_fit = inc_fit

    cur, cur_fit = None, np.inf
    for sd, _st, _sa in seeds:
        fit, t, alpha = _fitness(ctx, sd, penalty, restrict_to_nonperiodic, defer_frac)
        if fit < cur_fit:
            cur, cur_fit = sd.copy(), fit
        if fit < inc_fit:
            inc_fit, inc_t, inc_alpha = fit, t, alpha
    dim = cur.size
    # Perturb a handful of keys per step: a full-vector resample is just a
    # random restart, and a single key rarely changes the decoded order.
    n_mut = max(1, dim // 50)
    stall = 0
    stopped = "iteration limit"
    it = 0

    for it in range(iters):
        if time_budget is not None and time.perf_counter() - started > time_budget:
            stopped = "time budget"
            break
        if stall_iters is not None and stall >= stall_iters:
            stopped = f"no gain for {stall} steps"
            break
        temp = t0 * (t1 / t0) ** (it / max(1, iters - 1))
        cand = cur.copy()
        idx = rng.integers(0, dim, n_mut)
        cand[idx] = np.clip(cand[idx] + rng.normal(0, 0.2, n_mut), 0.0, 1.0)
        fit, t, alpha = _fitness(ctx, cand, penalty, restrict_to_nonperiodic, defer_frac)
        if fit < cur_fit or rng.random() < np.exp(-(fit - cur_fit) / max(temp, 1e-9)):
            cur, cur_fit = cand, fit
        if fit < inc_fit:
            inc_fit, inc_t, inc_alpha = fit, t, alpha
            stall = 0
        else:
            stall += 1

    if verbose:
        print(f"  sa: stopped at iteration {it}/{iters} on {stopped}; "
              f"fitness {inc_fit:.3f} (best seed {seed_fit:.3f})")
    return inc_t, inc_alpha
