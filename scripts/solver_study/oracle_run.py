"""Oracle gap study, upper-bound half: best solution obtainable at max effort.

The frozen sweep (`data/wl_sweep_baseline.json`) reports each solver at its
standard budget, so "best" there means "best among the methods we ran at 20 s /
60 s". This driver spends real time on a handful of specs instead: it seeds
CP-SAT with the best VALID cheap schedule and lets it run for 600-1800 s at
several random seeds, recording CP-SAT's own proven objective bound alongside
the incumbent.

Validity is the gate. `misses > 0` means a periodic window was overrun, and a
schedule that overruns a window is not a solution to our problem -- it is
excluded from "best known" no matter how small its objective. Every row records
`misses` so an excluded row stays visible rather than disappearing.

Usage:  oracle_run.py <spec-name> <seed> <time-limit-s> [warm-json]
Writes one JSON per run into $XPURT_ORACLE_OUT.
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wl_sweep_bench import build, DATA          # noqa: E402  (sets sys.path)

from schedule_decoder import DecoderContext, evaluate     # noqa: E402
import greedy_scheduler as gs                             # noqa: E402
import metaheuristics as mh                               # noqa: E402
import cpsat_scheduler as cps                             # noqa: E402

OUT = os.environ.get("XPURT_ORACLE_OUT", ".")


def best_valid_heuristic(w, ctx, budget=30.0):
    """Best schedule among the cheap methods that misses no periodic window.

    CP-SAT only benefits from a hint it can complete feasibly, and an invalid
    hint is dropped -- so an infeasible-but-shorter schedule is worse than
    useless here even before the validity rule excludes it from the answer.
    """
    cands = {
        "greedy": lambda: gs.greedy_schedule(w),
        "greedy_periodic": lambda: gs.greedy_periodic_schedule(w),
        "greedy_reserved": lambda: gs.greedy_reserved_schedule(w),
        "decomposed": lambda: gs.decomposed_schedule(w),
        "heft": lambda: mh.heft_schedule(w),
        "heft_edf": lambda: mh.heft_edf_schedule(w),
        "pso": lambda: mh.pso_schedule(w, time_budget=budget, seed=0),
        "sa": lambda: mh.sa_schedule(w, time_budget=budget, seed=0),
    }
    rows, best = [], None
    for name, fn in cands.items():
        t0 = time.perf_counter()
        try:
            t, alpha = fn()
            wall = time.perf_counter() - t0
            obj, misses, all_end = evaluate(ctx, t, alpha, True)
        except Exception as e:                      # noqa: BLE001
            rows.append(dict(method=name, error=f"{type(e).__name__}: {e}"))
            continue
        rows.append(dict(method=name, objective=round(float(obj), 4),
                         misses=int(misses), all_ops=round(float(all_end), 4),
                         wall_s=round(wall, 2)))
        if misses == 0 and (best is None or obj < best[0]):
            best = (float(obj), name, (t, alpha), wall)
    return rows, best


def main():
    spec, seed, tl = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
    warm_json = sys.argv[4] if len(sys.argv) > 4 else None
    os.environ.setdefault(
        "XPURT_CPSAT_PYTHON",
        "/tmp/claude-1172/-scratch2-dima-misc-sw-XPU-RT/"
        "cb67e7aa-73a1-4f96-95bd-e31812fb2543/scratchpad/cpsat-venv/bin/python")

    w, nd = build(f"{DATA}/data/toplevel/wl_sweep/{spec}.json")
    ctx = DecoderContext(w)

    if warm_json:
        # Chained polish: re-seed from a schedule an earlier round found.
        d = json.load(open(warm_json))
        t = np.array(d["start"], dtype=float)
        alpha = np.zeros((ctx.n, ctx.n_combos))
        for i, c in enumerate(d["combo"]):
            alpha[i, int(c)] = 1.0
        obj, misses, _ = evaluate(ctx, t, alpha, True)
        if misses:
            raise SystemExit(f"refusing to seed from {warm_json}: it misses "
                             f"{misses} periodic windows")
        hrows, seed_obj, seed_name = [], float(obj), f"warm:{os.path.basename(warm_json)}"
        warm = (t, alpha)
    else:
        hrows, best = best_valid_heuristic(w, ctx)
        if best is None:
            json.dump(dict(spec=spec, seed=seed, time_limit=tl, heuristics=hrows,
                           note="no valid heuristic schedule; solving cold"),
                      open(f"{OUT}/{spec}.s{seed}.t{int(tl)}.json", "w"), indent=1)
            warm, seed_obj, seed_name = None, None, None
        else:
            seed_obj, seed_name, warm, _ = best

    t0 = time.perf_counter()
    err = None
    try:
        t, alpha = cps.cpsat_schedule(w, time_limit=tl, workers=4, verbose=True,
                                      warm_start=warm, random_seed=seed)
        wall = time.perf_counter() - t0
        obj, misses, all_end = evaluate(ctx, t, alpha, True)
        res = dict(objective=round(float(obj), 4), misses=int(misses),
                   all_ops=round(float(all_end), 4),
                   start=[float(x) for x in t],
                   combo=[int(c) for c in np.argmax(alpha, axis=1)])
    except Exception as e:                          # noqa: BLE001
        wall = time.perf_counter() - t0
        err = f"{type(e).__name__}: {e}"
        res = {}

    out = dict(spec=spec, seed=seed, time_limit=tl, wall_s=round(wall, 2),
               warm_from=seed_name, warm_objective=seed_obj,
               heuristics=hrows, cpsat=dict(cps.LAST_SOLVE), error=err, **res)
    json.dump(out, open(f"{OUT}/{spec}.s{seed}.t{int(tl)}.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items()
                      if k not in ("start", "combo", "heuristics")}, indent=1))


if __name__ == "__main__":
    main()
