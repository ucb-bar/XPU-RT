#!/usr/bin/env python3
"""Did the actuation-parity swap change the simulation?

The question is EQUIVALENCE, not difference, and totals alone cannot answer it: the
wide sweep found same-experiment runs whose success COUNTS matched while the successful
EPISODES differed (seed 129, 11/11/11 with different vectors). So this reports, for each
config against the registered reference:

  rate        successes / episodes, with a design-corrected CI (clustering by episode
              config, the same DEFF = 1+(m-1)*ICC correction fig_metrics.py uses).
  agree       episodes where the two configs give the SAME outcome, out of n. This is
              the strong test -- two configs that are behaviourally identical should
              agree episode-by-episode, not merely in total.
  delta       rate difference, judged against the MEASURED harness noise floor rather
              than against zero.

NOISE FLOOR (measured, wide sweep, same-experiment run pairs; provisional -- it grew as
pairs accumulated, so treat as a lower bound):
    coke (google_robot)   95% band +/-2.50 pts; 83% of pairs bit-identical;
                          conditional on differing, median jump 6.25 pts, max 16.67
A single-seed n=24 screen therefore CANNOT establish equivalence -- one nondeterminism
firing moves a 24-episode run by a median of 1.5 successful episodes. It can only rule
out COLLAPSE (the 0/24 the original fine port produced) and gross divergence. Read the
verdicts accordingly.
"""
from __future__ import annotations
import json, sys, collections
from pathlib import Path

HERE = Path(__file__).parent
RUNS = HERE / "runs"
# task -> measured 95% noise band in points (None = not measured for this embodiment)
FLOOR = {"close_drawer": None, "pick_coke_can": 2.50}
ORDER = ["A_stock_native", "B_tgtplan_native", "C_tgt_native",
         "D_tgtplan_fine27", "E_tgt_fine27"]
LABEL = {"A_stock_native":   "registered (pose-rel + planner) @ 3 Hz   [REFERENCE]",
         "B_tgtplan_native":  "target + planner                @ 3 Hz",
         "C_tgt_native":      "target, no planner              @ 3 Hz",
         "D_tgtplan_fine27":  "target + planner                @ 27 Hz",
         "E_tgt_fine27":      "target, no planner              @ 27 Hz"}


def load(task, cfg, n):
    d = RUNS / f"{task}_{cfg}_n{n}"
    f = d / "summary.json"
    if not f.exists():
        return None
    s = json.load(open(f))
    out = {e["episode_id"]: bool(e["success"]) for e in s["episodes"]}
    return dict(n=s["n_episodes"], k=s["n_success"], vec=out,
                mode=s.get("control_mode_registered") or "?",
                act=s["actuation"], tick=s["tick_hz"], scale=s["delta_scale"])


def design_ci(vec):
    """95% CI on the rate, inflated by the design effect from config clustering.

    With one seed there is exactly one observation per config, so m=1 and DEFF=1 --
    the correction is a no-op here and the interval is the plain binomial one. It is
    kept so the number is computed the same way as everywhere else in this work, and
    so it tightens automatically if seeds are added.
    """
    import math
    n = len(vec); k = sum(vec.values())
    p = k / n if n else float("nan")
    se = math.sqrt(max(p * (1 - p), 1e-12) / n)
    return 100 * p, 100 * (p - 1.96 * se), 100 * (p + 1.96 * se)


N = int(sys.argv[1]) if len(sys.argv) > 1 else 24
for task in ("close_drawer", "pick_coke_can"):
    got = {c: load(task, c, N) for c in ORDER}
    got = {c: v for c, v in got.items() if v}
    if not got:
        continue
    ref = got.get("A_stock_native")
    floor = FLOOR.get(task)
    print(f"\n{'='*100}\n{task}   (n={N} episodes, one seed, ZERO latency)")
    print(f"  measured noise floor: "
          + (f"+/-{floor:.2f} pts" if floor else "NOT MEASURED for this task -- "
             "coke's is NOT transferable (different actuations per policy result)"))
    print(f"  {'config':52s} {'rate':>16s} {'agree vs ref':>13s}  verdict")
    for c in ORDER:
        v = got.get(c)
        if not v:
            continue
        r, lo, hi = design_ci(v["vec"])
        if ref is None or c == "A_stock_native":
            ag, verdict = "--", "reference"
        else:
            common = set(v["vec"]) & set(ref["vec"])
            same = sum(1 for e in common if v["vec"][e] == ref["vec"][e])
            ag = f"{same}/{len(common)}"
            d = r - 100 * ref["k"] / ref["n"]
            if v["k"] == 0:
                verdict = "COLLAPSE"
            elif floor is None:
                verdict = f"delta {d:+.1f} pts (no floor to judge against)"
            elif abs(d) <= floor:
                verdict = f"delta {d:+.1f} pts -- within the +/-{floor:.2f} floor"
            else:
                verdict = f"delta {d:+.1f} pts -- EXCEEDS the +/-{floor:.2f} floor"
        print(f"  {LABEL[c]:52s} {v['k']:2d}/{v['n']} = {r:5.1f}% {ag:>13s}  {verdict}")
    print("\n  Equivalence cannot be concluded from a single seed at this n; a matching "
          "rate with a\n  LOW episode-agreement means the two configs differ and happen "
          "to tie. Agreement is the\n  informative column.")
