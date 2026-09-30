#!/usr/bin/env python3
"""The paired flight census behind one showdown figure: two arms, the same (cruise, seed) cells.

Same rule as docs/Evaluation/showdown_rate_sweep_reproduction.md §3 and scripts/figure_candidates.py:
rows come only through scripts/flight_quarantine.py (simulator-fault batches dropped), a cell counts
only when BOTH arms flew it, completions are counted over the non-timeout flights (a timeout is a
flight that ran out of steps, censored rather than scored), gates are 4 for a completion and the
recorded gates_passed otherwise, the completion difference carries a Newcombe (Wilson) 95 % interval
and the gate difference a percentile bootstrap over the PAIRED per-cell differences.

    scripts/census_submitted.py [--csv <campaign.csv>] [--xpu <trace>] [--ros <trace>]
                               [--resamples 4000] [--seed 11] [--json <out>]
"""
from __future__ import annotations
import argparse, collections, json, os, statistics as st, sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from flight_quarantine import flight_rows          # noqa: E402
from showdown_atlas import newcombe                # noqa: E402
from figure_candidates import boot_gate_diff, gates  # noqa: E402

RES = os.path.join(REPO, "results", "codesign_feedback")


def by_cell(rows, trace):
    out = {}
    for r in rows:
        if os.path.basename(r["ctrl_trace"]) != trace:
            continue
        k = (r["cruise_speed"], r["seed"])
        if k in out:
            raise SystemExit(f"census: {trace} flew cell {k} twice; decide which rows are the flight")
        out[k] = r
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=os.path.join(RES, "campaign_submitted", "campaign.csv"))
    ap.add_argument("--xpu", default="xpu_a_cpsat_hard.csv")
    ap.add_argument("--ros", default="ros_vanilla_c5045.csv")
    ap.add_argument("--resamples", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    rows = flight_rows(a.csv)
    X, R = by_cell(rows, a.xpu), by_cell(rows, a.ros)
    common = sorted(set(X) & set(R), key=lambda k: (float(k[0]), int(k[1])))
    if not common:
        raise SystemExit("census: the two arms share no (cruise, seed) cell")

    nx = sum(X[k]["outcome"] != "timeout" for k in common)
    kx = sum(X[k]["outcome"] == "success" for k in common)
    nr = sum(R[k]["outcome"] != "timeout" for k in common)
    kr = sum(R[k]["outcome"] == "success" for k in common)
    gx = [gates(X[k]) for k in common]
    gr = [gates(R[k]) for k in common]
    d, lo, hi = newcombe(kx, nx, kr, nr)
    gd, glo, ghi = boot_gate_diff(list(zip(gx, gr)), n=a.resamples, seed=a.seed)
    more = sum(x > r for x, r in zip(gx, gr))
    tie = sum(x == r for x, r in zip(gx, gr))
    fewer = sum(x < r for x, r in zip(gx, gr))

    print(f"{a.csv}  {len(common)} paired cells "
          f"({len(set(k[0] for k in common))} cruise speeds x {len(set(k[1] for k in common))} seeds)")
    print(f"{'arm':<26}{'completed':>12}{'non-timeout':>13}{'of':>5}{'mean gates':>12}")
    for name, k_, n_, g in ((a.xpu, kx, nx, gx), (a.ros, kr, nr, gr)):
        print(f"{name:<26}{k_:>12}{n_:>13}{len(common):>5}{st.fmean(g):>12.2f}")
    print(f"completion difference  {d*100:+.1f} pts  Newcombe 95 % [{lo*100:+.1f}, {hi*100:+.1f}]")
    print(f"gate difference        {gd:+.2f}       bootstrap 95 % [{glo:+.2f}, {ghi:+.2f}] "
          f"({a.resamples} resamples, seed {a.seed}, paired per-cell)")
    print(f"per-cell gates: XPU-RT more {more}, tie {tie}, fewer {fewer}")
    print()
    print(f"{'cruise':>7}  {'XPU k/n':>10} {'ROS k/n':>10}  {'XPU gates':>10} {'ROS gates':>10}")
    for c in sorted({k[0] for k in common}, key=float):
        cc = [k for k in common if k[0] == c]
        print(f"{c:>7}  {sum(X[k]['outcome']=='success' for k in cc):>4}/{sum(X[k]['outcome']!='timeout' for k in cc):<5}"
              f"{sum(R[k]['outcome']=='success' for k in cc):>5}/{sum(R[k]['outcome']!='timeout' for k in cc):<5} "
              f"{st.fmean(gates(X[k]) for k in cc):>10.2f} {st.fmean(gates(R[k]) for k in cc):>10.2f}")

    out = {"csv": os.path.relpath(a.csv, REPO), "xpu_trace": a.xpu, "ros_trace": a.ros, "pairs": len(common),
           "xpu_success": [kx, nx], "ros_success": [kr, nr],
           "xpu_mean_gates": round(st.fmean(gx), 3), "ros_mean_gates": round(st.fmean(gr), 3),
           "separation_pts": [round(d * 100, 1), round(lo * 100, 1), round(hi * 100, 1)],
           "gate_diff": [round(gd, 3), round(glo, 3), round(ghi, 3)],
           "bootstrap": {"resamples": a.resamples, "seed": a.seed},
           "per_cell_gates": {"xpu_more": more, "tie": tie, "xpu_fewer": fewer}}
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)
        print(f"\n-> {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
