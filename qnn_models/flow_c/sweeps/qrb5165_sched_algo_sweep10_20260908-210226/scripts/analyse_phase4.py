#!/usr/bin/env python3
"""Phase 5 — does the reference's predicted solver ranking survive measurement?

Two comparisons, kept apart on purpose:

  NON-PERIODIC makespan   `evaluate(..., restrict_to_nonperiodic=True)` is what
                          sweep10 ranks on, so this is the comparison its
                          headline claim is about. Measured from the trace as
                          the last end time of any non-periodic network's ops.
  ALL-OPERATIONS makespan the board's wall clock. Usually pinned by the last
                          periodic instance, so solvers differ far less here —
                          reporting only this would flatten the ranking by
                          construction and say nothing.

Everything is held against the measured NOISE FLOOR: the per-point rep spread,
reported as a distribution rather than assumed. A predicted difference smaller
than the spread of the points it separates is reported as unresolved, not as a
result.

    python3 analyse_phase4.py [--dest ../results] [--plots]
"""
from __future__ import annotations

import argparse, collections, json, math, os, statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
SWEEP = os.path.abspath(os.path.join(HERE, ".."))
RES = os.path.join(SWEEP, "results")
TEN = ["greedy", "greedy_periodic", "greedy_reserved", "decomposed",
       "heft", "heft_edf", "pso", "sa", "cpsat", "cpsat:warm"]
EXTRA = ["best-of-fast", "cpsat:warmbest"]
SOLVERS = TEN + EXTRA
#: the reference's own headline, for the side-by-side
REF_MEAN_IMPR = {"cpsat:warmbest": 9.75, "pso": 9.10, "sa": 8.75,
                 "cpsat:warm": 8.30, "best-of-fast": 7.63, "cpsat": 6.03,
                 "heft_edf": 2.37, "greedy": 0.0, "decomposed": -4.92,
                 "greedy_reserved": -0.10, "greedy_periodic": 1.04,
                 "heft": -1.66}


def load(name, default=None):
    p = os.path.join(RES, name)
    return json.load(open(p)) if os.path.exists(p) else default


def famcfg(wl):
    b = wl[len("networks_"):] if wl.startswith("networks_") else wl
    fam, _, cfg = b.rpartition("_")
    return fam, cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default=RES)
    ap.add_argument("--plots", action="store_true")
    a = ap.parse_args()
    p4 = load("phase4_results.json", [])
    p3 = load("phase3_all_results.json") or load("all_results.json") or []
    pred = {(r["workload"], r["solver"]): r for r in p3}

    ran = [r for r in p4 if r.get("measured_np_median_ms")]
    if not ran:
        print("no measured points yet")
        return 1

    # ---------------- noise floor
    spreads_np, spreads_all, reps_n = [], [], []
    for r in ran:
        m, s = r.get("measured_np_median_ms"), r.get("measured_np_spread_ms")
        if m and s is not None:
            spreads_np.append(s / m * 100)
        m, s = r.get("measured_median_ms"), r.get("measured_spread_ms")
        if m and s is not None:
            spreads_all.append(s / m * 100)
        reps_n.append(len(r.get("measured_np_reps_ms") or []))
    noise = dict(
        n_points=len(ran), reps_per_point=sorted(set(reps_n)),
        np_spread_pct=dict(median=round(st.median(spreads_np), 3),
                           p90=round(sorted(spreads_np)[int(.9 * (len(spreads_np) - 1))], 3),
                           max=round(max(spreads_np), 3)) if spreads_np else None,
        all_spread_pct=dict(median=round(st.median(spreads_all), 3),
                            max=round(max(spreads_all), 3)) if spreads_all else None)

    # ---------------- per-cell tables and rank comparison
    cells = collections.defaultdict(dict)
    for r in ran:
        cells[r["workload"]][r["solver"]] = r
    per_cell, rank_rows = [], []
    for wl in sorted(cells):
        m = cells[wl]
        if "greedy" not in m:
            continue
        gP = pred.get((wl, "greedy"), {}).get("objective")
        gM = m["greedy"]["measured_np_median_ms"]
        rows = []
        for s, r in m.items():
            P = pred.get((wl, s), {}).get("objective") or r.get("objective")
            M = r["measured_np_median_ms"]
            rows.append(dict(
                solver=s, predicted_ms=P, measured_ms=M,
                spread_ms=r.get("measured_np_spread_ms"),
                ratio=round(M / P, 4) if P else None,
                pred_impr_pct=round((gP - P) / gP * 100, 3) if gP else None,
                meas_impr_pct=round((gM - M) / gM * 100, 3) if gM else None,
                misses=r.get("misses"),
                measured_all_ms=r.get("measured_median_ms"),
                predicted_all_ms=r.get("table_predicted_makespan_ms"),
                duplicate_of=r.get("duplicate_of")))
        fam, cfg = famcfg(wl)
        per_cell.append(dict(workload=wl, family=fam, config=cfg,
                             n_solvers=len(rows), rows=sorted(
                                 rows, key=lambda x: x["predicted_ms"] or math.inf)))
        # rank agreement over the solvers measured in this cell
        have = [r for r in rows if r["predicted_ms"] and r["measured_ms"]]
        if len(have) >= 3:
            pr = {r["solver"]: i for i, r in enumerate(
                sorted(have, key=lambda x: x["predicted_ms"]))}
            mr = {r["solver"]: i for i, r in enumerate(
                sorted(have, key=lambda x: x["measured_ms"]))}
            n = len(have)
            conc = dis = 0
            ss = list(pr)
            for i in range(n):
                for j in range(i + 1, n):
                    a_, b_ = ss[i], ss[j]
                    sgn = (pr[a_] - pr[b_]) * (mr[a_] - mr[b_])
                    conc += sgn > 0
                    dis += sgn < 0
            tau = (conc - dis) / (conc + dis) if (conc + dis) else None
            # how many of the discordant pairs are inside the noise floor
            unresolved = 0
            for i in range(n):
                for j in range(i + 1, n):
                    a_, b_ = have[i], have[j]
                    d = abs(a_["measured_ms"] - b_["measured_ms"])
                    sp = max(a_["spread_ms"] or 0, b_["spread_ms"] or 0)
                    if d <= sp:
                        unresolved += 1
            rank_rows.append(dict(workload=wl, family=fam, config=cfg, n=n,
                                  kendall_tau=round(tau, 4) if tau is not None else None,
                                  concordant=conc, discordant=dis,
                                  pairs_inside_noise=unresolved,
                                  total_pairs=conc + dis))

    # ---------------- per-solver aggregate (measured), feasible-first
    agg = {}
    for s in SOLVERS:
        impr, cnt = [], 0
        for c in per_cell:
            row = next((r for r in c["rows"] if r["solver"] == s), None)
            g = next((r for r in c["rows"] if r["solver"] == "greedy"), None)
            if not row or not g or row["meas_impr_pct"] is None:
                continue
            if (row.get("misses") or 0) > 0:
                continue
            impr.append(row["meas_impr_pct"]); cnt += 1
        if impr:
            agg[s] = dict(
                cells=cnt, mean_impr_pct=round(st.mean(impr), 3),
                median_impr_pct=round(st.median(impr), 3),
                worst_pct=round(min(impr), 3), best_pct=round(max(impr), 3),
                reference_mean_impr_pct=REF_MEAN_IMPR.get(s))
    predagg = {}
    for s in SOLVERS:
        impr = []
        for c in per_cell:
            row = next((r for r in c["rows"] if r["solver"] == s), None)
            if row and row["pred_impr_pct"] is not None and (row.get("misses") or 0) == 0:
                impr.append(row["pred_impr_pct"])
        if impr:
            predagg[s] = dict(cells=len(impr),
                              mean_impr_pct=round(st.mean(impr), 3),
                              median_impr_pct=round(st.median(impr), 3))

    ratios = [r["ratio"] for c in per_cell for r in c["rows"] if r["ratio"]]
    out = dict(noise_floor=noise,
               cell_ratio_measured_over_predicted=dict(
                   n=len(ratios), median=round(st.median(ratios), 4),
                   min=round(min(ratios), 4), max=round(max(ratios), 4)) if ratios else None,
               per_solver_measured=agg, per_solver_predicted=predagg,
               rank_agreement=rank_rows, per_cell=per_cell)
    os.makedirs(a.dest, exist_ok=True)
    json.dump(out, open(os.path.join(a.dest, "phase5_analysis.json"), "w"), indent=1)

    print(f"noise floor: {noise['n_points']} measured points, "
          f"{noise['reps_per_point']} reps each")
    if noise["np_spread_pct"]:
        print(f"  non-periodic makespan rep spread: median "
              f"{noise['np_spread_pct']['median']}%  p90 "
              f"{noise['np_spread_pct']['p90']}%  max {noise['np_spread_pct']['max']}%")
    if noise["all_spread_pct"]:
        print(f"  wall-clock rep spread:            median "
              f"{noise['all_spread_pct']['median']}%  max "
              f"{noise['all_spread_pct']['max']}%")
    if ratios:
        print(f"  measured/predicted (non-periodic): median "
              f"{st.median(ratios):.3f}x  min {min(ratios):.3f}  max {max(ratios):.3f}")
    print(f"\n{'solver':<18}{'cells':>6}{'measured mean':>15}{'median':>9}"
          f"{'worst':>9}{'predicted mean':>16}{'reference':>11}")
    for s in sorted(agg, key=lambda x: -agg[x]["mean_impr_pct"]):
        v, pv = agg[s], predagg.get(s, {})
        print(f"{s:<18}{v['cells']:>6}{v['mean_impr_pct']:>14.2f}%"
              f"{v['median_impr_pct']:>8.2f}%{v['worst_pct']:>8.2f}%"
              f"{pv.get('mean_impr_pct', float('nan')):>15.2f}%"
              f"{(v['reference_mean_impr_pct'] if v['reference_mean_impr_pct'] is not None else float('nan')):>10.2f}%")
    if rank_rows:
        taus = [r["kendall_tau"] for r in rank_rows if r["kendall_tau"] is not None]
        ins = sum(r["pairs_inside_noise"] for r in rank_rows)
        tot = sum(r["total_pairs"] for r in rank_rows)
        print(f"\nrank agreement over {len(rank_rows)} cell(s): Kendall tau "
              f"median {st.median(taus):+.3f}, min {min(taus):+.3f}, max {max(taus):+.3f}")
        print(f"  {ins}/{tot} solver pairs are separated by less than the rep "
              f"spread of the two points, i.e. unresolved by measurement")
    print(f"\n-> {os.path.join(a.dest, 'phase5_analysis.json')}")
    if a.plots:
        make_plots(out, os.path.join(SWEEP, "plots"))
    return 0


def make_plots(out, dest):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    os.makedirs(dest, exist_ok=True)

    # 1. predicted vs measured improvement over greedy, per (cell, solver)
    xs, ys, cs = [], [], []
    for c in out["per_cell"]:
        for r in c["rows"]:
            if r["pred_impr_pct"] is None or r["meas_impr_pct"] is None:
                continue
            xs.append(r["pred_impr_pct"]); ys.append(r["meas_impr_pct"])
            cs.append(SOLVERS.index(r["solver"]) if r["solver"] in SOLVERS else -1)
    if xs:
        fig, ax = plt.subplots(figsize=(7.5, 7))
        lim = max(abs(min(xs + ys)), abs(max(xs + ys))) * 1.1 + 1
        ax.plot([-lim, lim], [-lim, lim], "k--", lw=.8, label="perfect agreement")
        nf = out["noise_floor"].get("np_spread_pct") or {}
        if nf.get("median"):
            ax.fill_between([-lim, lim], [-lim - nf["median"], lim - nf["median"]],
                            [-lim + nf["median"], lim + nf["median"]],
                            color="0.85", zorder=0,
                            label=f"median rep spread ±{nf['median']:.2f}%")
        sc = ax.scatter(xs, ys, c=cs, cmap="tab20", s=42, edgecolor="k", lw=.4)
        ax.set_xlabel("PREDICTED improvement over greedy, %  (cost model)")
        ax.set_ylabel("MEASURED improvement over greedy, %  (board, median of reps)")
        ax.set_title("Does the predicted ranking survive measurement?\n"
                     "non-periodic makespan, one point per (cell, solver)")
        ax.axhline(0, color="0.4", lw=.6); ax.axvline(0, color="0.4", lw=.6)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(alpha=.3)
        fig.tight_layout(); fig.savefig(f"{dest}/predicted_vs_measured.png", dpi=120)
        plt.close(fig)

    # 2. per-solver mean improvement: this board vs the reference
    agg = out["per_solver_measured"]
    order = sorted(agg, key=lambda s: -agg[s]["mean_impr_pct"])
    if order:
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(order))
        ax.bar(x - .27, [agg[s]["mean_impr_pct"] for s in order], .27,
               label="measured here (QRB5165)")
        ax.bar(x, [out["per_solver_predicted"].get(s, {}).get("mean_impr_pct", 0)
                   for s in order], .27, label="predicted here (same cost model)")
        ax.bar(x + .27, [agg[s]["reference_mean_impr_pct"] or 0 for s in order], .27,
               label="reference (FireSim, predicted only)")
        ax.set_xticks(x); ax.set_xticklabels(order, rotation=60, fontsize=8, ha="right")
        ax.axhline(0, color="k", lw=.8)
        ax.set_ylabel("mean improvement over greedy, %")
        ax.set_title("Solver ranking: measured on silicon vs predicted, "
                     "against the reference's prediction")
        ax.legend(fontsize=8); ax.grid(axis="y", alpha=.3)
        fig.tight_layout(); fig.savefig(f"{dest}/ranking_vs_reference.png", dpi=120)
        plt.close(fig)

    # 3. rep spread distribution -- the noise floor, shown not asserted
    sp = []
    for c in out["per_cell"]:
        for r in c["rows"]:
            if r.get("spread_ms") and r.get("measured_ms"):
                sp.append(r["spread_ms"] / r["measured_ms"] * 100)
    if sp:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(sp, bins=24, color="#4477aa", edgecolor="k", lw=.4)
        ax.axvline(st.median(sp), color="crimson", lw=1.5,
                   label=f"median {st.median(sp):.2f}%")
        ax.set_xlabel("rep spread as % of the point's median (non-periodic makespan)")
        ax.set_ylabel("points")
        ax.set_title("The noise floor every difference in this sweep is held against")
        ax.legend(); ax.grid(alpha=.3)
        fig.tight_layout(); fig.savefig(f"{dest}/noise_floor.png", dpi=120)
        plt.close(fig)
    print(f"  plots -> {dest}")


if __name__ == "__main__":
    raise SystemExit(main())
