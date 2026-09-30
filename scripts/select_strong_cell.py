#!/usr/bin/env python3
"""The strongest displayable cell across every campaign on disk: for each (campaign, scene, arm pair,
gain policy, speed) with 12 seeds per arm, the cell where XPU-RT completes the most flights while the
baseline completes at most 1/12 and crashes after entering the course (1-2 gates) in at least half.
Ties go to the faster speed, then the lower latency replay. The same rule as campaign_select_v2.py,
scanned over all campaign CSVs (campaign_v2, campaign_env, campaign_percep and the follow-ups) so the
display is chosen where XPU-RT is strong, not only where the baseline is weakest.
    scripts/select_strong_cell.py [--min-xpu 3] [--xpu-arms xpu_a_cpsat_hard.csv,...] [--ros-arms ros_vanilla445.csv,...]
"""
from __future__ import annotations
import argparse, collections, csv, glob, os
from flight_quarantine import flight_rows  # drops simulator-fault batches (results/codesign_feedback/flight_quarantine.csv)
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(REPO, "results", "codesign_feedback")
CSVS = ["campaign_v2/campaign_v2.csv", "campaign_v2_courseB/campaign_v2.csv", "campaign_env/env_sweep.csv"] + \
       sorted(os.path.relpath(p, RES) for p in glob.glob(os.path.join(RES, "campaign_*", "campaign.csv")))


def scene(r):
    return (r.get("course", "a") or "a", f"{float(r.get('prop_density', 0.3) or 0.3):.2f}", r.get("person_h", "1.7") or "1.7",
            f"{float(r.get('walk_speed', 0) or 0):.1f}", r.get("walk_cross", "0") or "0")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-xpu", type=int, default=3)
    ap.add_argument("--xpu-arms", default="xpu_a_cpsat_hard.csv,xpu_a90_cpsat.csv,xpu_a120h_cpsat.csv,xpu_b5_cpsat.csv")
    ap.add_argument("--ros-arms", default="ros_vanilla445.csv,ros_vanilla4t45.csv,ros_vanilla4_q145.csv,ros_rvanilla445.csv,ros_rvanilla490.csv")
    a = ap.parse_args()
    xa, ra = a.xpu_arms.split(","), a.ros_arms.split(",")
    cells = collections.defaultdict(dict)   # (csv, scene, trace, lat, hold, gain, cruise) -> seed -> row
    for rel in CSVS:
        p = os.path.join(RES, rel)
        if not os.path.exists(p):
            continue
        for r in flight_rows(p):
            k = (rel, scene(r), r["ctrl_trace"], f"{float(r.get('percep_latency_ms', 0) or 0):g}", f"{float(r.get('percep_hold_ms', 0) or 0):g}",
                 f"{float(r['moment_scale']):g}", round(float(r["cruise_speed"]), 3))
            cells[k].setdefault(r["seed"], r)
    def st(rows):
        succ = sum(r["outcome"] == "success" for r in rows)
        ent = sum(1 for r in rows if r["outcome"] == "crash" and 1 <= int(float(r["gates_passed"])) <= 2)
        return len(rows), succ, ent
    table = []
    for (rel, sc, tr, lat, hold, gain, cru), rows in cells.items():
        if tr not in xa:
            continue
        n, succ, _ = st(list(rows.values()))
        if n < 12 or succ < a.min_xpu:
            continue
        for rtr in ra:
            for (rel2, sc2, tr2, lat2, hold2, gain2, cru2), rows2 in cells.items():
                if rel2 != rel or sc2 != sc or tr2 != rtr or cru2 != cru:
                    continue
                fixed = abs(float(gain) - 0.0055) < 1e-9 and abs(float(gain2) - 0.0055) < 1e-9
                cal = abs(float(gain) - 0.0055) > 1e-9 and abs(float(gain2) - 0.0055) > 1e-9
                if not (fixed or cal):
                    continue
                n2, s2, e2 = st(list(rows2.values()))
                if n2 < 12:
                    continue
                ok = s2 <= 1 and e2 >= n2 // 2
                table.append((succ, cru, -float(lat), rel, sc, tr, lat, hold, gain, rtr, lat2, hold2, gain2, n, n2, s2, e2, ok))
    table.sort(reverse=True)
    print(f"{'XPU':>5} {'cruise':>6} {'campaign':<34} scene(course,dens,people,walk,cross)  xpu arm / lat / hold / gain    ros arm / lat / hold / gain   ROS k/n entered")
    for t in table[:25]:
        succ, cru, _, rel, sc, tr, lat, hold, gain, rtr, lat2, hold2, gain2, n, n2, s2, e2, ok = t
        print(f"{succ:>2}/{n:<2} {cru:>6} {rel:<34} {str(sc):<36} {tr}/{lat}/{hold}/{gain}   {rtr}/{lat2}/{hold2}/{gain2}   {s2}/{n2} {e2:>2}{'  <- qualifies' if ok else ''}")
    best = next((t for t in table if t[-1]), None)
    if best:
        print(f"\nSTRONGEST: XPU-RT {best[0]}/{best[13]} at {best[1]} m/s in {best[3]} scene {best[4]} ({best[5]} lat {best[6]} hold {best[7]} gain {best[8]}) vs {best[9]} (lat {best[10]} hold {best[11]} gain {best[12]}) {best[15]}/{best[14]}, entered-then-crashed {best[16]}")
    else:
        print("\nno qualifying cell")


if __name__ == "__main__":
    main()
