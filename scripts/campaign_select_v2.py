#!/usr/bin/env python3
"""Which cell of the trace-driven campaign the showdown figure displays — a rule written before
the runs (results/codesign_feedback/campaign_v2/campaign_v2.csv, scripts/campaign_v2.sh).

Arms are named by the control-output trace they replay (`ctrl_trace` column). The figure shows
one XPU-RT arm and one ROS 2 arm at one cruise speed:

  * every cell has the same gain (moment_scale) for both arms and 12 seeds each;
  * display the FASTEST cruise speed at which the XPU-RT arm completes the course at least
    3/12 and the ROS 2 arm crashes after entering the course (1 or 2 gates cleared) in at least
    half of its flights and completes at most 1/12;
  * ties (none expected) go to the slower speed.

Displayed flights inside the cell: XPU-RT = the first success, re-dumped with
--post_success_steps so the fourth-gate crossing is in frame; ROS 2 = a crash after exactly two
gates if the cell has one (else one gate), the one that got furthest in time, a collision
("clip") preferred over a ground strike.

    scripts/campaign_select_v2.py --xpu xpu_a_cpsat_hard.csv --ros ros_vanilla445.csv [--json]
"""
from __future__ import annotations
import argparse, collections, csv, json, os, sys
from flight_quarantine import flight_rows  # drops simulator-fault batches (results/codesign_feedback/flight_quarantine.csv)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(REPO, "results/codesign_feedback/campaign_v2/campaign_v2.csv")


def load(path):
    """Cells keyed by (trace, gain, cruise); a seed flown twice in a cell (two drivers meeting on it)
    counts once, the first flight."""
    cells = collections.defaultdict(dict)
    for r in flight_rows(path):
        k = (r["ctrl_trace"], f"{float(r['moment_scale']):g}", round(float(r["cruise_speed"]), 3))
        cells[k].setdefault(r["seed"], r)
    return {k: list(v.values()) for k, v in cells.items()}


def stats(rows):
    succ = sum(r["outcome"] == "success" for r in rows)
    entered = [r for r in rows if r["outcome"] == "crash" and 1 <= int(float(r["gates_passed"])) <= 2]
    two = [r for r in entered if int(float(r["gates_passed"])) == 2] or entered
    two.sort(key=lambda r: (r.get("crash_type") != "clip", -int(r["steps"])))
    return {"n": len(rows), "success": succ, "entered_then_crashed": len(entered),
            "gates": dict(sorted(collections.Counter(int(float(r["gates_passed"])) for r in rows).items())),
            "display_seed": two[0]["seed"] if two else None,
            "first_success_seed": next((r["seed"] for r in rows if r["outcome"] == "success"), None)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xpu", required=True, help="basename of the XPU-RT arm's trace"); ap.add_argument("--ros", required=True)
    ap.add_argument("--csv", default=CSV); ap.add_argument("--json", action="store_true")
    ap.add_argument("--policy", choices=["fixed", "calibrated", "any"], default="any",
                    help="fixed = every arm at moment_scale 0.0055; calibrated = each arm at 0.5/its replayed rate; any = both, fastest cell wins")
    a = ap.parse_args()
    cells = load(a.csv)
    speeds = sorted({k[2] for k in cells})
    gains = sorted({k[1] for k in cells})
    table = []
    for cru in speeds:
        # a cell pairs the two arms at the same gain (fixed) or each at its own calibrated gain
        pairs = []
        for gx in gains:
            for gr in gains:
                fixed = abs(float(gx) - 0.0055) < 1e-9 and abs(float(gr) - 0.0055) < 1e-9
                cal = (not fixed) and abs(float(gx) - 0.0055) > 1e-9 and abs(float(gr) - 0.0055) > 1e-9
                if (a.policy == "fixed" and not fixed) or (a.policy == "calibrated" and not cal) or not (fixed or cal):
                    continue
                pairs.append((gx, gr, "fixed" if fixed else "calibrated"))
        for gx, gr, pol in pairs:
            g = f"{pol}:{gx}/{gr}"
            x = cells.get((a.xpu, gx, cru)); r = cells.get((a.ros, gr, cru))
            if not x or not r:
                continue
            xs, rs = stats(x), stats(r)
            ok = (xs["n"] >= 12 and rs["n"] >= 12 and xs["success"] >= 3 and rs["success"] <= 1
                  and rs["entered_then_crashed"] >= rs["n"] // 2)
            table.append({"gain": g, "cruise": cru, "xpu": xs, "ros": rs, "qualifies": ok})
    choice = max((t for t in table if t["qualifies"]), key=lambda t: t["cruise"], default=None)
    if a.json:
        print(json.dumps({"choice": choice, "table": table}, indent=2)); return 0
    for t in table:
        print(f"gain {t['gain']:<7} cruise {t['cruise']:<4} XPU-RT {t['xpu']['success']:>2}/{t['xpu']['n']:<2} gates {t['xpu']['gates']}   "
              f"ROS {t['ros']['success']:>2}/{t['ros']['n']:<2} entered-then-crashed {t['ros']['entered_then_crashed']:>2} gates {t['ros']['gates']}"
              f"{'   <- qualifies' if t['qualifies'] else ''}")
    if choice:
        print(f"\nDISPLAY: gain {choice['gain']} cruise {choice['cruise']} -- XPU-RT first success seed {choice['xpu']['first_success_seed']}, "
              f"ROS display seed {choice['ros']['display_seed']}")
    else:
        print("\nno cell qualifies yet")
    return 0


if __name__ == "__main__":
    sys.exit(main())
