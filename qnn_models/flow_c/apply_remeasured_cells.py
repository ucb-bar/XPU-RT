#!/usr/bin/env python3
"""Fold the two-phase re-measurement into the cost model, and show the effect.

Compares four numbers per cell:
    current      what measurements/qrb5165_v66.json holds today (loop mean)
    loop_p50     this re-measurement, old methodology
    gap_p50      this re-measurement, in-situ invocation methodology
    in_situ_p50  what the sweep actually observed, pooled over its placements

The sweep is the ground truth we are trying to predict, so the test of the new
methodology is simply: does gap_p50 land closer to in_situ_p50 than the current
cell does? Contended cells are reported but NOT judged on this -- contention is
the scheduler's job to model, not the cell's, and baking it into a cell is what
made the per-run `feedback` stage non-convergent.

    python3 apply_remeasured_cells.py [--write] [--gap-us-file FILE]
"""
import argparse, json, os, statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
MEAS = os.path.join(HERE, "measurements", "qrb5165_v66.json")
SWEEP = os.path.join(HERE, "sweeps", "qrb5165_20260829-200620")


def in_situ():
    """Pooled per-(cell, backend) in-situ p50 from the sweep + addendum."""
    agg = {}
    for rel in ("results.json", os.path.join("addendum_periodic_only", "results.json")):
        p = os.path.join(SWEEP, rel)
        if not os.path.exists(p):
            continue
        for r in json.load(open(p)):
            if r.get("status") != "run":
                continue
            for k, v in (r.get("per_tile") or {}).items():
                # "job/tile@lane" -> strip the job instance suffix
                tile_lane = k.split("/", 1)[1] if "/" in k else k
                tile, lane = tile_lane.rsplit("@", 1)
                agg.setdefault((tile, lane), []).append(v["actual_p50_ms"])
    return {k: st.median(v) for k, v in agg.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="update the cost model")
    ap.add_argument("--remeasured",
                    default=os.path.join(HERE, "measurements", "qrb5165_v66_remeasured.json"))
    a = ap.parse_args()

    rm = json.load(open(a.remeasured))
    meas = json.load(open(MEAS))
    cells = meas["cells"]
    situ = in_situ()

    rows = []
    for r in rm["results"]:
        if r.get("status") != "ok":
            continue
        cell, be = r["cell"], r["backend"]
        tile = cell.split("/", 1)[1]
        cur = cells.get(cell, {}).get(be)
        rows.append({
            "cell": cell, "be": be, "tile": tile,
            "cur": cur,
            "loop": r["median_us"], "gap": r["gap_median_us"],
            "delta": r["gap_minus_loop_us"],
            "situ": situ.get((tile, be), None),
        })

    print(f"{'cell@backend':<40} {'current':>9} {'loop p50':>9} {'gap p50':>9} "
          f"{'delta':>8} {'in-situ':>9} {'cur err':>8} {'gap err':>8}")
    better = worse = 0
    cur_errs, gap_errs = [], []
    for r in sorted(rows, key=lambda x: (x["gap"] or 0)):
        s = r["situ"] * 1000 if r["situ"] else None
        ce = abs(r["cur"] - s) / s if (s and r["cur"]) else None
        ge = abs(r["gap"] - s) / s if s else None
        if ce is not None and ge is not None:
            cur_errs.append(ce); gap_errs.append(ge)
            better += ge < ce; worse += ge > ce
        f = lambda v, w=9, p=1: f"{v:{w}.{p}f}" if v is not None else " " * (w - 2) + "--"
        pct = lambda v: f"{100*v:7.1f}%" if v is not None else "      --"
        print(f"{r['cell']+'@'+r['be']:<40} {f(r['cur'])} {f(r['loop'])} {f(r['gap'])} "
              f"{f(r['delta'],8)} {f(s)} {pct(ce)} {pct(ge)}")

    print(f"\n  cells with an in-situ reference: {len(cur_errs)}")
    if cur_errs:
        print(f"  median |error| vs in-situ   current {100*st.median(cur_errs):5.1f}%"
              f"   ->  gap {100*st.median(gap_errs):5.1f}%")
        print(f"  cells improved {better}, worsened {worse}")
    d = [r["delta"] for r in rows]
    small = [r["delta"] for r in rows if r["gap"] < 1000]
    print(f"\n  gap - loop, all cells      median {st.median(d):+8.1f} us")
    if small:
        print(f"  gap - loop, cells < 1 ms   median {st.median(small):+8.1f} us  "
              f"(n={len(small)})   <- the fixed per-invocation cost")

    if a.write:
        meas.setdefault("_previous_cells_loop_mean", json.loads(json.dumps(cells)))
        for r in rows:
            cells.setdefault(r["cell"], {})[r["be"]] = round(r["gap"], 1)
        meas["statistic"] = "gap_median"
        meas["iters"] = rm["iters"]
        meas["captured_at"] = "2026-08-30"
        meas["_comment"] = (
            "Per-(binding, backend) whole-graph wallclock around QnnGraph_execute, "
            "captured by qnn_models/runtime/profile_segments.cpp on the physical "
            "QRB5165 at 10.44.120.201. Cells are the GAP-PHASE median: each timed "
            "execute is preceded by an idle gap so the measurement matches how the "
            "scheduled runtime invokes a tile -- once per period, from a lane thread "
            "that was asleep until its gate fired. The previous cells (mean of 50 "
            "back-to-back executes) are kept in _previous_cells_loop_mean; they ran "
            "1.65x optimistic for tiles under 1 ms because a tight loop amortises "
            "away a per-invocation cost the runtime pays every time.")
        meas.setdefault("_notes", []).append(
            "cells_are_gap_phase: sweep qrb5165_20260829-200620 showed loop-mean "
            "cells predicted in-situ at 0.999x for tiles >= 1 ms but 1.655x for "
            "tiles < 1 ms, a near-fixed +0.234 ms miss. Cells are now measured with "
            "an idle gap before each execute. Contention is deliberately NOT in the "
            "cells -- the scheduler models lane exclusion, and baking contention "
            "into a cell is what made the per-run feedback stage non-convergent.")
        with open(MEAS, "w") as f:
            json.dump(meas, f, indent=2)
        print(f"\n  wrote {len(rows)} cells -> {MEAS}")
    else:
        print("\n  (dry run; pass --write to update the cost model)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
