#!/usr/bin/env python3
"""Rebuild the QRB5165 cost model from pooled in-situ measurements.

Why in-situ rather than a better standalone measurement:

The sweep showed cells predicted in-situ tile duration at 0.999x for tiles
>= 1 ms but 1.655x for tiles < 1 ms. Two standalone fixes were tried and
measured, not assumed:

  * a fixed per-dispatch offset      -- the size effect is real (r = -0.579
    between log predicted duration and log error), but the offset is not
    constant across lanes.
  * measuring with an idle gap before each execute, calibrated against
    in-situ (measurements/calibration_gap_sweep.json) -- this halves the
    error, 28.6% -> 18.7% median, but ~19% residual remains and two cells
    (fused_vision_conv@hta, mlp_control_full@cpu) stay 75-170% wrong under
    every setting. The obvious physical explanation, that cost tracks the
    idle gap the schedule leaves in front of each dispatch, was tested
    against the traces and REJECTED: evaluating each tile at its own median
    idle gap gives 31.0% error, worse than any flat choice.

The in-situ p50 pooled over the sweep's placements is by construction the
quantity the scheduler is trying to predict, and for accelerator lanes it is
stable because the lane model already serialises those tiles.

Deliberately NOT promoted:
  * cells with fewer than MIN_N placements -- too thin to pool.
  * multi-threaded CPU tiles flagged non-convergent in the measurements
    notes: their cost is a function of what runs beside them, so promoting
    an in-situ value changes the schedule, which changes the value. The
    `feedback` stage already demonstrated that loop for ViNT's decoder.

    python3 rebuild_cost_model.py [--write] [--min-n 5]
"""
import argparse, json, os, statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
MEAS = os.path.join(HERE, "measurements", "qrb5165_v66.json")
S = os.path.join(HERE, "sweeps", "qrb5165_20260829-200620")

# Tiles whose in-situ cost is schedule-dependent, per the measurements notes
# (`feedback_does_not_converge_for_contended_cpu_tiles`). Left alone.
NON_CONVERGENT = {("vint_decoder", "cpu"), ("vint_par_decoder", "cpu"),
                  ("fused_full_net", "cpu")}


def pooled(extra=()):
    agg = {}
    srcs = [os.path.join(S, "results.json"),
            os.path.join(S, "addendum_periodic_only", "results.json")]
    srcs += list(extra)
    for p in srcs:
        if not os.path.exists(p):
            continue
        for r in json.load(open(p)):
            if r.get("status") != "run":
                continue
            for k, v in (r.get("per_tile") or {}).items():
                job_tile = k.split("@")[0]
                lane = k.rsplit("@", 1)[1]
                net = job_tile.split("/", 1)[0]
                tile = job_tile.split("/", 1)[1]
                # job names carry an _a/_b/... copy suffix; the cost cell does not
                base = net.rstrip("_abcdefgh") if net[-2:-1] == "_" else net
                agg.setdefault((base, tile, lane), []).append(v["actual_p50_ms"])
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--min-n", type=int, default=5)
    ap.add_argument("--also", action="append", default=[],
                    help="additional results.json to pool in (repeatable). Use this "
                         "to iterate: promote, re-run, then promote again including "
                         "the new run, and check whether the cells settle.")
    a = ap.parse_args()

    meas = json.load(open(MEAS))
    cells = meas["cells"]
    agg = pooled(a.also)

    print(f"{'cell@lane':<40} {'n':>3} {'current':>9} {'in-situ p50':>12} "
          f"{'change':>9} {'action'}")
    promoted, skipped = [], []
    for (net, tile, lane), vals in sorted(agg.items()):
        key = f"{net}/{tile}"
        cur = cells.get(key, {}).get(lane)
        new_us = round(st.median(vals) * 1000.0, 1)
        n = len(vals)
        if (tile, lane) in NON_CONVERGENT:
            act, tag = "skip", "schedule-dependent"
        elif n < a.min_n:
            act, tag = "skip", f"n<{a.min_n}"
        elif cur is None:
            act, tag = "add", "new cell"
        else:
            act, tag = "promote", f"{100*(new_us-cur)/cur:+.0f}%"
        (promoted if act in ("promote", "add") else skipped).append(
            (key, lane, cur, new_us, n, tag))
        cf = f"{cur:9.1f}" if cur is not None else "       --"
        print(f"{key+'@'+lane:<40} {n:>3} {cf} {new_us:>12.1f} {tag:>9}  {act}")

    print(f"\n  {len(promoted)} cell(s) to update, {len(skipped)} left alone")
    if a.write:
        meas.setdefault("_previous_cells_standalone", json.loads(json.dumps(cells)))
        for key, lane, cur, new_us, n, tag in promoted:
            cells.setdefault(key, {})[lane] = new_us
        meas["statistic"] = "in_situ_p50_pooled"
        meas["captured_at"] = "2026-08-30"
        meas.setdefault("_notes", {})["cells_rebuilt_from_in_situ"] = (
            "Cells for accelerator and single-threaded tiles are the MEDIAN of the "
            "per-placement in-situ p50 recorded by sweep qrb5165_20260829-200620 "
            "(12 points) and its periodic-only addendum (6 points), pooled over up "
            "to 41 placements per cell and requiring at least "
            f"{a.min_n}. Standalone cells are kept in _previous_cells_standalone. "
            "Rationale: standalone loop-mean cells ran 1.655x optimistic for tiles "
            "under 1 ms. A calibrated idle-gap measurement "
            "(measurements/calibration_gap_sweep.json) halves that error but leaves "
            "~19% residual; the hypothesis that cost tracks the schedule's idle gap "
            "was tested against the traces and rejected (31% error, worse than any "
            "flat choice). Multi-threaded CPU tiles whose cost depends on what runs "
            "beside them are deliberately excluded -- see "
            "feedback_does_not_converge_for_contended_cpu_tiles.")
        with open(MEAS, "w") as f:
            json.dump(meas, f, indent=2)
        print(f"  wrote -> {MEAS}")
    else:
        print("  (dry run; pass --write)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
