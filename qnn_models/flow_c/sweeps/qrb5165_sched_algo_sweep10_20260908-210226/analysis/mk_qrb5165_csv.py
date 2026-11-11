#!/usr/bin/env python3
"""Emit this sweep's results in the schema `plot_pareto_panels.py` reads.

Port of `RoSE/experiments/xpurt_rerun/analysis/mk_fpga_csv.py` onto the
QRB5165 sweep's own records. Two CSVs, because this sweep has something the
FPGA rerun's schedule-side CSV does not: the schedules were actually executed.

  results.csv           PREDICTED. `objective` is the solver's own value, from
                        results/all_results.json. 504 rows = 42 cells x 12
                        solvers, the direct analogue of the reference's
                        schedule-side CSV.
  measured/results.csv  MEASURED. `objective` is the measured aperiodic
                        makespan (`measured_np_median_ms`, median of 3 reps) --
                        the same substitution mk_fpga_csv.py makes, and for the
                        same reason: the campaign minimises the makespan of
                        one-shot work while periodic tasks keep their periods.

`wall_s` stays the SOLVER's time in both. It is a property of solving, not of
running, so it is identical across the two -- this is the reference's rule and
it is what keeps the Pareto x-axis meaningful on measured data.

TWO HONEST GAPS, carried into the figures rather than papered over:

  * `misses` in measured/results.csv is the PREDICTED miss count. The reference
    recomputes misses from the trace ("predicted misses would defeat the
    purpose of plotting hardware") and is right to; this sweep did not record
    per-instance release times, so the feasibility cue on the measured panel --
    the hollow markers -- is a property of the schedule, not of the run. Every
    figure drawn from that file must say so.
  * Only 12 of 42 cells were measured with all twelve solvers (the rest carry
    winner+greedy), so the measured panel ranks over 12 cells, not 42. It is
    written out with those cells only; mixing in cells where a solver never ran
    would compare solvers on different workload sets.

The non-discriminating exclusion is computed STRUCTURALLY here rather than
hardcoded by family name, which is the fix the reference's own comment
describes: a taskset with no aperiodic work has no makespan to minimise, so
every solver that finds a feasible schedule returns the identical objective,
and averaging those cells dilutes every solver toward zero. The criterion is
`ops == periodic_ops` for every cell in the family.

    python3 mk_qrb5165_csv.py
"""
import collections
import csv
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SWEEP = os.path.dirname(HERE)
RESULTS = os.path.join(SWEEP, "results")

COLS = ["arm", "family", "pair", "workload", "ops", "periodic_ops", "combos",
        "solver", "objective", "all_ops", "misses", "wall_s", "cpsat_status",
        "cpsat_gap", "prec_viol", "overlap_viol", "inf_dur_assign", "error"]


def split_cell(workload):
    """networks_<family>_<pair> -> (family, pair)."""
    stem = workload[len("networks_"):] if workload.startswith("networks_") else workload
    family, _, pair = stem.rpartition("_")
    return family, pair


def nondiscriminating(rows):
    """Families whose every cell is all-periodic, hence cannot rank a solver."""
    per_family = collections.defaultdict(list)
    for r in rows:
        family, _ = split_cell(r["workload"])
        per_family[family].append(int(r["ops"]) == int(r["periodic_ops"]))
    return sorted(f for f, flags in per_family.items() if all(flags))


def row_out(rec, objective, misses):
    family, pair = split_cell(rec["workload"])
    val = rec.get("validation") or {}
    return {
        "arm": rec["arm"], "family": family, "pair": pair,
        "workload": rec["workload"], "ops": rec["ops"],
        "periodic_ops": rec["periodic_ops"], "combos": rec.get("combos", ""),
        "solver": rec["solver"], "objective": objective,
        "all_ops": rec.get("all_ops", ""), "misses": misses,
        "wall_s": rec.get("wall_s", ""),
        "cpsat_status": (rec.get("cpsat") or {}).get("status", ""),
        "cpsat_gap": (rec.get("cpsat") or {}).get("gap", ""),
        "prec_viol": val.get("prec_viol", 0),
        "overlap_viol": val.get("overlap_viol", 0),
        "inf_dur_assign": val.get("inf_dur_assign", 0),
        "error": rec.get("error", ""),
    }


def write(path, rows, nondisc):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(os.path.dirname(path), "nondiscriminating.json"), "w") as fh:
        json.dump(nondisc, fh, indent=1)
    print(f"  {len(rows):>4} rows -> {path}")


def main():
    solves = json.load(open(os.path.join(RESULTS, "all_results.json")))
    nondisc = nondiscriminating(solves)
    print("non-discriminating families (all cells all-periodic):", nondisc)

    # ---- predicted ----
    write(os.path.join(HERE, "results.csv"),
          [row_out(r, r.get("objective", ""), r.get("misses", "")) for r in solves],
          nondisc)

    # ---- measured ----
    points = json.load(open(os.path.join(RESULTS, "phase4_results.json")))
    measured = [p for p in points if p.get("measured_np_median_ms") is not None]
    per_cell = collections.defaultdict(set)
    for p in measured:
        per_cell[p["workload"]].add(p["solver"])
    n_solvers = max((len(s) for s in per_cell.values()), default=0)
    full = {w for w, s in per_cell.items() if len(s) >= n_solvers}
    print(f"measured cells: {len(per_cell)}; with all {n_solvers} solvers: {len(full)}")

    write(os.path.join(HERE, "measured", "results.csv"),
          [row_out(p, p["measured_np_median_ms"], p.get("misses", ""))
           for p in measured if p["workload"] in full],
          nondisc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
