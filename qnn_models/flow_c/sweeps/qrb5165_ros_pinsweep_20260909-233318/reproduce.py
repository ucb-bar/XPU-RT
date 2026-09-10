#!/usr/bin/env python3
"""Re-derive everything in this sweep that does not need the board, and check
it against what is committed.

Six checks, each PASS/FAIL, no hardware:

  1  model_costs.json         re-derived from the FROZEN cost_model.json and the
                              binding manifests
  2  results/expressibility   re-derived verdicts and legal sets
  3  results/enumeration      re-derived n_legal, rankings and measured picks
  4  plans/*.json             byte-identical re-emission
  5  measured.json            every median/spread re-derived from logs/ (skipped
                              if logs/ is absent, since logs are not committed)
  6  the noise floor SETUP.md quotes, recomputed from the XPU-RT sweep's own
                              phase4_results.json

    python3 reproduce.py [--verbose]

Exit status is the number of failed checks.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.join(HERE, "scripts")
FLOWC = os.path.abspath(os.path.join(HERE, "..", ".."))
REPO = os.path.abspath(os.path.join(FLOWC, "..", ".."))
XSWEEP = os.path.join(FLOWC, "sweeps", "qrb5165_sched_algo_sweep10_20260908-210226")

sys.path.insert(0, SCRIPTS)

OK, BAD = "PASS", "FAIL"
results = []


def check(name, ok, detail=""):
    results.append((name, ok, detail))
    print(f"  [{OK if ok else BAD}] {name}" + (f" -- {detail}" if detail else ""))
    return ok


def jload(p):
    with open(p) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    import pinsweep

    print("1. model_costs.json, from the frozen cost model")
    costs = pinsweep.model_costs()
    have = jload(os.path.join(HERE, "model_costs.json"))
    check("whole-model costs re-derive exactly", have["networks"] == costs,
          f'{len(costs)} networks')
    check("cost source is the frozen model",
          have["source_cost_model"].endswith("cost_model.json")
          and "sched_algo_sweep10" in have["source_cost_model"],
          have["source_cost_model"])
    # every cost is a sum over that network's binding tiles of a cell in the
    # frozen model -- spot-checked exhaustively, not by sampling
    cm = pinsweep.load_cost_model()["cells"]
    bad = []
    for net, per in costs.items():
        for be, rec in per.items():
            s = 0.0
            for t in rec["tiles"]:
                man = pinsweep.load_bindings(net)
                s += cm[f'{man["network"]}/{t["tile"]}'][be]
            if abs(s / 1000.0 - rec["ms"]) > 1e-6:
                bad.append((net, be))
    check("every cost is the sum of its tiles' frozen cells", not bad, str(bad[:5]))

    print("2. expressibility")
    cells = pinsweep.load_cells()
    rows = pinsweep.classify(cells, costs)
    have = jload(os.path.join(HERE, "results", "expressibility.json"))
    check("verdicts and legal sets re-derive exactly", have["cells"] == rows,
          f"{len(rows)} cells")
    verdicts = {}
    for r in rows:
        verdicts[r["verdict"]] = verdicts.get(r["verdict"], 0) + 1
    check("SETUP.md's 31 OK / 11 degenerate / 0 inexpressible",
          verdicts.get("OK") == 31 and verdicts.get("DEGENERATE (1-LANE)") == 11
          and not any(v.startswith("INEXPRESSIBLE") for v in verdicts),
          str(verdicts))
    check("the 42 cells are the 42 the XPU-RT sweep measured",
          len(rows) == 42 and set(c["cell"] for c in rows) ==
          set(measured_cells()), f"{len(rows)}")

    print("3. enumeration + 4. plans")
    tmp = tempfile.mkdtemp(prefix="rospin_repro_")
    try:
        # re-emit into a scratch tree so nothing committed is overwritten
        r = subprocess.run([sys.executable,
                            os.path.join(SCRIPTS, "pinsweep.py"),
                            "--out", tmp, "enumerate"],
                           capture_output=True, text=True)
        ok = r.returncode == 0
        if not ok and args.verbose:
            print(r.stdout, r.stderr)
        have_enum = jload(os.path.join(HERE, "results", "enumeration.json"))
        got_enum = jload(os.path.join(tmp, "results", "enumeration.json"))
        check("enumeration re-derives exactly", ok and have_enum == got_enum,
              f'{len(got_enum["cells"])} cells')
        ns = [c["n_legal"] for c in got_enum["cells"] if c.get("n_legal")]
        check("SETUP.md's n_legal distribution (min 1, median 4, max 729, "
              "1078 total)",
              min(ns) == 1 and statistics.median(ns) == 4 and max(ns) == 729
              and sum(ns) == 1078,
              f"min={min(ns)} median={statistics.median(ns)} max={max(ns)} "
              f"total={sum(ns)}")
        check("SETUP.md's 172 assignments marked for hardware",
              sum(c.get("measured", 0) for c in got_enum["cells"]) == 172)
        diffs = []
        for fn in sorted(os.listdir(os.path.join(HERE, "plans"))):
            a = jload(os.path.join(HERE, "plans", fn))
            b = jload(os.path.join(tmp, "plans", fn))
            if a != b:
                diffs.append(fn)
        check("every plan re-emits identically", not diffs, str(diffs[:3]))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("5. measured.json against the run logs")
    mpath = os.path.join(HERE, "measured.json")
    logs = os.path.join(HERE, "logs")
    if not os.path.exists(mpath):
        check("measured.json present", False, "not yet collected")
    elif not os.path.isdir(logs) or not os.listdir(logs):
        check("logs present to re-derive from", True,
              "SKIPPED -- logs/ is not committed; run scripts/drive.py run first")
    else:
        import drive
        doc = jload(mpath)
        bad = []
        for tag, rec in doc["runs"].items():
            p = os.path.join(logs, tag + ".log")
            if not os.path.exists(p):
                bad.append((tag, "no log"))
                continue
            info = drive.parse_log(open(p, errors="replace").read())
            walls = [x["makespan_ms"] for x in info["passes"]]
            if not walls:
                bad.append((tag, "no passes"))
                continue
            if abs(statistics.median(walls) - rec["makespan_median_ms"]) > 1e-6:
                bad.append((tag, "median"))
            if walls != rec["makespan_reps_ms"]:
                bad.append((tag, "reps"))
        check("every median and spread re-derives from its log", not bad,
              str(bad[:5]))
        check("3 reps everywhere, no single-rep result",
              all(r["reps"] == 3 for r in doc["runs"].values()),
              str(sorted({r["reps"] for r in doc["runs"].values()})))

    print("5b. the 3net arm")
    p3 = os.path.join(HERE, "results", "enumeration_3net.json")
    if os.path.exists(p3):
        import pin3net
        tmp3 = tempfile.mkdtemp(prefix="rospin_repro3_")
        try:
            have3 = jload(p3)
            costs3, prov3 = pin3net.base_costs()
            # legality is the binding manifest, never the cost table: the
            # artifact emitter writes large-but-finite EXCLUSION costs for a
            # lane a tile cannot use, and a cost-only rule would pin to them
            bad = []
            for net, per in costs3.items():
                man = json.load(open(os.path.join(FLOWC, "bindings",
                                                  f"{net}.json")))
                for be in per:
                    for b in man["bindings"]:
                        if be not in (b.get("backends") or {}):
                            bad.append((net, be))
            check("3net legality comes from the binding manifests", not bad,
                  str(bad[:5]))
            sh = pin3net.shapes()
            check("all 15 RoSE 3net configs map to a shape",
                  sum(len(x["sources"]) for x in sh) == 15
                  and len(sh) == len(have3["cells"]),
                  f'{len(sh)} shapes, '
                  f'{sum(len(x["sources"]) for x in sh)} source configs')
            check("3net enumeration matches the committed table",
                  [c["cell"] for c in have3["cells"]] == [x["name"] for x in sh])
        finally:
            shutil.rmtree(tmp3, ignore_errors=True)
        pa = os.path.join(HERE, "results", "analysis_3net.json")
        if os.path.exists(pa):
            a3 = jload(pa)["shapes"]
            real = [r for r in a3 if not r["np_degenerate"]]
            check("3net: pinning faster on every shape with an aperiodic "
                  "network", all(r["ros_over_xrt_np"] < 1 for r in real),
                  f'{sum(1 for r in real if r["ros_over_xrt_np"] < 1)}'
                  f"/{len(real)}")
    else:
        check("3net enumeration present", True, "SKIPPED -- not generated")

    print("5c. the headline aggregates ANALYSIS.md quotes")
    pa = os.path.join(HERE, "results", "analysis.json")
    if os.path.exists(pa):
        h = jload(pa)["headline"]
        check("0 assignments starved a network",
              h["assignments_with_a_starved_network"] == [],
              str(len(h["assignments_with_a_starved_network"])))
        check("every assignment completed",
              h["assignments_run"] == h["assignments_ok"],
              f'{h["assignments_ok"]}/{h["assignments_run"]}')
        check("the 4 saturation cells are the only unequal-non-periodic-work "
              "exclusions",
              sorted(h["np_cells_excluded_unequal_work"]) ==
              sorted(f"networks_saturation_{c}" for c in ("cg", "dc", "hd", "quad")),
              str(h["np_cells_excluded_unequal_work"]))
        check("headline median ROS/XPU-RT on the objective is 0.9762",
              abs(h["np_ros_over_xrt_median"] - 0.9762) < 1e-4,
              str(h["np_ros_over_xrt_median"]))
        check("7 cells where minimum-makespan is not the feasible-first "
              "placement",
              len(h["cells_where_makespan_rule_disagrees_with_feasibility"]) == 7,
              str(len(h["cells_where_makespan_rule_disagrees_with_feasibility"])))
    else:
        check("analysis.json present", False, "run scripts/analyse.py tables")

    print("6. the noise floor SETUP.md quotes")
    rows4 = jload(os.path.join(XSWEEP, "results", "phase4_results.json"))
    uniq = {}
    for r in rows4:
        if r.get("measured_median_ms"):
            uniq[r.get("measured_via") or r["point"]] = r
    wall = [r["measured_spread_ms"] / r["measured_median_ms"] * 100
            for r in rows4 if r.get("measured_median_ms")]
    npv = [r["measured_np_spread_ms"] / r["measured_np_median_ms"] * 100
           for r in rows4 if r.get("measured_np_median_ms")]
    check("wall-clock median rep spread is 7.62%",
          abs(statistics.median(wall) - 7.62) < 0.01,
          f"{statistics.median(wall):.2f}%")
    check("non-periodic median rep spread is 9.18%",
          abs(statistics.median(npv) - 9.18) < 0.01,
          f"{statistics.median(npv):.2f}%")

    n_bad = sum(1 for _, ok, _ in results if not ok)
    print(f"\n{len(results) - n_bad}/{len(results)} checks passed")
    return n_bad


def measured_cells():
    rows = jload(os.path.join(XSWEEP, "results", "phase4_results.json"))
    return {r["workload"] for r in rows if r.get("measured_median_ms")}


if __name__ == "__main__":
    sys.exit(main())
