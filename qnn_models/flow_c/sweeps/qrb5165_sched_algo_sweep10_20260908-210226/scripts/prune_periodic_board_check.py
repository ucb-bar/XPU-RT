#!/usr/bin/env python3
"""Run the PRUNED schedule on the board and check the measured non-periodic
makespan against the untrimmed run of the same solve.

`prune_periodic_check.py` proves the claim in the cost model's own arithmetic:
`trim_periodic_after_nonperiodic_makespan` drops only periodic operations that
start at or after the non-periodic makespan, so the objective it is scored on
cannot move. That is an argument about the schedule. This is the measurement:
build a runtime from the trimmed schedule, run it the same number of reps
through the same harness, and compare the median non-periodic makespan with the
untrimmed point's.

Both numbers come from the runtime's own trace block, so they are the same
quantity computed the same way -- the only difference is which entries are in
the dispatch table.

The comparison is deliberately NOT the wall clock. The trim's whole purpose is
to shorten that, and it does (bimodal_dc: 32.45 ms of table against 8.69 ms).

    prune_periodic_board_check.py --cells bimodal_dc,saturation_dc,...
"""
from __future__ import annotations

import argparse, importlib.util, json, os, statistics, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
SWEEP = os.path.abspath(os.path.join(HERE, ".."))
FLOWC = os.path.abspath(os.path.join(SWEEP, "..", ".."))
BOARD = os.environ.get("QNN_BOARD_HOST", "root@10.44.120.201")


def _drive():
    """Reuse the campaign's own run parsing and lock probe verbatim -- a second
    implementation of `analyse_run` would be a second definition of the number
    being compared."""
    spec = importlib.util.spec_from_file_location(
        "sweep10_drive", os.path.join(HERE, "drive.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", required=True,
                    help="comma-separated short cell names, e.g. bimodal_dc")
    ap.add_argument("--solver", default="cpsat:warmbest")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--run-timeout", type=int, default=1800)
    ap.add_argument("--out", default=os.path.join(
        SWEEP, "results", "prune_periodic_board_check.json"))
    a = ap.parse_args()

    dv = _drive()
    # phase4_RESULTS, not phase4_state: a point whose schedule deduped onto
    # another point's has no reps of its own in the state file, and its
    # measurement lives only in the results file, attributed by content hash.
    # Reading the state here is how the first run of this script printed a
    # blank baseline for saturation_dc and depth_contended_hd, both duplicates.
    st = {r["workload"]: r for r in json.load(open(
        os.path.join(SWEEP, "results", "phase4_results.json")))
        if r["solver"] == a.solver}
    sv_tag = a.solver.replace(":", "-")
    out = []
    for short in [c.strip() for c in a.cells.split(",") if c.strip()]:
        wl = f"networks_{short}"
        base = st.get(wl) or {}
        pruned = os.path.join(SWEEP, "schedules", "pruned",
                              f"scheduled_{short}__{sv_tag}.json")
        if not os.path.exists(pruned):
            print(f"  {short}: no pruned schedule at {pruned}")
            continue
        pid = f"{short}__{sv_tag}-pruned"
        out_dir = os.path.join(SWEEP, "runtimes", pid)
        q, _ = dv.sh([sys.executable, "flow_c.py", "runtime", "--workload",
                      os.path.join(SWEEP, "specs", f"{wl}.flowc.json"),
                      "--tag", pid, "--lane-mode", "kind-network",
                      "--schedule", pruned, "--out-dir", out_dir],
                     log=os.path.join(SWEEP, "logs", "runtime", pid + ".log"),
                     cwd=FLOWC, timeout=1800)
        if q.returncode != 0:
            print(f"  {short}: runtime emit failed rc={q.returncode}")
            continue
        table = dv.parse_table(os.path.join(out_dir, "dispatch_table.h"))
        per_nets, np_nets = dv.periodic_networks(wl)
        reps = {}
        for rep in range(1, a.reps + 1):
            log_dir = os.path.join(SWEEP, "runs", pid, f"rep{rep}")
            os.makedirs(log_dir, exist_ok=True)
            wait, wrc = dv.lock_wait_s()
            r, dt = dv.sh([sys.executable, "flow_c.py", "run", "--workload",
                           os.path.join(SWEEP, "specs", f"{wl}.flowc.json"),
                           "--tag", pid, "--tuned", "--out-dir", out_dir,
                           "--log-dir", log_dir, "--board", BOARD,
                           "--board-dir", "/root/flowc_s10run"],
                          log=os.path.join(log_dir, "driver.log"),
                          cwd=FLOWC, timeout=a.run_timeout)
            info = dv.analyse_run(os.path.join(log_dir, "run.log"),
                                  per_nets, np_nets)
            info.update(lock_wait_s=wait, lock_probe_rc=wrc, flow_c_rc=r.returncode)
            reps[f"rep{rep}"] = info
            print(f"  {pid:<44} rep{rep}  wall {info.get('wall_ms')}  np "
                  f"{info.get('measured_nonperiodic_ms')}  lock {wait}s  "
                  f"ok={info.get('ok')}")
        nps = [v["measured_nonperiodic_ms"] for v in reps.values()
               if v.get("measured_nonperiodic_ms")]
        walls = [v["wall_ms"] for v in reps.values() if v.get("wall_ms")]
        row = dict(
            cell=wl, solver=a.solver,
            untrimmed_entries=base.get("n_entries"),
            pruned_entries=len(table),
            untrimmed_np_median_ms=base.get("measured_np_median_ms"),
            pruned_np_median_ms=round(statistics.median(nps), 3) if nps else None,
            untrimmed_np_reps_ms=base.get("measured_np_reps_ms"),
            pruned_np_reps_ms=nps,
            untrimmed_wall_median_ms=base.get("measured_median_ms"),
            pruned_wall_median_ms=round(statistics.median(walls), 3) if walls else None,
            untrimmed_table_makespan_ms=base.get("table_predicted_makespan_ms"),
            pruned_table_makespan_ms=round(
                max(t["start_ms"] + t["dur_ms"] for t in table), 3) if table else None,
            reps_ok=all(v.get("ok") for v in reps.values()),
        )
        if row["untrimmed_np_median_ms"] and row["pruned_np_median_ms"]:
            row["np_ratio"] = round(row["pruned_np_median_ms"]
                                    / row["untrimmed_np_median_ms"], 4)
            # both medians carry the campaign's own rep spread, so "unchanged"
            # has to mean "inside that spread", not "bit-identical"
            sp = [abs(max(x) - min(x)) / statistics.median(x) * 100
                  for x in (row["untrimmed_np_reps_ms"] or [None],
                            row["pruned_np_reps_ms"]) if x and None not in x]
            row["rep_spread_pct"] = [round(v, 2) for v in sp]
        out.append(row)
        dv.board(f"rm -rf /root/flowc_s10run_{pid}; df -h / | tail -1", timeout=120)

    json.dump(out, open(a.out, "w"), indent=1)
    print(f"\n{'cell':26s}{'entries':>16s}{'np untrim':>11s}{'np pruned':>11s}"
          f"{'ratio':>8s}{'wall untrim':>13s}{'wall pruned':>13s}")
    for r in out:
        print(f'{r["cell"][len("networks_"):]:26s}'
              f'{str(r["untrimmed_entries"]) + "->" + str(r["pruned_entries"]):>16s}'
              f'{(r["untrimmed_np_median_ms"] or 0):11.3f}'
              f'{(r["pruned_np_median_ms"] or 0):11.3f}'
              f'{r.get("np_ratio", 0):8.3f}'
              f'{(r["untrimmed_wall_median_ms"] or 0):13.3f}'
              f'{(r["pruned_wall_median_ms"] or 0):13.3f}')
    print(f"  -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
