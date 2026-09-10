#!/usr/bin/env python3
"""Apply `prune_periodic` to the emitted schedules, and prove it left the
non-periodic objective alone.

Why this is a separate pass rather than part of the measurement emit. Every
ported spec carries `scheduler.prune_periodic: true`
(`scripts/mk_workloads_qrb5165.py:282`) and nothing in this tree ever read it:
the trim lives in `scripts/run_xpurt_schedule.py`, which sweep10 bypasses in
favour of `sweep10_runner.make_solver`. So the 199 schedules phase 4 measured
are all untrimmed, and re-emitting the new ones trimmed would give them
different content hashes from the very schedules they have to dedupe against.
The board therefore keeps running the untrimmed schedules -- which is what
keeps the campaign comparable with itself -- and the trim is applied here, to
the emitted and reported artefacts.

That is safe precisely because the trim is post-hoc.
`postprocessing.trim_periodic_after_nonperiodic_makespan` takes an
already-computed (t, alpha), computes the cut as the makespan over the
non-periodic operations, and drops periodic operations whose window does not
overlap [0, cut). It cannot move a placement, and because every dropped
operation starts at or after the cut it cannot move the non-periodic makespan
either. This script asserts that rather than assuming it: `emit_schedule.py
--prune-periodic` re-evaluates the objective on the trimmed workload and refuses
to write a schedule whose objective moved.

Usage:
  XPURT_CODE_ROOT=... XPURT_DATA_ROOT=... XPURT_CPSAT_PYTHON=... \\
  prune_periodic_check.py --solver cpsat:warmbest
"""
from __future__ import annotations

import argparse, json, os, subprocess, sys
import concurrent.futures as cf

HERE = os.path.dirname(os.path.abspath(__file__))
SWEEP = os.path.abspath(os.path.join(HERE, ".."))
EMIT = os.path.join(SWEEP, "fpga", "emit_schedule.py")
COST_MODEL = os.path.join(SWEEP, "cost_model.json")
STATE = os.path.join(SWEEP, "results", "phase4_state.json")
ARM = "s10port"


def one(job):
    wl, solver, outdir, cpsat_time, cpsat_workers = job
    tag = f'{wl.replace("networks_", "")}__{solver.replace(":", "-")}'
    out = os.path.join(outdir, f"scheduled_{tag}.json")
    meta = os.path.join(outdir, f"scheduled_{tag}.meta.json")
    p = subprocess.run(
        [sys.executable, EMIT, "--arm", ARM, "--name", wl, "--solver", solver,
         "--out", out, "--meta-out", meta, "--cpsat-time", str(cpsat_time),
         "--cpsat-workers", str(cpsat_workers), "--prune-periodic",
         "--mask-undeclared", COST_MODEL],
        capture_output=True, text=True)
    if p.returncode != 0 or not os.path.exists(meta):
        return dict(workload=wl, error=((p.stderr or p.stdout) or "").strip()[-300:])
    return json.load(open(meta))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--solver", default="cpsat:warmbest")
    ap.add_argument("--outdir", default=os.path.join(SWEEP, "schedules", "pruned"))
    ap.add_argument("--out", default=os.path.join(SWEEP, "results",
                                                  "prune_periodic_check.json"))
    ap.add_argument("--cpsat-time", type=float, default=60.0)
    ap.add_argument("--cpsat-workers", type=int, default=8)
    ap.add_argument("--parallel", type=int, default=5)
    a = ap.parse_args()

    os.makedirs(a.outdir, exist_ok=True)
    st = json.load(open(STATE))
    # every cell that has an emitted schedule for this solver, and the objective
    # that schedule was recorded with -- so a re-solve that drifted is caught
    # rather than silently compared against itself
    want = {r["workload"]: r for r in st.values() if r.get("solver") == a.solver
            and r.get("sched_hash")}
    jobs = [(wl, a.solver, a.outdir, a.cpsat_time, a.cpsat_workers)
            for wl in sorted(want)]
    with cf.ThreadPoolExecutor(max(1, a.parallel)) as ex:
        rows = list(ex.map(one, jobs))

    out, bad, drift = [], 0, 0
    for r in rows:
        wl = r.get("workload")
        rec = want.get(wl, {})
        if r.get("error"):
            print(f"  FAIL {wl}: {r['error']}")
            bad += 1
            out.append(dict(workload=wl, error=r["error"]))
            continue
        pr = r.get("pruned") or {}
        ok_np = bool(pr.get("objective_unchanged"))
        # the untrimmed solve has to be the one that was measured
        same_solve = (rec.get("objective") is not None
                      and abs(pr.get("pre_objective", -1) - rec["objective"]) <= 1e-6)
        if not same_solve:
            drift += 1
        row = dict(workload=wl, solver=a.solver,
                   measured_objective=rec.get("objective"),
                   pre_objective=pr.get("pre_objective"),
                   post_objective=r.get("objective"),
                   np_unchanged=ok_np,
                   solve_matches_measured=same_solve,
                   pre_all_ops=pr.get("pre_all_ops"),
                   post_all_ops=r.get("all_ops"),
                   pre_ops=pr.get("pre_ops"), post_ops=pr.get("post_ops"),
                   dropped_ops=pr.get("dropped_ops"),
                   pre_misses=pr.get("pre_misses"), post_misses=r.get("misses"),
                   masked_cells=r.get("masked_cells") or [],
                   pruned_schedule=os.path.relpath(r["schedule"], SWEEP))
        out.append(row)
        if not ok_np:
            bad += 1
    json.dump(out, open(a.out, "w"), indent=1)

    good = [r for r in out if not r.get("error")]
    print(f"\n{'cell':<34}{'np pre':>10}{'np post':>10}{'all pre':>10}"
          f"{'all post':>10}{'ops':>10}")
    for r in sorted(good, key=lambda r: r["workload"]):
        print(f'  {r["workload"][len("networks_"):]:<32}'
              f'{r["pre_objective"]:>10.3f}{r["post_objective"]:>10.3f}'
              f'{r["pre_all_ops"]:>10.3f}{r["post_all_ops"]:>10.3f}'
              f'{r["pre_ops"]:>6}->{r["post_ops"]:<4}')
    n_same = sum(1 for r in good if r["np_unchanged"])
    n_short = sum(1 for r in good if r["post_all_ops"] < r["pre_all_ops"] - 1e-9)
    n_drop = sum(1 for r in good if r["dropped_ops"])
    print(f"\n  {len(good)} cells: non-periodic objective unchanged on {n_same}, "
          f"all-operations makespan shortened on {n_short}, "
          f"operations dropped on {n_drop}")
    print(f"  solve reproduced the measured schedule's objective on "
          f"{sum(1 for r in good if r['solve_matches_measured'])}/{len(good)}")
    print(f"  -> {a.out}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
