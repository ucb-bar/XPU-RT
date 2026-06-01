"""Profile multiple schedulers on one networks benchmark, then advise on each.

Runs run_xpurt_schedule.py for a list of (solver, scheduler) combos on the same
networks JSON, reads each emitted SchedulerReport (*_report.json), runs the
deadline-aware advisor, and prints a comparison table. Optionally renders a
terminal Gantt per scheduler.

Example (Dima's firesim 3-model benchmark):
    python3 scripts/profile_schedulers.py \
        --networks-json data/toplevel/networks_mlp10_dronet20_yolov8_firesim_static_q31profile.json \
        --schedulers decomposed,greedy,heft,peft,edf,fifo,round_robin,critical_path,fastest_device \
        --deadline-us 70 --gantt

`decomposed`/`greedy`/`greedy_periodic` run as --solver; everything else runs as
--solver milp --scheduler <name> (the registry path). Each run is independent;
a failure is recorded and the sweep continues.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "xpu-rt"))

GREEDY_FAMILY = {"decomposed", "greedy", "greedy_periodic"}


def _report_path(networks_json: str, solver: str, scheduler: str, profiled: bool) -> str:
    base = os.path.splitext(os.path.basename(networks_json))[0]
    if solver in GREEDY_FAMILY:
        tag = f"_{solver}"
    else:
        tag = "" if scheduler == "mosek" else f"_{scheduler}"
    ptag = "_profiled" if profiled else ""
    return os.path.join(REPO, "schedules", f"scheduled_{base}{tag}{ptag}_report.json")


def run_one(networks_json: str, name: str, timeout: int, profiled: bool) -> dict:
    if name in GREEDY_FAMILY:
        cmd = [sys.executable, os.path.join(REPO, "scripts", "run_xpurt_schedule.py"),
               "--networks-json", networks_json, "--solver", name]
        scheduler = name
    else:
        cmd = [sys.executable, os.path.join(REPO, "scripts", "run_xpurt_schedule.py"),
               "--networks-json", networks_json, "--solver", "milp", "--scheduler", name]
        scheduler = name
    if not profiled:
        cmd.append("--no-profiled")
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"scheduler": name, "status": "timeout", "wall_s": timeout}
    wall = time.time() - t0
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout)[-300:].replace("\n", " ")
        return {"scheduler": name, "status": f"error: {tail}", "wall_s": round(wall, 1)}
    rp = _report_path(networks_json, name if name in GREEDY_FAMILY else "milp", scheduler, profiled)
    return {"scheduler": name, "status": "ok", "wall_s": round(wall, 1), "report": rp}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--networks-json", required=True)
    ap.add_argument("--schedulers", default="decomposed,heft,peft,edf,fifo,round_robin,critical_path",
                    help="comma-separated solver/scheduler names")
    ap.add_argument("--deadline-us", type=float, default=None)
    ap.add_argument("--timeout", type=int, default=120, help="per-run timeout (s)")
    ap.add_argument("--no-profiled", action="store_true")
    ap.add_argument("--gantt", action="store_true", help="print a terminal Gantt per scheduler")
    ap.add_argument("--emit", default=None, help="write the comparison rows to this JSON")
    args = ap.parse_args()

    import advisor as advisor_mod
    import plot_gantt

    names = [s.strip() for s in args.schedulers.split(",") if s.strip()]
    rows = []
    for name in names:
        print(f"\n=== running {name} ===", flush=True)
        res = run_one(args.networks_json, name, args.timeout, profiled=not args.no_profiled)
        if res["status"] != "ok":
            print(f"  {name}: {res['status']}")
            rows.append({**res, "makespan_us": None})
            continue
        try:
            with open(res["report"]) as f:
                report = json.load(f)
            diag = advisor_mod.advise_schedule(report, deadline_us=args.deadline_us)
            top = next((r for r in diag.recommendations if r.kind != "none"), None)
            row = {
                "scheduler": name, "status": "ok", "wall_s": res["wall_s"],
                "makespan_us": round(diag.makespan_us, 2),
                "deadline_miss_count": diag.deadline_miss_count,
                "meets_deadline": diag.meets_deadline,
                "bottleneck": diag.bottleneck_backend,
                "granularity": diag.granularity_verdict,
                "top_rec": (f"{top.kind}:{top.target}" if top else None),
            }
            rows.append(row)
            if args.gantt:
                print(plot_gantt.render_terminal_gantt(report, deadline_us=args.deadline_us, width=72))
        except Exception as exc:
            print(f"  {name}: advise/gantt failed: {exc}")
            rows.append({"scheduler": name, "status": f"advise_error: {exc}", "makespan_us": None})

    # comparison table
    print("\n" + "=" * 88)
    print(f"{'scheduler':<16}{'makespan_us':>13}{'miss':>6}{'meets':>7}{'gran':>11}{'wall_s':>8}  top_rec")
    print("-" * 88)
    ok = [r for r in rows if r.get("makespan_us") is not None]
    for r in sorted(ok, key=lambda r: r["makespan_us"]):
        print(f"{r['scheduler']:<16}{r['makespan_us']:>13.2f}{r.get('deadline_miss_count', 0):>6}"
              f"{str(r.get('meets_deadline')):>7}{r.get('granularity', '?'):>11}{r.get('wall_s', 0):>8.1f}"
              f"  {r.get('top_rec') or ''}")
    for r in rows:
        if r.get("makespan_us") is None:
            print(f"{r['scheduler']:<16}{'—':>13}  {r['status']}")
    if ok:
        best = min(ok, key=lambda r: r["makespan_us"])
        print(f"\nBest makespan: {best['scheduler']} @ {best['makespan_us']:.2f} us")

    if args.emit:
        with open(args.emit, "w") as f:
            json.dump(rows, f, indent=2)
        print(f"wrote {args.emit}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
