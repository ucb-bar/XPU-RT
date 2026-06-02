"""Axis B: compare a workload across profiler/backends (same schedule strategy).

Schedules the SAME networks workload under each available profile_hw backend
(e.g. gemmini_q31 vs V256D128_rvv; backends with no gen/profile data are
skipped), runs the advisor on each, and reports which backend wins and why.

    python3 scripts/compare_backends.py \
        --networks-json data/toplevel/networks_1yolo_4mlp_2dronet_firesim.json \
        --solver greedy --deadline-us 70
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _sched_eval as ev


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--networks-json", required=True)
    ap.add_argument("--solver", default="decomposed",
                    help="solver to hold fixed across backends (default decomposed: fast single-pass)")
    ap.add_argument("--scheduler", default=None)
    ap.add_argument("--backends", default=None,
                    help="comma-separated profile_hw names (default: auto-discover)")
    ap.add_argument("--deadline-us", type=float, default=None)
    ap.add_argument("--timeout", type=int, default=180)
    ap.add_argument("--emit", default=None)
    args = ap.parse_args()

    requested = [b.strip() for b in args.backends.split(",")] if args.backends else ev.available_backends()
    avail = ev.available_backends()
    rows = []
    for hw in requested:
        if hw not in avail:
            print(f"  [skip] {hw}: no gen/profile data for all models")
            rows.append({"backend": hw, "status": "skipped: no profiles"})
            continue
        print(f"=== backend {hw} ({args.solver}) ===", flush=True)
        res = ev.run_candidate(args.networks_json, profile_hw={"cpu_p": hw, "cpu_e": hw},
                               solver=args.solver, scheduler=args.scheduler, timeout=args.timeout)
        if res["status"] != "ok":
            print(f"  {hw}: {res['status']}")
            rows.append({"backend": hw, "status": res["status"]})
            continue
        diag = ev.advise(res["report"], args.deadline_us)
        rows.append({
            "backend": hw, "status": "ok",
            "makespan_us": round(diag.makespan_us, 2),
            "meets_deadline": diag.meets_deadline,
            "bottleneck": diag.bottleneck_backend,
            "granularity": diag.granularity_verdict,
            "report_path": os.path.relpath(res["report_path"], ev.REPO),
        })

    ok = [r for r in rows if r.get("status") == "ok"]
    print("\n" + "=" * 78)
    print(f"{'backend':<16}{'makespan_us':>13}{'meets':>8}{'granularity':>13}  bottleneck")
    print("-" * 78)
    for r in sorted(ok, key=lambda r: r["makespan_us"]):
        print(f"{r['backend']:<16}{r['makespan_us']:>13.2f}{str(r['meets_deadline']):>8}"
              f"{r['granularity']:>13}  {r['bottleneck']}")
    for r in rows:
        if r.get("status") != "ok":
            print(f"{r['backend']:<16}{'—':>13}  {r['status']}")
    if ok:
        best = min(ok, key=lambda r: r["makespan_us"])
        why = ("meets the deadline and " if best["meets_deadline"] else "")
        print(f"\nBest backend: {best['backend']} @ {best['makespan_us']:.2f} us "
              f"({why}lowest makespan; bottleneck {best['bottleneck']}, granularity {best['granularity']}).")

    if args.emit:
        with open(args.emit, "w") as f:
            json.dump(rows, f, indent=2)
        print(f"wrote {args.emit}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
