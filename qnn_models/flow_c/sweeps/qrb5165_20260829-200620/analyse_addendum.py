#!/usr/bin/env python3
"""Q1 on the periodic-only addendum. Read-only.

The addendum was built on the premise that a periodic-only draw puts the
mid-size model (dronet vs fused_split) on the critical path, which the 12
accepted points never do -- there a yolov8n_head always finishes last.

The solves show the premise holds for ONE of the three pairs. In a purely
periodic workload the makespan is `last release of some periodic network +
that network's duration`, and which network that is depends on the draw:

  seed 4, 5   the highest-count network (mlp_control) is released last, so
              the makespan is set by the RELEASE SCHEDULE and is identical
              across arms. Uninformative for Q1 -- but still a measurement
              of predicted-vs-actual in a periodic-only regime, which the
              main sweep never covers.
  seed 7      the mid-size network is released last and owns the makespan.
              This is the informative pair.

Both facts are reported below rather than assumed: the owner of the makespan
is read out of the solved schedule and, independently, out of the traces.
"""
import csv, glob, json, os, statistics

HERE = os.path.dirname(os.path.abspath(__file__))
ADD = os.path.join(HERE, "addendum_periodic_only")
A = {r["point"]: r for r in json.load(open(os.path.join(ADD, "results.json")))}
M = {r["point"]: r for r in json.load(open(os.path.join(HERE, "results.json")))}
MID = ("dronet", "fused_split")


def predicted_owner(point):
    """(job, end_ms) of the last-ending dispatch in the solved schedule."""
    for pat in (f"schedules/scheduled_{point}_profiled.json",
                f"schedules/scheduled_{point}_greedy_periodic_profiled.json"):
        p = os.path.join(ADD, pat)
        if not os.path.exists(p):
            continue
        d = json.load(open(p))["dispatches"]
        if not d:
            continue
        e = max(d.values(), key=lambda v: v["start_time"] + v["duration"])
        return e["job_name"], e["start_time"] + e["duration"]
    return None, None


def measured_owner(point, root):
    """(network, end_ms) of the last-finishing entry, max over reps."""
    best = None
    for p in sorted(glob.glob(os.path.join(root, "runs", point, "rep*", "run.log"))):
        rows, on = [], False
        for ln in open(p):
            if "TRACE_BEGIN" in ln: on = True; continue
            if "TRACE_END" in ln: break
            if on: rows.append(ln)
        rd = list(csv.DictReader(rows)) if rows else []
        if not rd: continue
        # Flow C traces carry actual_end_cycles + time_unit (us with
        # --clock-mhz 1); the ModelBlaster-style actual_end_ms is absent.
        def end_ms(r):
            v = float(r.get("actual_end_cycles") or r.get("actual_end_ms") or 0)
            return v / 1000.0 if r.get("time_unit") == "us" else v
        e = max(rd, key=end_ms)
        c = (e["network"], end_ms(e))
        if best is None or c[1] > best[1]: best = c
    return best or (None, None)


def spread_pct(r):
    v = r.get("actual_makespan_ms") or []
    return 100 * (max(v) - min(v)) / statistics.median(v) if len(v) > 1 else None


def is_mid(name):
    return name is not None and any(name.startswith(m) for m in MID)


print("### premise check: which network owns the makespan?\n")
print(f"{'point':<20} {'set':<9} {'predicted owner':<20} {'measured owner':<20} {'mid-size?':>10}")
for pt in sorted(A):
    po, _ = predicted_owner(pt)
    mo, _ = measured_owner(pt, ADD)
    print(f"{pt:<20} {'addendum':<9} {str(po):<20} {str(mo):<20} "
          f"{('YES' if is_mid(mo) else 'no'):>10}")
for pt in sorted(M):
    if M[pt]["status"] != "run": continue
    mo, _ = measured_owner(pt, HERE)
    print(f"{pt:<20} {'main':<9} {'--':<20} {str(mo):<20} "
          f"{('YES' if is_mid(mo) else 'no'):>10}")

print("\n### Q1 matched pairs (periodic-only)\n")
print(f"{'seed':<5} {'ops b/f':<8} {'pred b':>8} {'pred f':>8} {'d%':>6} | "
      f"{'act b':>8} {'act f':>8} {'d%':>6} | {'noise b/f %':>12}  {'informative':<12} verdict")
for s in (4, 5, 7):
    b, u = A.get(f"baseline_seed{s}"), A.get(f"fused_seed{s}")
    if not b or not u or b["status"] != "run" or u["status"] != "run":
        print(f"{s:<5} incomplete"); continue
    pb, pf = b["predicted_makespan_ms"], u["predicted_makespan_ms"]
    ab, af = b["actual_makespan_median_ms"], u["actual_makespan_median_ms"]
    dp, da = 100*(pf-pb)/pb, 100*(af-ab)/ab
    nb, nf = spread_pct(b) or 0.0, spread_pct(u) or 0.0
    info = is_mid(measured_owner(f"baseline_seed{s}", ADD)[0]) or \
           is_mid(measured_owner(f"fused_seed{s}", ADD)[0])
    if not info:
        v = "release-dominated"
    elif da < -max(nb, nf): v = "fused wins"
    elif da > max(nb, nf):  v = "dronet wins"
    else:                   v = "within noise"
    print(f"{s:<5} {b['op_count']:>3}/{u['op_count']:<4} {pb:>8.2f} {pf:>8.2f} {dp:>6.1f} | "
          f"{ab:>8.2f} {af:>8.2f} {da:>6.1f} | {nb:>5.1f}/{nf:<6.1f} "
          f"{('YES' if info else 'no'):<12} {v}")

print("\n### predicted vs actual, periodic-only regime (new: main sweep has no such point)\n")
print(f"{'point':<20} {'pred ms':>9} {'reps (ms)':<30} {'median':>9} {'ratio':>7}")
for pt in sorted(A):
    r = A[pt]
    if r["status"] != "run":
        print(f"{pt:<20} {r['status']}"); continue
    print(f"{pt:<20} {r['predicted_makespan_ms']:>9.2f} "
          f"{str(r.get('actual_makespan_ms')):<30} "
          f"{r['actual_makespan_median_ms']:>9.2f} {r.get('ratio_median', 0):>7.3f}")
mr = [r["ratio_median"] for r in A.values() if r["status"] == "run" and r.get("ratio_median")]
if mr:
    print(f"\nmedian ratio, periodic-only: {statistics.median(mr):.3f}x   "
          f"(main sweep, mixed: 1.17x)")
