#!/usr/bin/env python3
"""Tables for ANALYSIS.md, computed from results.json. Read-only."""
import json, os, statistics, sys

HERE = os.path.dirname(os.path.abspath(__file__))
R = {r["point"]: r for r in json.load(open(os.path.join(HERE, "results.json")))}


def f(v, w=8, p=2):
    return f"{v:{w}.{p}f}" if isinstance(v, (int, float)) else " " * (w - 2) + "--"


print("### 1. matched pair: baseline vs fused\n")
print(f"{'seed':<6} {'ops b/f':<9} {'solver b/f':<24} "
      f"{'pred b':>9} {'pred f':>9} {'d%':>7} | "
      f"{'act b':>9} {'act f':>9} {'d%':>7} {'spread b/f':>14}")
for s in range(8):
    b, u = R.get(f"baseline_seed{s}"), R.get(f"fused_seed{s}")
    if not b or not u:
        continue
    if b["status"] == "REJECTED" or u["status"] == "REJECTED":
        print(f"{s:<6} REJECTED both arms: {b['validation_problems']}")
        continue
    pb, pf = b.get("predicted_makespan_ms"), u.get("predicted_makespan_ms")
    ab, af = b.get("actual_makespan_median_ms"), u.get("actual_makespan_median_ms")
    dp = 100 * (pf - pb) / pb if pb and pf else None
    da = 100 * (af - ab) / ab if ab and af else None
    print(f"{s:<6} {b['op_count']:>3}/{u['op_count']:<5} "
          f"{str(b.get('solver')):<11}/{str(u.get('solver')):<12} "
          f"{f(pb,9)} {f(pf,9)} {f(dp,7,1)} | {f(ab,9)} {f(af,9)} {f(da,7,1)} "
          f"{f(b.get('actual_makespan_spread_ms'),6,2)}/{f(u.get('actual_makespan_spread_ms'),6,2)}")

print("\n### 2/3. ViNT: fused vs fused_vint (identical periodic set + identical yolov8n set)\n")
for s in (0, 1):
    a, v = R.get(f"fused_seed{s}"), R.get(f"fused_vint_seed{s}")
    if not a or not v:
        continue
    print(f"seed {s}:")
    for tag, r in (("fused", a), ("fused_vint", v)):
        print(f"  {tag:<11} ops {r['op_count']:>3}  entries {str(r.get('n_entries')):>3}  "
              f"horizon {r['horizon_ms']:7.1f}  solver {str(r.get('solver')):<15} "
              f"pred {f(r.get('predicted_makespan_ms'),9)}  "
              f"act {f(r.get('actual_makespan_median_ms'),9)}  "
              f"ratio {f(r.get('ratio_median'),6,3)}  lanes {r.get('lane_entry_counts')}")
    print()

print("\n### per-tile predicted vs actual, pooled over all run points\n")
agg = {}
for r in R.values():
    for k, v in (r.get("per_tile") or {}).items():
        tile = k.split("/", 1)[1] if "/" in k else k
        agg.setdefault(tile, []).append(v)
print(f"{'tile@lane':<34} {'n pts':>6} {'pred ms':>9} {'act ms':>9} {'ratio':>8} {'ratio range':>18}")
for tile in sorted(agg):
    vs = agg[tile]
    pr = statistics.median(v["predicted_ms"] for v in vs)
    ac = statistics.median(v["actual_p50_ms"] for v in vs)
    rs = sorted(v["ratio"] for v in vs if v["ratio"])
    print(f"{tile:<34} {len(vs):>6} {pr:>9.3f} {ac:>9.3f} {ac/pr:>8.3f} "
          f"{rs[0]:>8.2f} .. {rs[-1]:<8.2f}")

print("\n### makespan ratio per point (actual / predicted), medians over 3 reps\n")
print(f"{'point':<22} {'solver':<16} {'pred ms':>9} {'reps (ms)':<34} {'median':>9} {'ratio':>7} {'lock wait s':>28}")
for pt in sorted(R):
    r = R[pt]
    if r["status"] != "run":
        print(f"{pt:<22} {r['status']}")
        continue
    print(f"{pt:<22} {str(r.get('solver')):<16} {f(r.get('predicted_makespan_ms'),9)} "
          f"{str(r.get('actual_makespan_ms')):<34} "
          f"{f(r.get('actual_makespan_median_ms'),9)} {f(r.get('ratio_median'),7,3)} "
          f"{str(r.get('lock_wait_s')):>28}")
