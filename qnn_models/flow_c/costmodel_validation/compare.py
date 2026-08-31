#!/usr/bin/env python3
"""A/B the rebuilt cost model against the sweep's frozen one.

Identical tasksets, identical bindings, identical board conditions. The only
difference is which cost cells the solver was given. The test is whether the
makespan ratio moves toward 1.0 -- and, just as important, whether it STAYS
there, since promoting in-situ values changes the schedule, which can change
the contention those values encoded.
"""
import json, os, statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
FZ = json.load(open(os.path.join(HERE, "..", "sweeps", "qrb5165_20260829-200620",
                                 "results.json")))
BEFORE = {r["point"]: r for r in FZ if r.get("status") == "run"}
AFTER = {r["point"]: r for r in json.load(open(os.path.join(HERE, "results.json")))
         if r.get("status") == "run"}

print(f"{'point':<20} | {'BEFORE (frozen cells)':^32} | {'AFTER (in-situ cells)':^32}")
print(f"{'':<20} | {'pred':>8} {'actual':>8} {'ratio':>7}      | "
      f"{'pred':>8} {'actual':>8} {'ratio':>7}      | {'ratio move':>11}")
b_r, a_r = [], []
for pt in sorted(AFTER):
    b, a = BEFORE.get(pt), AFTER[pt]
    if not b: continue
    b_r.append(b["ratio_median"]); a_r.append(a["ratio_median"])
    d = abs(a["ratio_median"] - 1) - abs(b["ratio_median"] - 1)
    verdict = "better" if d < -0.005 else ("worse" if d > 0.005 else "same")
    print(f"{pt:<20} | {b['predicted_makespan_ms']:>8.2f} "
          f"{b['actual_makespan_median_ms']:>8.2f} {b['ratio_median']:>7.3f}      | "
          f"{a['predicted_makespan_ms']:>8.2f} {a['actual_makespan_median_ms']:>8.2f} "
          f"{a['ratio_median']:>7.3f}      | {verdict:>11}")
if b_r:
    print(f"\n  median |ratio - 1|   before {st.median(abs(x-1) for x in b_r)*100:5.1f}%"
          f"   ->  after {st.median(abs(x-1) for x in a_r)*100:5.1f}%")
    print(f"  median ratio         before {st.median(b_r):.3f}   ->  after {st.median(a_r):.3f}")

# did the measured wall time itself change? (a different schedule may simply be faster)
print()
for pt in sorted(AFTER):
    b, a = BEFORE.get(pt), AFTER[pt]
    if not b: continue
    dw = 100 * (a["actual_makespan_median_ms"] - b["actual_makespan_median_ms"]) / b["actual_makespan_median_ms"]
    print(f"  {pt:<20} measured wall {b['actual_makespan_median_ms']:7.2f} -> "
          f"{a['actual_makespan_median_ms']:7.2f} ms  ({dw:+.1f}%)   "
          f"lanes {b.get('lane_entry_counts')} -> {a.get('lane_entry_counts')}")
