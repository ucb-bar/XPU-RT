#!/usr/bin/env python3
"""What each arm actually does to the command stream, MEASURED from the summaries.

The success numbers alone do not say whether an arm hurt because the observation
was stale or because the command stream changed shape. This prints, per task and
arm, the three quantities that separate those:

  age@act   mean age (ms) of the observation behind the command at the moment it
            is ACTUATED. This is the staleness the robot actually experiences --
            not the same as the modelled latency, because the ZOH holds a command
            across the gap between arrivals.
  disp/ep   policy dispatches per episode = how often a fresh result is produced.
  upd/act   distinct results actually reaching the actuator per actuation. On
            google_robot (--actuation native) this is capped at 1.0 by the 3 Hz
            control grid no matter how fast the cadence is -- which is exactly
            why a faster cadence cannot buy "more replanning" there the way it
            can in the widowx fine model.
"""
from __future__ import annotations
import json, glob, collections, argparse
from pathlib import Path

HERE = Path(__file__).parent
ARM_ORDER = ["lat0", "pipe110", "pipe200", "serial283", "fp32_555", "cpu685"]

ap = argparse.ArgumentParser()
ap.add_argument("--runs", default=str(HERE / "runs"))
a = ap.parse_args()

rows = collections.defaultdict(list)
meta = {}
for f in glob.glob(str(Path(a.runs) / "*" / "*" / "summary.json")):
    try: d = json.load(open(f))
    except Exception: continue
    name = Path(f).parent.name
    if "_rng" not in name: continue
    head, _ = name.rsplit("_rng", 1)
    task, arm = head.split("_", 1)
    if d.get("policy_setup") == "google_robot" and d.get("actuation") != "native":
        continue
    eps = d.get("episodes", [])
    if not eps: continue
    meta[task] = (d.get("actuation", "fine"), d.get("act_ms", d["tick_ms"]),
                  d["tick_ms"], d.get("act_every_ticks", 1))
    for e in eps:
        aa = e.get("act_age_mean_ms")
        na = e.get("n_actuations", e["ticks"])
        rows[(task, arm)].append((aa if aa is not None else e.get("age_mean_ms"),
                                  e["n_inferences"], na))

for task in sorted({t for t, _ in rows}):
    act, act_ms, tick_ms, every = meta[task]
    print(f"\n{task}: actuation={act}, {act_ms:.1f} ms grid ({every} tick(s) of "
          f"{tick_ms:.2f} ms)")
    print(f"  {'arm':11s} {'age@act ms':>11s} {'disp/ep':>9s} {'act/ep':>8s} {'upd/act':>8s}")
    for arm in ARM_ORDER:
        v = rows.get((task, arm))
        if not v: continue
        ages = [x for x, _, _ in v if x is not None]
        disp = sum(y for _, y, _ in v) / len(v)
        acts = sum(z for _, _, z in v) / len(v)
        age = sum(ages) / len(ages) if ages else float("nan")
        print(f"  {arm:11s} {age:11.1f} {disp:9.1f} {acts:8.1f} {min(disp/acts,1.0):8.2f}")
print("\nupd/act is min(dispatches, actuations)/actuations: the fraction of control\n"
      "steps that get a NEW result. Capped at 1.0 -- extra cadence beyond the\n"
      "control rate is discarded by the zero-order hold.")
