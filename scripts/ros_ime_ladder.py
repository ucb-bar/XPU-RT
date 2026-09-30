#!/usr/bin/env python3
"""The ROS-with-IME ladder: what blanket IME kernels cost or buy a ROS 2 deployment.

Reads the traced ROS runs under results/codesign_feedback/ros_traced/ (pulled by
scripts/ros_traced_matrix.sh, summarised by scripts/pull_ros_traced.py) and prints
one row per arm, aggregated across replicates. Writes the same numbers as JSON so a
document can quote them without re-deriving anything.

The arms, and why each exists, are in docs/Baselines/ros_with_ime.md. An arm whose second
perception process died (the naive IME deployment: smt.vmadot raises SIGILL on
cluster 1) is not a failed measurement -- it is the result -- so it is reported by
what survived: how many of the released frames ever produced a goal.

"frames late" is counted here the way a 30 Hz camera makes it matter: a released
frame whose goal reaches the control node more than one frame period after its
release is late, because by then the next frame is already out. Frames with no
goal at all are counted separately as dropped, never folded into the median.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import statistics

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.join(REPO, "results/codesign_feedback/ros_traced")

#: arm tag prefix -> (label, what the placement is)
ARMS = [
    ("30_vanilla4x2_r",      "RVV kernels, both clusters (8 harts)"),
    ("30_vanilla4x2_ime_r",  "IME kernels, both clusters (8 harts) -- the naive deployment"),
    ("30_vanilla4x2c0_rvv_r", "RVV kernels, cluster 0 only (4 harts)"),
    ("30_vanilla4x2c0_ime_r", "IME kernels, cluster 0 only (4 harts)"),
]


def run_dirs(prefix):
    return sorted(d for d in glob.glob(os.path.join(ROOT, prefix + "*")) if os.path.isdir(d))


def late_and_dropped(d, rate_hz, warmup_ms=3000.0, ticks_per_ms=24000.0):
    """(n_late, n_with_goal, n_dropped) over the post-warm-up frames of one run."""
    path = os.path.join(d, "chain.csv")
    if not os.path.exists(path):
        return None
    man = json.load(open(os.path.join(d, "manifest.json")))
    warm = int(man.get("run_t0_ticks") or 0) + warmup_ms * ticks_per_ms
    period = 1000.0 / rate_hz
    late = have = drop = 0
    for r in csv.DictReader(open(path)):
        if int(r["t_release_ticks"]) < warm:
            continue
        if r["e2e_goal_ms"] == "":
            drop += 1
            continue
        have += 1
        late += float(r["e2e_goal_ms"]) > period
    return late, have, drop


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(
        REPO, "results/codesign_feedback/ros_with_ime/ladder.json"))
    a = ap.parse_args()

    out = []
    for prefix, label in ARMS:
        dirs = run_dirs(prefix)
        if not dirs:
            print(f"  (no runs for {prefix}*)")
            continue
        per = []
        for d in dirs:
            s = json.load(open(os.path.join(d, "summary.json")))
            ld = late_and_dropped(d, s["rate_hz"])
            per.append({"tag": s["tag"], "gap_mean_ms": s["gap_mean_ms"], "gap_max_ms": s["gap_max_ms"],
                        "e2e_goal_med_ms": s["e2e_goal_med_ms"], "e2e_goal_p95_ms": s["e2e_goal_p95_ms"],
                        "n_frames": s["n_frames"], "n_goals": s["n_goals"],
                        "busy_pct": s["busy_pct"],
                        "n_late": ld[0] if ld else None, "n_with_goal": ld[1] if ld else None,
                        "n_dropped": ld[2] if ld else None})
        med = lambda k: statistics.median([p[k] for p in per if p[k] is not None])
        busy = {c: round(statistics.median([p["busy_pct"].get(str(c), p["busy_pct"].get(c, 0.0))
                                            for p in per]), 1) for c in range(8)}
        row = {"arm": prefix.rstrip("_r").rstrip("_"), "label": label, "n_replicates": len(per),
               "gap_mean_ms": round(med("gap_mean_ms"), 2), "gap_max_ms": round(med("gap_max_ms"), 2),
               "control_hz": round(1000.0 / med("gap_mean_ms"), 2),
               "e2e_goal_med_ms": round(med("e2e_goal_med_ms"), 2),
               "e2e_goal_p95_ms": round(med("e2e_goal_p95_ms"), 2),
               "frames": int(med("n_frames")), "goals": int(med("n_goals")),
               "late": int(med("n_late")), "with_goal": int(med("n_with_goal")),
               "dropped": int(med("n_dropped")), "busy_pct": busy, "runs": per}
        out.append(row)
        print(f"{row['label']:<58} ctrl {row['gap_mean_ms']:6.2f} ms ({row['control_hz']:5.2f} Hz)  "
              f"cam->goal med {row['e2e_goal_med_ms']:6.2f} p95 {row['e2e_goal_p95_ms']:6.2f}  "
              f"late {row['late']}/{row['with_goal']}  dropped {row['dropped']}  "
              f"busy " + " ".join(f"c{c}:{v:.0f}" for c, v in busy.items()))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=2)
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
