#!/usr/bin/env bash
# The third rung of the ROS 2 effort ladder: the all-eight-hart baseline with a worker pool for the
# NAVIGATION network as well, so every network in the graph is parallel.
#
# `vanilla4x2` already runs two four-hart YOLO pools. Its nav ran on one hart, not by choice but
# because the nav build on the board had its parallel convolutions compiled out (the shard is an
# output-channel split and this backend packs weights IHWOC). Rebuilt from the shard-4 nav and
# armed, nav's convolutions now run on a four-worker pool -- the board trace shows its slices on
# three and four harts at a time.
#
# On the board it costs rather than saves: chain 37.4 ms against 32.6 ms, nav's own callback 5.76 ms
# against 4.15 ms, YOLO's 29.93 against 25.89. The machine has no spare harts, so a third pool takes
# its threads from the two that were already there. Alone on an idle board the same nav model runs
# 1.66x faster on four harts than on one, so the kernel is not the limit -- the absence of anything
# coordinating the three pools is.
#
# This flies it, on the same scene, layout, cruise and gain as the other two arms in the cell, so
# panel A can carry all three as one population.
#
#   scripts/campaign_navpool36.sh              env CELL LAYOUT_SEED CRUISE EPISODES MAX_SIMS
set -u
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; R="$WT/results/codesign_feedback"
T="$R/ctrl_traces"; CELL="${CELL:-tall1000_ac36}"; LOG="$R/campaign_navpool36.log"
say(){ echo "=== $(date +%H:%M:%S) $* ===" | tee -a "$LOG"; }

say "census: the nav-pool rung, layout ${LAYOUT_SEED:-1000}, gain 0.0055, cruise ${CRUISE:-1.4}, 36 Hz"
CELL="$CELL" LAYOUT_SEED="${LAYOUT_SEED:-1000}" CRUISE="${CRUISE:-1.4}" GAIN=0.0055 \
  EPISODES="${EPISODES:-12}" MAX_SIMS="${MAX_SIMS:-3}" \
  ARMS="ros_vanilla4x236ns4:$T/ros_vanilla4x236ns4.csv:37.4:27.8" \
  bash "$WT/scripts/scene_runs_pair.sh" 2>&1 | tee -a "$LOG" | tail -4

say "the three rungs on this scene"
.venv/bin/python - "$R/campaign_scene/$CELL" <<'PY' 2>&1 | tee -a "$LOG"
import collections, glob, os, sys
sys.path.insert(0, "scripts")
from flight_quarantine import flight_rows
rows = []
for f in sorted(glob.glob(os.path.join(sys.argv[1], "**", "*.csv"), recursive=True)):
    try: rows += flight_rows(f)
    except Exception as e: print("skipped", f, e)
by = collections.defaultdict(dict)
for r in rows:
    by[r["ctrl_trace"].replace(".csv", "")][int(r["seed"])] = (r["outcome"], int(float(r["gates_passed"])))
seeds = sorted({s for d in by.values() for s in d})
print(f"\n{'arm':24s}" + "".join(f"{s:>7}" for s in seeds) + f"{'done':>8}{'mean gates':>12}")
for a in sorted(by):
    d = by[a]; s = sum(1 for v in d.values() if v[0] == "success")
    mg = sum(v[1] for v in d.values()) / max(1, len(d))
    cells = "".join((f"{'OK'+str(d[x][1]):>7}" if x in d and d[x][0] == "success"
                     else (f"{'x'+str(d[x][1]):>7}" if x in d else f"{'-':>7}")) for x in seeds)
    print(f"{a:24s}{cells}{f'{s}/{len(d)}':>8}{mg:>12.2f}")
x = by.get("xpu_p36free", {}); n = by.get("ros_vanilla4x236ns4", {})
sep = [s for s in seeds if x.get(s, ("",0))[0] == "success" and n.get(s, ("",0))[0] != "success"]
print("\nseeds where XPU-RT completes and the nav-pool baseline does not:", sep or "none")
PY
say "CAMPAIGN_NAVPOOL36_DONE"
