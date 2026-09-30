#!/usr/bin/env bash
# The showdown against a baseline that leaves no hart idle.
#
# The unpinned 45 Hz baseline in warehouse_showdown_cam45_unpinned_best is ROS 2 as normally written,
# and its board trace shows YOLO on four harts, nav on one, control on one -- five of eight. That is
# a true measurement of the default deployment, but it invites the reply that the baseline was not
# given the machine.
#
# `vanilla4x2` is the same ROS 2 pattern scaled out: two YOLO node pools, `perception` on CPU_P#0-3
# and `perception2` on CPU_E#0-3, affinity mask 0xff on every node, nothing pinned. The sampler over
# its board run reads 52-58 % busy on ALL EIGHT harts, and at a 30 Hz camera it is late on none of
# 298 frames -- it meets the perception deadline it was given. Chain 31.44 ms against our 26.75.
#
# It still cannot fly the course, and the reason is the one the paper is about: ROS 2 chains control
# to the perception output, so its command rate is the camera rate, 30 Hz, under the control-rate
# floor. Ours schedules control in its own slot and commands at 100 Hz off the same 30 Hz camera.
# Nothing here is a handicap: the baseline uses every core and misses no deadline of its own.
#
# What was missing is a fair flight comparison. On disk, `ros_vanilla4x230` has 84 flights at the
# deployed gain 0.0055 and `xpu_p30free` has 144 at 0.005 -- different gains, so not a pair. This
# flies both arms at 0.0055, the gain every other figure in the set uses, on the display scene.
#
# Stage 1 is panel A's census, twelve episodes per arm. Stage 2 reports which seeds separate them,
# so a display pair can be flown from a disclosed population rather than searched for.
#
#   scripts/campaign_allcores30.sh              env CELL LAYOUT_SEED CRUISE EPISODES MAX_SIMS
set -u
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; R="$WT/results/codesign_feedback"
T="$R/ctrl_traces"; CELL="${CELL:-tall1000_ac30}"; LOG="$R/campaign_allcores30.log"
say(){ echo "=== $(date +%H:%M:%S) $* ===" | tee -a "$LOG"; }

say "stage 1: census, layout ${LAYOUT_SEED:-1000}, gain 0.0055, cruise ${CRUISE:-1.4}, 30 Hz camera"
CELL="$CELL" LAYOUT_SEED="${LAYOUT_SEED:-1000}" CRUISE="${CRUISE:-1.4}" GAIN=0.0055 \
  EPISODES="${EPISODES:-12}" MAX_SIMS="${MAX_SIMS:-3}" \
  ARMS="xpu_p30free:$T/xpu_p30free.csv:26.8:33.3 ros_vanilla4x230:$T/ros_vanilla4x230.csv:31.4:33.3" \
  bash "$WT/scripts/scene_runs_pair.sh" 2>&1 | tee -a "$LOG" | tail -6

say "stage 2: which episode seeds separate the two arms"
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
print(f"\n{'arm':22s}" + "".join(f"{s:>7}" for s in seeds) + f"{'completed':>11}")
for a in sorted(by):
    d = by[a]; s = sum(1 for v in d.values() if v[0] == "success")
    cells = "".join((f"{'OK'+str(d[x][1]):>7}" if x in d and d[x][0] == "success"
                     else (f"{'x'+str(d[x][1]):>7}" if x in d else f"{'-':>7}")) for x in seeds)
    print(f"{a:22s}{cells}{f'{s}/{len(d)}':>11}")
x = by.get("xpu_p30free", {}); r = by.get("ros_vanilla4x230", {})
sep = [s for s in seeds if x.get(s, ("",0))[0] == "success" and r.get(s, ("",0))[0] != "success"]
print("\nseeds where XPU-RT completes and the all-8-hart baseline does not:", sep or "none")
for s in sep:
    print(f"   seed {s}: baseline crashes after {r[s][1]} gate(s)")
PY
say "CAMPAIGN_ALLCORES30_DONE"
