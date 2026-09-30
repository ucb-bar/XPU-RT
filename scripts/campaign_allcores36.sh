#!/usr/bin/env bash
# The all-eight-hart showdown at the camera rate where the baseline flies into the course.
#
# Same baseline as scripts/campaign_allcores30.sh -- `vanilla4x2`, two ROS 2 YOLO node pools
# (`perception` on CPU_P#0-3, `perception2` on CPU_E#0-3), affinity 0xff, nothing pinned, the
# sampler reading 28-55 % busy on all eight harts over its board run. At a 36 Hz camera it is
# late on none of its frames and its chain lands at 32.29 ms inside a 33.3 ms end-to-end budget:
# a baseline given the whole machine, meeting the deadline it was given.
#
# Its command rate is still the camera rate, because ROS 2 chains control to the perception
# output: 27.67 ms between commands, 36.1 Hz. The XPU-RT arm `p36free` runs the same three
# networks from a placement-free CP-SAT solve (no hart is named in the spec; the solver assigns
# all three networks across all eight harts), chain 25.82 ms, and fires control in its own slot
# every 9.86 ms -- 101 Hz off the same 36 Hz camera.
#
# 30 Hz separates the arms harder but makes a worse figure: the baseline falls over before the
# first gate, so the flight panel shows nothing being attempted. At 36 Hz the baseline enters the
# course and loses it in 80 % of flights, which is the comparison worth drawing.
#
# Stage 1 is panel A's census, twelve episodes per arm. Stage 2 reports which seeds separate them,
# so the display pair is drawn from a disclosed population rather than searched for.
#
#   scripts/campaign_allcores36.sh              env CELL LAYOUT_SEED CRUISE EPISODES MAX_SIMS
set -u
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; R="$WT/results/codesign_feedback"
T="$R/ctrl_traces"; CELL="${CELL:-tall1000_ac36}"; LOG="$R/campaign_allcores36.log"
say(){ echo "=== $(date +%H:%M:%S) $* ===" | tee -a "$LOG"; }

say "stage 1: census, layout ${LAYOUT_SEED:-1000}, gain 0.0055, cruise ${CRUISE:-1.4}, 36 Hz camera"
CELL="$CELL" LAYOUT_SEED="${LAYOUT_SEED:-1000}" CRUISE="${CRUISE:-1.4}" GAIN=0.0055 \
  EPISODES="${EPISODES:-12}" MAX_SIMS="${MAX_SIMS:-3}" \
  ARMS="xpu_p36free:$T/xpu_p36free.csv:25.8:27.8 ros_vanilla4x236:$T/ros_vanilla4x236.csv:32.3:27.8" \
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
x = by.get("xpu_p36free", {}); r = by.get("ros_vanilla4x236", {})
sep = [s for s in seeds if x.get(s, ("",0))[0] == "success" and r.get(s, ("",0))[0] != "success"]
print("\nseeds where XPU-RT completes and the all-8-hart baseline does not:", sep or "none")
for s in sep:
    print(f"   seed {s}: baseline crashes after {r[s][1]} gate(s)")
PY
say "CAMPAIGN_ALLCORES36_DONE"
