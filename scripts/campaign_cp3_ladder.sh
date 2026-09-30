#!/usr/bin/env bash
# The pinned six-core baseline's own control-rate ladder.
#
# `cp3` is the deployment the Tier A showdown drew: a four-hart YOLO pool on harts 0-3, nav on
# hart 4, control on hart 5, harts 6 and 7 untouched. Because ROS 2 chains control to the
# perception output, its command rate is the camera rate -- until the pipeline saturates. The board
# runs already on disk say where that happens:
#
#   camera   15     25     30     45     60     75     90   Hz
#   control  15.0   25.0   29.9   38.6   39.0   39.1   39.1 Hz
#   cam->goal 30.8  30.7   30.7   56.7   56.2   56.1   56.0 ms
#
# Above a 45 Hz camera nothing moves: more frames buy no more commands. So this ladder flies the
# four rates that DO differ -- 15, 25, 30 and 45 Hz -- and stops there rather than burning flights
# on three cells that would land on top of each other.
#
# Every cell uses the deployed gain 0.0055 and a goal hold of exactly one camera period, so the
# only quantity that moves across the ladder is the rate at which commands reach the vehicle. That
# is what makes this a controlled sweep rather than a pooling of arms that differ in several ways
# at once.
#
#   scripts/campaign_cp3_ladder.sh          env SPEEDS SEEDS MAX_SIMS OUT
set -u
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
T="$WT/results/codesign_feedback/ctrl_traces"
OUT="${OUT:-$WT/results/codesign_feedback/campaign_cp3_ladder}"
mkdir -p "$OUT"

# name:trace:camera->control latency (ms):goal hold = one camera period (ms):gain
ARMLIST="${ARMLIST:-\
ros_cp3_15:$T/ros_cp315.csv:30.8:66.7:0.0055 \
ros_cp3_25:$T/ros_cp325.csv:30.7:40.0:0.0055 \
ros_cp3_30:$T/ros_cp330.csv:30.7:33.3:0.0055 \
ros_cp3_45:$T/ros_cp345.csv:56.2:22.2:0.0055}"

echo "=== $(date +%H:%M:%S) cp3 control-rate ladder: 4 camera rates x ${SPEEDS:-5} speeds x ${SEEDS:-12} seeds"
ARMLIST="$ARMLIST" SPEEDS="${SPEEDS:-1.0 1.2 1.4 1.6 1.8}" SEEDS="${SEEDS:-12}" \
  MAX_SIMS="${MAX_SIMS:-3}" OUT="$OUT" \
  bash "$WT/scripts/campaign_rate30_gain.sh"

.venv/bin/python - "$OUT/campaign.csv" <<'PY'
import collections, csv, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])) or ".", "scripts"))
sys.path.insert(0, "scripts")
from flight_quarantine import flight_rows
rows = flight_rows(sys.argv[1])
by = collections.defaultdict(list)
for r in rows:
    by[r["ctrl_trace"]].append(r)
print(f"\n{'trace':20s}{'ctrlHz':>8}{'n':>5}{'succ':>7}{'mean gates':>12}")
for k, v in sorted(by.items(), key=lambda kv: sum(float(r["eff_cmd_hz"]) for r in kv[1]) / len(kv[1])):
    g = [float(r["gates_passed"]) for r in v]
    s = sum(1 for r in v if r["outcome"] == "success")
    hz = sum(float(r["eff_cmd_hz"]) for r in v) / len(v)
    print(f"{k.replace('.csv',''):20s}{hz:>8.1f}{len(v):>5}{s:>4}/{len(v):<3d}{sum(g)/len(g):>10.2f}")
PY
echo "CAMPAIGN_CP3_LADDER_DONE $(date +%H:%M:%S)"
