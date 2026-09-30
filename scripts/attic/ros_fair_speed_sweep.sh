#!/usr/bin/env bash
# Give the baseline its best shot: at what cruise speed can real-ROS-2 latency fly the course?
#
# WHY. At 1.4 m/s the 53.21 ms arm reaches 0 of 4 gates in 12/12 seeds. A baseline that never
# reaches a single gate invites the reading that it was crippled, and that reading is worth
# pre-empting rather than arguing with. The envelope already says outcome depends on speed as
# much as on rate, so the fair question is not "does ROS crash" but "how slowly must ROS fly
# before it can hold the course, and does the global schedule have to slow down too".
#
# Both arms at their MEASURED chain latency, gain fixed at 0.0055 on both, speed swept:
#   XPU-RT       29.93 ms  (global schedule, board-executed)
#   real ROS 2   53.21 ms  (35.58 static-pin chain + 17.63 measured middleware tax;
#                           confirmed directly at 53.22 ms by the on-board chain sweep)
#
# Slow speeds first, and the baseline before the other arm at each speed, so an interrupted
# run still answers the question that motivated it.
set -u
. "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/scripts/env.sh"   # SIM_TREE, ISAAC_PY, RES: see scripts/env.local.sh.example
CAN="${CAN:-$SIM_TREE}"
PY="$ISAAC_PY"
W="$CAN/sims/models/warehouse/nav_fused_v12_cnn.pt"
OUT="${OUT:-$RES/fair_speed}"
EPS="${EPS:-6}"
mkdir -p "$OUT/tmp"; export TMPDIR="$OUT/tmp"
cd "$CAN" || exit 1

fly() {  # $1=tag $2=latency $3=cruise
  echo "=== $(date +%H:%M:%S) $1 lat=$2 cruise=$3 ==="
  timeout 3000 $PY sims/scripts/sweep_rate_demo.py --headless --controller rl --weights "$W" \
    --sim_dt 0.01 --decimation 1 --obstacle_level 8 --prop_density 0.30 \
    --sched_latency_ms "$2" --percep_hold_ms 0 --moment_scale 0.0055 \
    --cruise_speed "$3" --walk_speed 0.0 --episodes "$EPS" --seed 1000 --max_steps 2400 \
    --sweep-csv "$OUT/fair_speed.csv" 2>&1 | grep -E "\[SWEEP\]|\[sched\]|Traceback|Error"
}

for CRU in 0.6 0.8 1.0 1.2; do
  fly ros 53.21 "$CRU"
  fly xpu 29.93 "$CRU"
done
echo "FAIR_SPEED_DONE $(date +%H:%M:%S)"
