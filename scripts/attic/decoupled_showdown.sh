#!/usr/bin/env bash
# The FAIR comparison: control at full rate on both arms, perception staleness is what differs.
#
# WHY THIS REPLACES THE EARLIER PAIR. Flying the baseline at --sched_latency_ms 53.21 said its
# CONTROL loop runs at 16.7 Hz. That is not what a ROS deployment does. mlp_control is 0.08 ms
# and has its own hart in the static partition, so it holds its own 100 Hz timer and acts on
# the LAST AVAILABLE nav goal. What the chain latency actually costs the baseline is perception
# STALENESS, not control rate -- and the simulator models exactly that with --percep_hold_ms,
# which holds the nav goal at the perception cadence while the command stays at full rate.
# Modelling it as a slow control loop understated the baseline, which is the opposite of the
# error worth making.
#
# Three arms, all at 100 Hz control, all at gain 0.0055, differing only in perception cadence:
#
#   XPU-RT           35.71 ms   measured warm chain (yolo executed 31.21 + nav 4.43 + ctrl 0.08)
#   ROS best case    44.12 ms   every advantage we can measurably grant it: perception node on
#                               all 8 harts (measured 1.42x sharding), plus the CHEAPER of the
#                               two measured executors (17.63 ms, single-threaded)
#   ROS as-measured  53.22 ms   the configuration actually measured end-to-end on the board
#
# Best case first: if the baseline cannot clear a gate even there, no configuration we can
# reach will, and the result is not a strawman.
set -u
. "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/scripts/env.sh"   # SIM_TREE, ISAAC_PY, RES: see scripts/env.local.sh.example
CAN="${CAN:-$SIM_TREE}"
PY="$ISAAC_PY"
W="$CAN/sims/models/warehouse/nav_fused_v12_cnn.pt"
OUT="${OUT:-$RES/decoupled}"
EPS="${EPS:-12}"
mkdir -p "$OUT/tmp"; export TMPDIR="$OUT/tmp"
cd "$CAN" || exit 1

fly() {  # $1=tag  $2=perception cadence ms
  echo "=== $(date +%H:%M:%S) $1  percep_hold=$2 ms  (control 100 Hz) ==="
  timeout 3000 $PY sims/scripts/sweep_rate_demo.py --headless --controller rl --weights "$W" \
    --sim_dt 0.01 --decimation 1 --obstacle_level 8 --prop_density 0.30 \
    --sched_latency_ms 0 --percep_hold_ms "$2" --moment_scale 0.0055 \
    --cruise_speed 1.4 --walk_speed 0.0 --episodes "$EPS" --seed 1000 --max_steps 1800 \
    --dump_figure_data "$OUT/${1}_figdata" --sweep-csv "$OUT/decoupled.csv" 2>&1 \
    | grep -E "\[SWEEP\]|\[percep|\[sched\]|Traceback|Error"
  echo "${1}_DONE $(date +%H:%M:%S)"
}

fly ros_best 44.12
fly xpu      35.71
fly ros      53.22
echo "DECOUPLED_DONE $(date +%H:%M:%S)"
