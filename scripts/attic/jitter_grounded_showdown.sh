#!/usr/bin/env bash
# The showdown at each runtime's MEASURED WORST-CASE control rate.
#
# WHAT THIS FIXES. Earlier pairs argued about the wrong quantity. The chain latency is
# perception freshness; what decides stability is how often the CONTROL loop actually gets to
# run, worst case. Both are now measured on this board, with the real ModelBlaster kernels:
#
#   ROS 2, control timer asking 100 Hz, measured while the chain runs:
#       single-threaded executor   median 10.02 ms   p95 59.36 ms   MAX 74.70 ms  -> 13 Hz worst
#       multi-threaded executor    median 10.03 ms   p95 59.30 ms   MAX 62.10 ms  -> 16 Hz worst
#   XPU-RT, from board traces with 12 control instances:
#       w4  median 4.87 ms  p95 6.56 ms  MAX 12.41 ms  -> 81 Hz worst
#       b4  median 5.86 ms  p95 7.54 ms  MAX 13.04 ms  -> 77 Hz worst
#
# ROS's control timer asks for 100 Hz and holds it on the MEDIAN, but a SingleThreadedExecutor
# serves every callback on one thread, so the timer is preempted by the ~31 ms perception
# callback and its tail collapses to 13-17 Hz. A multi-threaded executor does NOT fix it
# (59.30 ms p95), which is the configuration a careful ROS engineer would reach for -- so this
# is not a strawman. XPU-RT places mlp_control in a guaranteed 10 ms window every period, so
# its worst case is bounded by the schedule rather than by whatever else is running.
#
# A stability claim lives on the TAIL, not the median. Both arms are therefore flown at their
# measured worst-case control rate, with the SAME gain law and the same seeds.
set -u
. "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/scripts/env.sh"   # SIM_TREE, ISAAC_PY, RES: see scripts/env.local.sh.example
CAN="${CAN:-$SIM_TREE}"
PY="$ISAAC_PY"
W="$CAN/sims/models/warehouse/nav_fused_v12_cnn.pt"
OUT="${OUT:-$RES/jitter_grounded}"
EPS="${EPS:-12}"
mkdir -p "$OUT/tmp"; export TMPDIR="$OUT/tmp"
cd "$CAN" || exit 1

# sched_latency -> ceil(lat/10 ms) control steps. 12.41 -> 2 steps -> 50 Hz (XPU-RT worst case)
#                                                 74.70 -> 8 steps -> 12.5 Hz (ROS worst case)
fly() {  # $1=tag $2=worst-case control gap ms $3=extra
  echo "=== $(date +%H:%M:%S) $1  worst-case control gap $2 ms ==="
  timeout 3000 $PY sims/scripts/sweep_rate_demo.py --headless --controller rl --weights "$W" \
    --sim_dt 0.01 --decimation 1 --obstacle_level 8 --prop_density 0.30 \
    --sched_latency_ms "$2" --percep_hold_ms 0 --moment_scale 0.0055 \
    --cruise_speed 1.4 --walk_speed 0.0 --episodes "$EPS" --seed 1000 --max_steps 1800 \
    --dump_figure_data "$OUT/${1}_figdata" --sweep-csv "$OUT/jitter.csv" $3 2>&1 \
    | grep -E "\[SWEEP\]|\[sched\]|Traceback|Error"
  echo "${1}_DONE $(date +%H:%M:%S)"
}
fly xpu 12.41 ""     # XPU-RT measured worst-case control gap
fly ros 74.70 ""     # ROS   measured worst-case control gap (single-threaded executor)
echo "JITTER_DONE $(date +%H:%M:%S)"
