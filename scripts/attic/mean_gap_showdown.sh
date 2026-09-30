#!/usr/bin/env bash
# The showdown at each runtime's MEAN MEASURED control gap -- the same statistic on both arms.
#
# WHY THE MEAN AND NOT THE MEDIAN OR THE MAX. ROS's control-gap distribution is BIMODAL, and no
# single order statistic describes it honestly on its own (n=801, real ROS 2 C++ nodes running
# the real ModelBlaster kernels on the K1, control timer asking for 100 Hz):
#
#   med 10.06 | p60 10.19 | p70 52.52 | p80 55.85 | p90 59.17 | p95 59.28 | max 74.36 | MEAN 24.96
#
# About 65% of control periods are the 10 ms the timer asked for; the other 35% are 52-59 ms,
# because a SingleThreadedExecutor cannot fire the control timer while the ~31 ms perception
# callback is running. Quoting the median (10 ms) hides the starvation entirely; quoting the max
# (74 ms) describes only the worst period. The MEAN is the average control period the loop
# actually delivers, it is one number, and it is the SAME statistic on both arms:
#
#   XPU-RT   6.30 ms mean gap (pooled over the w4/b4/b5y board traces)  -> 100 Hz in the sim
#   ROS     24.96 ms mean gap (measured as above)                       -> 33.3 Hz in the sim
#
# XPU-RT's is small because the schedule places mlp_control in a guaranteed 10 ms window every
# period, so nothing else can occupy its core inside that window. A MultiThreadedExecutor does
# not close the gap for ROS (mean 24.92 ms, p95 59.41), so this is not an artifact of choosing
# the naive executor.
#
# WHAT IS MEASURED AND WHAT IS SIMULATED. The TIMING is measured on the K1. The FLIGHT is an
# Isaac Lab simulation that consumes it: --sched_latency_ms makes the simulated controller act
# every ceil(lat/10 ms) steps, so the drone flies with the control cadence the board actually
# delivers. This shows the flight CONSEQUENCE of a measured control cadence; it is not a real
# drone, and the caption must say so.
set -u
. "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/scripts/env.sh"   # SIM_TREE, ISAAC_PY, RES: see scripts/env.local.sh.example
CAN="${CAN:-$SIM_TREE}"
PY="$ISAAC_PY"
W="$CAN/sims/models/warehouse/nav_fused_v12_cnn.pt"
OUT="${OUT:-$RES/mean_gap}"
EPS="${EPS:-12}"
mkdir -p "$OUT/tmp"; export TMPDIR="$OUT/tmp"
cd "$CAN" || exit 1

fly() {  # $1=tag  $2=mean control gap ms  $3=seed base  $4=episodes
  echo "=== $(date +%H:%M:%S) $1  mean control gap $2 ms  seed $3 x$4 ==="
  timeout 3000 "$PY" sims/scripts/sweep_rate_demo.py --headless --controller rl --weights "$W" \
    --sim_dt 0.01 --decimation 1 --obstacle_level 8 --prop_density 0.30 \
    --sched_latency_ms "$2" --percep_hold_ms 0 --moment_scale 0.0055 \
    --cruise_speed 1.4 --walk_speed 0.0 --episodes "$4" --seed "$3" --max_steps 1800 \
    --dump_figure_data "$OUT/${1}_figdata" --sweep-csv "$OUT/mean_gap.csv" 2>&1 \
    | grep -E "\[SWEEP\]|\[sched\]|Traceback|Error"
  echo "${1}_DONE $(date +%H:%M:%S)"
}

fly xpu 6.30  1000 "$EPS"
fly ros 24.96 1000 "$EPS"
# One extra baseline flight on a single seed that crosses gates and then crashes, so panel A
# draws a CRASH. The sweep's own dump rule is "first success, else deepest crash", which picked
# a SUCCESSFUL baseline flight last time and drew a completed trajectory with a crash marker.
fly ros_crash 24.96 1002 1
echo "MEANGAP_DONE $(date +%H:%M:%S)"
