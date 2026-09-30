#!/usr/bin/env bash
# The showdown at each runtime's MEASURED WORST-CASE control response.
#
# WHY WORST CASE. A real-time guarantee is a statement about the worst case, not the average:
# a control loop that holds 100 Hz for 65% of periods and stalls for 75 ms in the rest is not
# a 100 Hz loop for stability purposes. Both worst cases are measured on this board, with the
# real ModelBlaster kernels:
#
#   XPU-RT                                12.41 ms  max gap over 3 board traces (w4/b4/b5y)
#   ROS 2, default SingleThreadedExecutor 75.44 ms  max gap, n=601, real ROS 2 C++ nodes
#
# Through the simulator's tick rule, ceil(gap / 10 ms control step):
#   12.41 -> 2 ticks -> 50.0 Hz        75.44 -> 8 ticks -> 12.5 Hz
#
# THE SCOPE THIS CLAIM HAS, AND THE ONE IT DOES NOT. 75.44 ms is the DEFAULT single-threaded
# executor, which is what a ROS 2 node graph gets unless the author changes it. It is a real
# and common deployment. It is NOT the only one: with a MultiThreadedExecutor and spin(), the
# same graph holds control at 10.00 ms mean / 11.46 ms max at every camera rate up to 40 Hz --
# as good as XPU-RT. So this arm is "ROS 2 as configured by default", and the caption must say
# so. What survives the multi-threaded configuration is the PERCEPTION latency, 53.22 ms
# against 31.21 ms, measured end to end on the same kernels.
#
# Gain is fixed at 0.0055 on both arms so the only difference is control cadence, and that also
# places both flights on the 240-flight envelope grid rather than beside it.
set -u
. "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/scripts/env.sh"   # SIM_TREE, ISAAC_PY, RES: see scripts/env.local.sh.example
CAN="${CAN:-$SIM_TREE}"
PY="$ISAAC_PY"
W="$CAN/sims/models/warehouse/nav_fused_v12_cnn.pt"
OUT="${OUT:-$RES/worstcase}"
EPS="${EPS:-12}"
mkdir -p "$OUT/tmp"; export TMPDIR="$OUT/tmp"
cd "$CAN" || exit 1

fly() {  # $1=tag  $2=worst-case control response ms  $3=seed  $4=episodes
  echo "=== $(date +%H:%M:%S) $1  worst-case control response $2 ms  seed $3 x$4 ==="
  timeout 3000 "$PY" sims/scripts/sweep_rate_demo.py --headless --controller rl --weights "$W" \
    --sim_dt 0.01 --decimation 1 --obstacle_level 8 --prop_density 0.30 \
    --sched_latency_ms "$2" --percep_hold_ms 0 --moment_scale 0.0055 \
    --cruise_speed 1.4 --walk_speed 0.0 --episodes "$4" --seed "$3" --max_steps 1800 \
    --dump_figure_data "$OUT/${1}_figdata" --sweep-csv "$OUT/worstcase.csv" 2>&1 \
    | grep -E "\[SWEEP\]|\[sched\]|Traceback|Error"
  echo "${1}_DONE $(date +%H:%M:%S)"
}

fly ros 75.44 1000 "$EPS"   # baseline first: if it does not crash, nothing else matters
fly xpu 12.41 1000 "$EPS"
