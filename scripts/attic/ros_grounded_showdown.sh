#!/usr/bin/env bash
# The showdown flights, at latencies grounded in MEASUREMENT rather than in a model.
#
# WHAT CHANGED. The previous pair used --sched_latency_ms 4.89 for XPU-RT and 35.58 for the
# baseline. Those are different quantities -- 4.89 ms is a worst-response, 35.58 ms is a
# camera->control chain end-to-end -- so the arms were not comparable. And 35.58 ms is the
# static-pin POLICY MODEL, which charges nothing for middleware.
#
# Both arms here report the SAME quantity, chain end-to-end latency, and the baseline carries
# the middleware cost we measured on this board:
#
#   XPU-RT   29.93 ms  = cmp_coupled_cpsat_board end-to-end, board-recost and board-executed
#   ROS      53.21 ms  = 35.58 (static-pin chain E2E) + 17.63 (MEASURED ROS 2 Jazzy tax over
#                        3 hops, single-threaded executor -- the CHEAPER of the two measured
#                        executors; multi-threaded is 39.73 ms, so this is the arm that
#                        flatters ROS)
#
# Through the simulator's own refresh rule, ceil(latency / 10 ms control step):
#   29.93 -> 3 steps -> 33.3 Hz        53.21 -> 6 steps -> 16.7 Hz
#
# GAIN IS HELD FIXED AT 0.0055 ON BOTH ARMS, deliberately. The calibrated law
# (moment_scale = 0.5/eff_hz) would give the two arms different control authority, and a
# difference in outcome could then be attributed to gain rather than to latency. Holding it
# fixed means the ONLY thing that differs between these two flights is the schedule's
# end-to-end latency, which is the claim. It also matches the 240-flight envelope, so these
# flights sit on that grid rather than beside it.
set -u
. "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/scripts/env.sh"   # SIM_TREE, ISAAC_PY, RES: see scripts/env.local.sh.example
CAN="${CAN:-$SIM_TREE}"
PY="$ISAAC_PY"
W="$CAN/sims/models/warehouse/nav_fused_v12_cnn.pt"
OUT="${OUT:-$RES/ros_grounded}"
EPS="${EPS:-12}"
mkdir -p "$OUT/tmp"; export TMPDIR="$OUT/tmp"
cd "$CAN" || exit 1

fly() {  # $1=tag $2=latency_ms
  echo "=== $(date +%H:%M:%S) $1  sched_latency=$2 ms ==="
  timeout 3000 $PY sims/scripts/sweep_rate_demo.py --headless --controller rl --weights "$W" \
    --sim_dt 0.01 --decimation 1 --obstacle_level 8 --prop_density 0.30 \
    --sched_latency_ms "$2" --percep_hold_ms 0 --moment_scale 0.0055 \
    --cruise_speed 1.4 --walk_speed 0.0 --episodes "$EPS" --seed 1000 --max_steps 1800 \
    --dump_figure_data "$OUT/${1}_figdata" --sweep-csv "$OUT/ros_grounded.csv" 2>&1 \
    | grep -E "\[SWEEP\]|\[sched\]|Traceback|Error"
  echo "${1}_DONE $(date +%H:%M:%S)"
}

fly xpu 29.93
fly ros 53.21
echo "ROS_GROUNDED_DONE $(date +%H:%M:%S)"
