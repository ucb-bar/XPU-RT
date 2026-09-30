#!/usr/bin/env bash
# The showdown on PERCEPTION LATENCY, modelled fully: a chain of latency L delivers a goal that
# is L old AND refreshes only every L. Earlier runs applied only the refresh rate
# (--percep_hold_ms) and left the delay at zero, which understated both arms and understated
# the baseline more, because its L is larger.
#
# Control is 100 Hz on BOTH arms and that is now measured, not assumed: with a
# MultiThreadedExecutor and spin(), ROS holds its control timer at 10.00 ms mean / 11.46 ms max
# at every camera rate up to 40 Hz -- as good as XPU-RT's 6.30 mean / 12.41 max. So control
# rate is NOT the differentiator, and any flight difference here comes from perception
# freshness alone.
#
# The two latencies are the measured chain times on the K1, same ModelBlaster kernels, same
# board, only the orchestrator differing:
#
#   XPU-RT  31.21 ms  (8 harts, width chosen per dispatch, executed on the board)
#   ROS     53.22 ms  (real ROS 2 C++ nodes running the same kernels, measured end-to-end;
#                      it also caps at 20 Hz -- at 25 Hz its chain diverges to 92.8 ms
#                      against a 40 ms period)
#
# Swept over cruise speed because staleness is a DISTANCE, not a time: at 1.4 m/s the two arms
# differ by 4.4 cm of travel per update, at 2.0 m/s by 6.3 cm. If the effect is real it should
# grow with speed, and if it does not, that is the answer.
set -u
. "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/scripts/env.sh"   # SIM_TREE, ISAAC_PY, RES: see scripts/env.local.sh.example
CAN="${CAN:-$SIM_TREE}"
PY="$ISAAC_PY"
W="$CAN/sims/models/warehouse/nav_fused_v12_cnn.pt"
OUT="${OUT:-$RES/percep_latency}"
EPS="${EPS:-12}"
mkdir -p "$OUT/tmp"; export TMPDIR="$OUT/tmp"
cd "$CAN" || exit 1

fly() {  # $1=tag  $2=chain latency ms  $3=cruise
  echo "=== $(date +%H:%M:%S) $1  chain $2 ms (delay AND refresh)  cruise $3 ==="
  timeout 3000 "$PY" sims/scripts/sweep_rate_demo.py --headless --controller rl --weights "$W" \
    --sim_dt 0.01 --decimation 1 --obstacle_level 8 --prop_density 0.30 \
    --sched_latency_ms 0 --percep_latency_ms "$2" --percep_hold_ms "$2" \
    --moment_scale 0.0055 --cruise_speed "$3" --walk_speed 0.0 \
    --episodes "$EPS" --seed 1000 --max_steps 1800 \
    --dump_figure_data "$OUT/${1}_figdata" --sweep-csv "$OUT/percep.csv" 2>&1 \
    | grep -E "\[SWEEP\]|\[percep|Traceback|Error"
  echo "${1}_DONE $(date +%H:%M:%S)"
}

# higher speed first: if staleness matters at all it shows there, and if the fast pair is null
# the slow pair will be too, so an interrupted run still answers the question.
fly ros_18 53.22 1.8
fly xpu_18 31.21 1.8
fly ros_14 53.22 1.4
fly xpu_14 31.21 1.4
echo "PERCEP_DONE $(date +%H:%M:%S)"
