#!/usr/bin/env bash
# G0: does perception-side backlog alone bring the drone down?
#
# A ROS 2 deployment that gives control its own core holds the 100 Hz timer, so the only way it
# loses the course is through the goal it acts on: with the default KEEP_LAST(10) queue, a
# camera that outruns the perception chain leaves the controller steering on a goal that is
# several frames old. This gate flies that situation with control at XPU-RT's measured cadence
# on BOTH arms, and varies only how stale the goal is:
#
#   xpu    delay 36 ms, refresh 40 ms   the global schedule keeps up with a 25 Hz camera
#   b150 / b300 / b500                  the goal is 150 / 300 / 500 ms old, refreshed every 57 ms
#
# Same gain on every arm, so the outcome is attributable to staleness and nothing else.
set -u
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; . "$WT/scripts/env.sh"
CAN="${CAN:-$SIM_TREE}"
PY="$ISAAC_PY"
W="$CAN/sims/models/warehouse/nav_fused_v12_cnn.pt"
OUT="${OUT:-$RES/g0_backlog}"
EPS="${EPS:-12}"
mkdir -p "$OUT/tmp"; export TMPDIR="$OUT/tmp"
cd "$CAN" || exit 1

fly() {  # $1=tag  $2=perception delay ms  $3=perception refresh ms
  echo "=== $(date +%H:%M:%S) $1  goal age $2 ms  refresh $3 ms ==="
  timeout 3000 "$PY" sims/scripts/sweep_rate_demo.py --headless --controller rl --weights "$W" \
    --sim_dt 0.01 --decimation 1 --obstacle_level 8 --prop_density 0.30 \
    --sched_latency_ms 6.30 --percep_latency_ms "$2" --percep_hold_ms "$3" --moment_scale 0.0055 \
    --cruise_speed 1.4 --walk_speed 0.0 --episodes "$EPS" --seed 1000 --max_steps 1800 \
    --dump_figure_data "$OUT/${1}_figdata" --sweep-csv "$OUT/g0.csv" 2>&1 \
    | grep -E "\[SWEEP\]|\[percep|\[sched\]|Traceback|Error"
  echo "${1}_DONE $(date +%H:%M:%S)"
}

fly b500 500 57      # the deepest backlog first: if it does not crash, the shallower ones will not
fly b300 300 57
fly b150 150 57
fly xpu   36 40
echo "G0_DONE $(date +%H:%M:%S)"
