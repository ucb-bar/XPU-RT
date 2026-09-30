#!/usr/bin/env bash
# NEW matched showdown flights, both at DERIVED control rates in the clean-crash zone.
#  XPU-RT: sched_latency 4.89 ms (certified sharded worst-response) -> 100 Hz, gain 0.005 -> flies.
#  ROS:    sched_latency 35.58 ms (coupled-chain camera->control E2E, deps-honored) -> 25 Hz, gain 0.02 -> crashes.
# Matched scene: rl_controller_velctrl_dr4.pt, cruise 1.4, decim 1, obstacle_level 8, prop_density 0.30,
# walk 0.0, seed base 1000, 6 eps. Sequential (one Isaac at a time). Dumps keep first-success-else-deepest-crash.
set -u
cd /scratch/agustin/projects/DIMA/XPU-RT
PY=/scratch2/agustin/miniforge3/envs/env_isaaclab/bin/python
CV=/scratch/agustin/xpurt-dev-sync/results/codesign_feedback/crash_verify
run() {  # $1=tag $2=lat $3=gain $4=tmp
  export TMPDIR=/scratch/agustin/tmp/claude-2621/-scratch-agustin-projects-DIMA/057226a3-598b-40aa-8396-ef0c5c742cd9/scratchpad/$4
  mkdir -p "$TMPDIR"
  echo ">>> $1 lat=$2 gain=$3 $(date +%H:%M:%S)"
  stdbuf -oL -eL $PY sims/scripts/sweep_rate_demo.py --headless --controller rl \
    --weights sims/models/warehouse/nav_fused_v12_cnn.pt \
    --sim_dt 0.01 --decimation 1 --obstacle_level 8 --prop_density 0.30 \
    --sched_latency_ms "$2" --percep_hold_ms 0 --moment_scale "$3" \
    --cruise_speed 1.4 --walk_speed 0.0 --episodes 6 --seed 1000 --max_steps 1800 \
    --dump_figure_data "$CV/${1}_figdata" --sweep-csv "$CV/${1}.csv" 2>&1 \
    | stdbuf -oL grep -E "\[SWEEP\]|\[ep0|\[figure\] wrote|\[sched\]|Traceback|Error|CUDA"
  echo "${1}_DONE $(date +%H:%M:%S)"
}
run new_xpu 4.89 0.005 new_xpu_tmp
run new_ros 35.58 0.02 new_ros_tmp
echo "ALL_NEWSHOWDOWN_DONE"
