#!/usr/bin/env bash
# SAME-SEED matched showdown dumps (seed 1004: XPU 4/4 success, ROS clip-crash) -> identical obstacle layout.
set -u
cd /scratch/agustin/projects/DIMA/XPU-RT
PY=/scratch2/agustin/miniforge3/envs/env_isaaclab/bin/python
CV=/scratch/agustin/xpurt-dev-sync/results/codesign_feedback/crash_verify
run() { # $1=tag $2=lat $3=gain $4=tmp
  export TMPDIR=/scratch/agustin/tmp/claude-2621/-scratch-agustin-projects-DIMA/057226a3-598b-40aa-8396-ef0c5c742cd9/scratchpad/$4
  mkdir -p "$TMPDIR"
  echo ">>> $1 lat=$2 gain=$3 seed=1004 $(date +%H:%M:%S)"
  stdbuf -oL -eL $PY sims/scripts/sweep_rate_demo.py --headless --controller rl \
    --weights sims/models/warehouse/nav_fused_v12_cnn.pt \
    --sim_dt 0.01 --decimation 1 --obstacle_level 8 --prop_density 0.30 \
    --sched_latency_ms "$2" --percep_hold_ms 0 --moment_scale "$3" \
    --cruise_speed 1.4 --walk_speed 0.0 --episodes 1 --seed 1004 --max_steps 1800 \
    --dump_figure_data "$CV/${1}_figdata" --sweep-csv "$CV/${1}.csv" 2>&1 \
    | stdbuf -oL grep -E "\[SWEEP\]|\[ep0|\[figure\] wrote|\[sched\]|Traceback|Error|CUDA"
  echo "${1}_DONE $(date +%H:%M:%S)"
}
run s1004_xpu 4.89 0.005 s1004x_tmp
run s1004_ros 12.40 0.01 s1004r_tmp
echo "ALL_S1004_DONE"
