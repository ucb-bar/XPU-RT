#!/usr/bin/env bash
# NEW ROS crash flight at 50 Hz (12.40 ms ZOH) — CONSERVATIVE to ROS: higher than its schedule-derived
# 25 Hz (coupled chain) / 11 Hz (single-executor), yet still crashes. gain 0.01 (0.5/50 calibrated).
# Matched scene to the XPU complete dump. Near-cliff rate -> ROS limps into the course then clips a gate.
set -u
cd /scratch/agustin/projects/DIMA/XPU-RT
export TMPDIR=/scratch/agustin/tmp/claude-2621/-scratch-agustin-projects-DIMA/057226a3-598b-40aa-8396-ef0c5c742cd9/scratchpad/ros50_tmp
mkdir -p "$TMPDIR"
PY=/scratch2/agustin/miniforge3/envs/env_isaaclab/bin/python
CV=/scratch/agustin/xpurt-dev-sync/results/codesign_feedback/crash_verify
stdbuf -oL -eL $PY sims/scripts/sweep_rate_demo.py --headless --controller rl \
  --weights sims/models/warehouse/nav_fused_v12_cnn.pt \
  --sim_dt 0.01 --decimation 1 --obstacle_level 8 --prop_density 0.30 \
  --sched_latency_ms 12.40 --percep_hold_ms 0 --moment_scale 0.01 \
  --cruise_speed 1.4 --walk_speed 0.0 --episodes 6 --seed 1000 --max_steps 1800 \
  --dump_figure_data "$CV/new_ros50_figdata" --sweep-csv "$CV/new_ros50.csv" 2>&1 \
  | stdbuf -oL grep -E "\[SWEEP\]|\[ep0|\[figure\] wrote|\[sched\]|Traceback|Error|CUDA"
echo "ROS50_DONE $(date +%H:%M:%S)"
