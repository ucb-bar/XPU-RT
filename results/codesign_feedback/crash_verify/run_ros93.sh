#!/usr/bin/env bash
# Honest ROS crash flight: sched_latency = DERIVED single-threaded-executor worst response 93.22 ms
# (ROS's default rclcpp executor serializes the unshardable perception ahead of the control callback
#  -> control output updates ~every 93 ms = ~10 Hz -> stability starved -> crash). Matched scene to the
# XPU-RT complete dump: same controller (rl_controller_velctrl_dr4.pt), cruise 1.4, decimation 1,
# obstacle_level 8, prop_density 0.30, walk 0.0, seed base 1000. Gain = 0.01 (50Hz-calibrated, GENEROUS
# to ROS: more authority than a 10Hz-fair 0.05 would give, and avoids the PhysX blowup 0.05 caused).
set -u
cd /scratch/agustin/projects/DIMA/XPU-RT
export TMPDIR=/scratch/agustin/tmp/claude-2621/-scratch-agustin-projects-DIMA/057226a3-598b-40aa-8396-ef0c5c742cd9/scratchpad/ros93_tmp
mkdir -p "$TMPDIR"
PY=/scratch2/agustin/miniforge3/envs/env_isaaclab/bin/python
CV=/scratch/agustin/xpurt-dev-sync/results/codesign_feedback/crash_verify
stdbuf -oL -eL $PY sims/scripts/sweep_rate_demo.py --headless --controller rl \
  --weights sims/models/warehouse/rl_controller_velctrl_dr4.pt \
  --sim_dt 0.01 --decimation 1 --obstacle_level 8 --prop_density 0.30 \
  --sched_latency_ms 93.22 --percep_hold_ms 0 --moment_scale 0.01 \
  --cruise_speed 1.4 --walk_speed 0.0 \
  --episodes 6 --seed 1000 --max_steps 1800 \
  --dump_figure_data "$CV/ros93_figdata" \
  --sweep-csv "$CV/ros93.csv" 2>&1 | grep -E "\[SWEEP\]|\[ep0|\[figure\]|\[sched\]|Traceback|Error|CUDA"
echo "ROS93_DONE rc=$?"
