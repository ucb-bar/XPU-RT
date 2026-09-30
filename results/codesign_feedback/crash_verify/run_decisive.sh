#!/bin/bash
# Decisive HONEST warehouse test (stable-FS output; scratchpad gets auto-wiped).
# Same scene/controller/gain, control FULL-rate 100 Hz (in-envelope), the ONLY variable is
# perception freshness: fresh (OUR schedule meets 23 ms) vs 150 ms (ROS honest achieved cadence:
# 356 deadline misses, 153 ms makespan -> a fresh detection only lands ~every 150 ms).
# walk_speed 1.5 = brisk patrol so a stale view of a MOVING obstacle actually routes the drone wrong.
# PhysX is non-deterministic run-to-run -> the honest claim is a SUCCESS-RATE gap over many seeds.
set -u
STABLE=/scratch/agustin/xpurt-dev-sync/results/codesign_feedback/crash_verify
PY=/scratch2/agustin/miniforge3/envs/env_isaaclab/bin/python
cd /scratch/agustin/projects/DIMA/XPU-RT
CSV=$STABLE/walkgrid.csv
mkdir -p $STABLE/logs
for walk in 1.5; do
  for hold in 0 150; do
    tag="w${walk}_ph${hold}"
    echo "=== START $tag $(date +%H:%M:%S) ==="
    $PY sims/scripts/sweep_rate_demo.py --headless --controller rl \
      --weights sims/models/warehouse/nav_fused_v12_cnn.pt \
      --sim_dt 0.01 --decimation 1 --obstacle_level 8 --prop_density 0.35 \
      --sched_latency_ms 0 --percep_hold_ms "$hold" --moment_scale 0.01 \
      --cruise_speed 1.2 --walk_speed "$walk" \
      --episodes 12 --seed 1000 --max_steps 1800 --gantt_schedule "" \
      --sweep-csv "$CSV" > "$STABLE/logs/${tag}.log" 2>&1
    echo "=== DONE $tag $(date +%H:%M:%S) rc=$? ==="
  done
done
echo "ALL DONE $(date +%H:%M:%S)"
