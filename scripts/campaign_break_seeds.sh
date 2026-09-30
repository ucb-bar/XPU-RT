#!/usr/bin/env bash
# Twelve more seeds (1012-1023), 30 s horizon (18 s times out at 0.8 m/s), for every cell of the breaking-point campaign, both arms, so each point carries 24
# flights: the 12-seed cells sit inside one another's Wilson bands (XPU-RT 9/12 vs 0/12 at 1.2 m/s, 4/12 vs 5/12 at
# 1.6). Appends to campaign_break/campaign.csv (the sweep writer appends; campaign_percep.sh's skip rule would
# refuse a cell that already holds 12 rows, so the flights are launched directly). Waits for the baseline's
# simulator to finish (CAMPAIGN_BREAK_ROS_DONE) and takes that slot; GPU-room gated per cell.
#   nohup bash scripts/campaign_break_seeds.sh > results/codesign_feedback/campaign_break_seeds.log 2>&1 &
. "$(dirname "$0")/env.sh"
set -u; cd "$(dirname "$0")/.."
WT=$PWD; R=$WT/results/codesign_feedback; T=$R/ctrl_traces; OUT=$R/campaign_break; export TMPDIR=$OUT/tmp; mkdir -p $TMPDIR $OUT/records
PY="$ISAAC_PY"; W=$WT/sims/models/warehouse/nav_fused_v12_cnn.pt
export WAREHOUSE_PERSON_H=1.7
say(){ echo "=== $(date +%H:%M:%S) $*"; }
gpu_room() { local free n; free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1); n=$(pgrep -fc 'sweep_rate_dem[o]\.py|record_sensor_dem[o]\.py'); [ "$free" -ge 7000 ] && [ "$n" -lt 6 ]; }
until grep -q CAMPAIGN_BREAK_ROS_DONE $R/campaign_break_ros.log 2>/dev/null; do sleep 300; done
fly(){ local name=$1 trace=$2 lat=$3 cru=$4
  until gpu_room; do sleep 60; done
  local tag="${name}_lat${lat}_h0_a_d0.30_w0.0_g0.0055_c${cru}_s1012"; say "$tag"
  (cd "$SIM_TREE" && timeout 9000 $PY sims/scripts/sweep_rate_demo.py --headless --controller rl --weights $W --sim_dt 0.01 --decimation 1 \
     --obstacle_level 8 --prop_density 0.30 --percep_hold_ms 0 --percep_latency_ms $lat --moment_scale 0.0055 --cruise_speed $cru --walk_speed 0.0 \
     --episodes 12 --seed 1012 --max_steps 3000 --ctrl_trace $trace --sweep-csv $OUT/campaign.csv --record_dir $OUT/records/$tag > $OUT/$tag.log 2>&1)
  grep -h '^\[SWEEP\]' $OUT/$tag.log | tail -n 1 | cut -c1-120; }
for cru in 1.2 1.6 1.0 1.4 2.0 0.8 1.8; do
  fly xpu_cpsat $T/xpu_a_cpsat_hard.csv 56.8 $cru
  fly ros_vanilla4 $T/ros_vanilla445.csv 242 $cru
done
echo CAMPAIGN_BREAK_SEEDS_DONE
