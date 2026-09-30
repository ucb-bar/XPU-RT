#!/usr/bin/env bash
# After queue_v3: the baseline's replicates, so every arm of the flight panels carries the same seeds the same
# number of times (the XPU-RT arms were flown three times over seeds 1000-1011 at course A d0.30; the vanilla
# graph once). Two more replicates of ROS 2 vanilla at 0.8-1.8 m/s with its latency, appended to
# campaign_percep/campaign.csv (the sweep writer appends; the skip rule of campaign_percep.sh is not used here),
# then the greedy table's missing speeds, then the renders. One simulator, GPU-room gated.
#   nohup bash scripts/queue_v3b.sh > results/codesign_feedback/queue_v3b.log 2>&1 &
. "$(dirname "$0")/env.sh"
set -u; cd "$(dirname "$0")/.."
WT=$PWD; R=$WT/results/codesign_feedback; T=$R/ctrl_traces; OUT=$R/campaign_percep; export TMPDIR=$OUT/tmp; mkdir -p $TMPDIR
PY="$ISAAC_PY"; W=$WT/sims/models/warehouse/nav_fused_v12_cnn.pt
say(){ echo "=== $(date +%H:%M:%S) $*"; }
gpu_room() { local free n; free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1); n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c .); [ "$free" -ge "${NEED_MB:-8500}" ] && [ "$n" -lt "${MAX_SIMS:-3}" ]; }
until grep -q QUEUE_V3_DONE $R/queue_v3.log 2>/dev/null; do sleep 300; done
fly(){ local name=$1 trace=$2 lat=$3 cru=$4 rep=$5   # one 12-seed replicate, recorded
  until gpu_room; do sleep 60; done
  local tag="${name}_lat${lat}_h0_a_d0.30_w0.0_g0.0055_c${cru}_rep${rep}"; say "$tag"
  (cd "$SIM_TREE" && timeout 9000 $PY sims/scripts/sweep_rate_demo.py --headless --controller rl --weights $W --sim_dt 0.01 --decimation 1 \
     --obstacle_level 8 --prop_density 0.30 --percep_hold_ms 0 --percep_latency_ms $lat --moment_scale 0.0055 --cruise_speed $cru --walk_speed 0.0 \
     --episodes 12 --seed 1000 --max_steps 1800 --ctrl_trace $trace --sweep-csv $OUT/campaign.csv --record_dir $OUT/records/$tag > $OUT/$tag.log 2>&1)
  grep -h '^\[SWEEP\]' $OUT/$tag.log | tail -n 1 | cut -c1-120; }
say "ROS 2 vanilla, two more replicates"
for rep in 2 3; do for cru in 1.0 1.2 1.4 1.6 1.8 0.8; do fly ros_vanilla4 $T/ros_vanilla445.csv 242 $cru $rep; done; done
say "XPU-RT greedy at the speeds not yet flown, then its replicates"
for cru in 1.2 1.6 0.8; do fly xpu_greedy $T/xpu_a_greedy.csv 748 $cru 1; done
for rep in 2 3; do for cru in 1.0 1.2 1.4 1.6 1.8; do fly xpu_greedy $T/xpu_a_greedy.csv 748 $cru $rep; done; done
say "renders"
for c in tall1005 tall1008; do CELL=$c bash scripts/render_showdown_v3.sh 2>&1 | grep -E 'FAIL|DONE'; done
echo QUEUE_V3B_DONE
