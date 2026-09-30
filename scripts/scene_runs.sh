#!/usr/bin/env bash
# Many runs, one scene: twelve flights per arm in the display scene (props and people from one layout seed),
# each arm replaying its board-measured cadence and camera-to-control latency, every flight recorded with the
# scene's obstacle layout (sweep_rate_demo.py --record_dir) so the trajectories can be drawn over the crates,
# racks, people and gate frames. One simulator, GPU-room gated; cells already recorded are skipped.
#   CELL=tall1008 LAYOUT_SEED=1008 CRUISE=1.2 [XGAIN=0.0055 RGAIN=0.0055 XLAT=56.8 RLAT=242 GLAT=748 PERSON_H=2.4 DENS=0.30] scripts/scene_runs.sh
. "$(dirname "$0")/env.sh"
set -u; cd "$(dirname "$0")/.."
WT=$PWD; R=$WT/results/codesign_feedback; T=$R/ctrl_traces
CELL=${CELL:?}; SEED=${LAYOUT_SEED:?}; CRU=${CRUISE:?}; XGAIN=${XGAIN:-0.0055}; RGAIN=${RGAIN:-0.0055}; XLAT=${XLAT:-56.8}; RLAT=${RLAT:-242}; GLAT=${GLAT:-748}
DENS=${DENS:-0.30}; export WAREHOUSE_PERSON_H=${PERSON_H:-2.4}; N=${EPISODES:-12}
OUT=$R/campaign_scene/$CELL; mkdir -p $OUT/tmp; export TMPDIR=$OUT/tmp
PY="$ISAAC_PY"; W=$WT/sims/models/warehouse/nav_fused_v12_cnn.pt
gpu_room() { local free n; free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1); n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c .); [ "$free" -ge "${NEED_MB:-8500}" ] && [ "$n" -lt "${MAX_SIMS:-3}" ]; }
cd "$SIM_TREE"
for arm in "xpu_cpsat:$T/xpu_a_cpsat_hard.csv:$XLAT:$XGAIN" "ros_vanilla:$T/ros_vanilla445.csv:$RLAT:$RGAIN" "xpu_greedy:$T/xpu_a_greedy.csv:$GLAT:$XGAIN"; do
  IFS=: read -r name trace lat gain <<< "$arm"
  if [ "$(ls $OUT/$name/ep*.npz 2>/dev/null | wc -l)" -ge "$N" ]; then echo "skip $name (recorded)"; continue; fi
  until gpu_room; do sleep 60; done
  echo "=== $(date +%H:%M:%S) scene $CELL seed $SEED $name lat $lat gain $gain cruise $CRU ==="
  timeout 9000 $PY sims/scripts/sweep_rate_demo.py --headless --controller rl --weights $W --sim_dt 0.01 --decimation 1 --obstacle_level 8 \
    --prop_density $DENS --percep_hold_ms 0 --percep_latency_ms $lat --moment_scale $gain --cruise_speed $CRU --walk_speed 0.0 \
    --episodes $N --seed 1000 --max_steps 1800 --ctrl_trace $trace --layout_seed $SEED \
    --sweep-csv $OUT/campaign.csv --record_dir $OUT/$name > $OUT/$name.log 2>&1
  grep -h '^\[SWEEP\]' $OUT/$name.log | tail -n 1 | cut -c1-160
done
echo SCENE_RUNS_DONE $CELL
