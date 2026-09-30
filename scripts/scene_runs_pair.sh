#!/usr/bin/env bash
# Panel A's census: twelve flights per arm in the display scene (props and people from one layout seed),
# each arm replaying its board-measured control cadence, camera-to-control latency and goal hold, every
# flight recorded. scene_runs.sh names its arms in code; this takes them as ARMS=name:trace:lat:hold.
#   CELL=r36_1005 LAYOUT_SEED=1005 CRUISE=1.2 ARMS="xpu:<trace>:30.1:27.8 ros8:<trace>:32.2:27.8" scripts/scene_runs_pair.sh
. "$(dirname "$0")/env.sh"
set -u; cd "$(dirname "$0")/.."
WT=$PWD; R=$WT/results/codesign_feedback
CELL=${CELL:?}; SEED=${LAYOUT_SEED:?}; CRU=${CRUISE:?}; ARMS=${ARMS:?}; GAIN=${GAIN:-0.0055}; DENS=${DENS:-0.30}
export WAREHOUSE_PERSON_H=${PERSON_H:-2.4}; N=${EPISODES:-12}
OUT=$R/campaign_scene/$CELL; mkdir -p $OUT/tmp; export TMPDIR=$OUT/tmp
PY="$ISAAC_PY"; W=$WT/sims/models/warehouse/nav_fused_v12_cnn.pt
gpu_room() { local free n; free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1); n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c .); [ "$free" -ge "${NEED_MB:-10000}" ] && [ "$n" -lt "${MAX_SIMS:-3}" ]; }
for arm in $ARMS; do
  IFS=: read -r name trace lat hold <<< "$arm"
  if [ "$(ls $OUT/$name/ep*.npz 2>/dev/null | wc -l)" -ge "$N" ]; then echo "skip $name (recorded)"; continue; fi
  exec 8>"$R/gpu_admit.lock"; flock 8
  until gpu_room; do sleep 30; done
  ( sleep "${ADMIT_SETTLE:-150}"; flock -u 8 ) & settle=$!
  echo "=== $(date +%H:%M:%S) scene $CELL layout $SEED $name lat $lat hold $hold cruise $CRU ==="
  (cd "$SIM_TREE" && timeout 9000 $PY sims/scripts/sweep_rate_demo.py --headless --controller rl --weights $W --sim_dt 0.01 --decimation 1 --obstacle_level 8 \
    --prop_density $DENS --percep_hold_ms $hold --percep_latency_ms $lat --moment_scale $GAIN --cruise_speed $CRU --walk_speed 0.0 \
    --episodes $N --seed 1000 --max_steps 1800 --ctrl_trace $trace --layout_seed $SEED \
    --sweep-csv $OUT/campaign.csv --record_dir $OUT/$name > $OUT/$name.log 2>&1 8>&-)
  kill $settle 2>/dev/null; flock -u 8
  if grep -qE "PhysX Internal CUDA error|Simulation cannot continue|GPU is out of memory|CUDA error, code" $OUT/$name.log; then
    echo "$name gpu_fault $(date +%FT%T)" >> $OUT/quarantine.txt; echo "   QUARANTINED $name"; fi
  grep -h '^\[SWEEP\]' $OUT/$name.log | tail -n 1 | cut -c1-160
done
echo SCENE_RUNS_DONE $CELL
