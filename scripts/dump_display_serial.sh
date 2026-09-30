#!/usr/bin/env bash
# The two displayed flights of the selected cell, dumped one at a time and only when the GPU has
# room for another simulator instance (a full card makes the renderer fail mid-flight).
set -u; WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; . "$WT/scripts/env.sh"; cd "$SIM_TREE"
D=$RES/campaign/display; mkdir -p $D/tmp; export TMPDIR=$D/tmp
PY="$ISAAC_PY"; W=sims/models/warehouse/nav_fused_v12_cnn.pt
CRU="${CRUISE:-1.4}"; XS="${XPU_SEED:-1001}"; RS="${ROS_SEED:-1008}"; ROS_LAT="${ROS_LAT:-29.93}"
COMMON="--headless --controller rl --weights $W --sim_dt 0.01 --decimation 1 --obstacle_level 8 --prop_density 0.30 --percep_hold_ms 0 --moment_scale 0.0055 --cruise_speed $CRU --walk_speed 0.0 --episodes 1 --max_steps 1900"
gpu_room() { local free; free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1); [ "$free" -ge "${NEED_MB:-7000}" ]; }
run_one() { # <tag> <extra args>
  local tag=$1; shift
  until gpu_room; do sleep 60; done
  rm -rf $D/${tag}_figdata
  echo "=== $(date +%H:%M:%S) $tag"
  timeout 1800 $PY sims/scripts/sweep_rate_demo.py $COMMON "$@" --dump_figure_data $D/${tag}_figdata --sweep-csv $D/display.csv > $D/$tag.log 2>&1
  grep -h SWEEP $D/$tag.log | sed 's/  s10.*//'
}
run_one xpu_s${XS}_through --sched_latency_ms 6.30 --seed $XS --post_success_steps 100
run_one ros_s${RS} --sched_latency_ms $ROS_LAT --seed $RS
echo DISPLAY_DONE
