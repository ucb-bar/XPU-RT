#!/usr/bin/env bash
# The two displayed flights recorded as videos (chase + every model input + overhead path), each
# also dumping its figure data, so the composite and the video are the same flight. One
# simulator at a time, gated on GPU room. The baseline is re-flown over the cell's mid-course
# crash seeds until an attempt reproduces the rule's flight (two gates, then the crash); every
# attempt's outcome is logged.
. "$(dirname "$0")/env.sh"
set -u; cd "$SIM_TREE"
D="$RES/campaign/video"; mkdir -p $D/tmp; export TMPDIR=$D/tmp
PY="$ISAAC_PY"; W=sims/models/warehouse/nav_fused_v12_cnn.pt
CRU="${CRUISE:-1.4}"; XS="${XPU_SEED:-1001}"; ROS_LAT="${ROS_LAT:-29.93}"; XPU_LAT="${XPU_LAT:-6.30}"; SEEDS="${SEEDS:-1008 1005 1002 1007}"; MAX="${MAX_ATTEMPTS:-8}"
COMMON="--headless --controller rl --weights $W --sim_dt 0.01 --decimation 1 --obstacle_level 8 --prop_density 0.30 --moment_scale 0.0055 --cruise_speed $CRU --episodes 1 --max_steps 1900 --keep_video"
gpu_room() { local free; free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1); [ "$free" -ge "${NEED_MB:-7000}" ]; }
fly() { # <tag> <args...>  -> prints the outcome line
  local tag=$1; shift; until gpu_room; do sleep 60; done
  rm -rf $D/${tag}_figdata; echo "=== $(date +%H:%M:%S) $tag"
  timeout 2400 $PY sims/scripts/record_sensor_demo.py $COMMON "$@" --gantt_schedule "${GANTT:-}" --save_video $D/$tag.mp4 --dump_figure_data $D/${tag}_figdata > $D/$tag.log 2>&1
  grep -h "outcome=" $D/$tag.log | tail -n 1
}
fly xpu_s${XS} --sched_latency_ms $XPU_LAT --seed $XS --post_success_steps 100
k=0; while [ $k -lt $MAX ]; do for s in $SEEDS; do k=$((k+1)); [ $k -gt $MAX ] && break
  line=$(fly ros_s${s}_a$k --sched_latency_ms $ROS_LAT --seed $s); echo "  $line"
  if echo "$line" | grep -q "outcome=crash .*gates=2/4"; then echo "ROS_VIDEO=$D/ros_s${s}_a$k"; echo VIDEOS_DONE; exit 0; fi
done; done
echo "no attempt reproduced a two-gate crash in $MAX tries"; echo VIDEOS_DONE
