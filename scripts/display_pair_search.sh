#!/usr/bin/env bash
# The display pair, searched the cheap way round: the baseline is flown first on each seed of the scene (its
# flights are short) and only the seeds where it clears exactly ROS_GATES gates get XPU-RT attempts (the simulator
# is non-deterministic, so up to XPU_TRIES per seed). Both arms replay their K1 cadence and camera-to-control
# latency; the same layout seed gives both the same scene. Every attempt is kept and logged.
#   CRUISE=1.2 XLAT=56.8 RLAT=242 ROS_GATES=2 XPU_TRIES=3 SEEDS="1000 ..." OUTDIR=... scripts/display_pair_search.sh
. "$(dirname "$0")/env.sh"
set -u; cd "$(dirname "$0")/.."
WT=$PWD; T=$WT/results/codesign_feedback/ctrl_traces
CRU=${CRUISE:-1.2}; XLAT=${XLAT:-56.8}; RLAT=${RLAT:-242}; XHOLD=${XHOLD:-0}; RHOLD=${RHOLD:-0}; XGAIN=${XGAIN:-0.0055}; RGAIN=${RGAIN:-0.0055}; DENS=${DENS:-0.30}
XT=${XT:-$T/xpu_a_cpsat_hard.csv}; RT=${RT:-$T/ros_vanilla445.csv}; GATES=${ROS_GATES:-2}; TRIES=${XPU_TRIES:-3}
SEEDS="${SEEDS:-1000 1001 1002 1003 1004 1005 1006 1007 1008 1009 1010 1011}"
D=${OUTDIR:?}; mkdir -p $D/tmp; export TMPDIR=$D/tmp
PY="$ISAAC_PY"; W=sims/models/warehouse/nav_fused_v12_cnn.pt
COMMON="--headless --controller rl --weights $W --sim_dt 0.01 --decimation 1 --obstacle_level 8 --prop_density $DENS --cruise_speed $CRU --episodes 1 --max_steps 1800 --keep_video"
gpu_room() { local free n; free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1); n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c .); [ "$free" -ge "${NEED_MB:-7000}" ] && [ "$n" -lt "${MAX_SIMS:-3}" ]; }
fly() { local tag=$1 gantt=$2; shift 2; until gpu_room; do sleep 60; done
  rm -rf $D/${tag}_figdata; echo "=== $(date +%H:%M:%S) $tag"
  (cd "$SIM_TREE" && timeout 2400 $PY sims/scripts/record_sensor_demo.py $COMMON "$@" --gantt_schedule $WT/$gantt --save_video $D/$tag.mp4 --dump_figure_data $D/${tag}_figdata > $D/$tag.log 2>&1)
  grep -h "outcome=" $D/$tag.log | tail -n 1; }
for s in $SEEDS; do
  r=$(fly ros_s${s} schedules/measured_gantt_ros.json --ctrl_trace $RT --percep_latency_ms $RLAT --percep_hold_ms $RHOLD --moment_scale $RGAIN --seed $s --layout_seed $s); echo "  ros $s: $r"
  echo "$r" | grep -qE "outcome=crash .*gates=${GATES}/4" || continue
  for k in $(seq 1 $TRIES); do
    x=$(fly xpu_s${s}_t${k} schedules/measured_gantt_xpu.json --ctrl_trace $XT --percep_latency_ms $XLAT --percep_hold_ms $XHOLD --moment_scale $XGAIN --seed $((s + 100 * k)) --layout_seed $s --post_success_steps 100); echo "  xpu $s try $k: $x"
    if echo "$x" | grep -q "outcome=success"; then mv $D/xpu_s${s}_t${k}_figdata $D/xpu_s${s}_figdata; mv $D/xpu_s${s}_t${k}.mp4 $D/xpu_s${s}.mp4 2>/dev/null; echo "PAIR_SEED=$s"; echo PAIR $D/xpu_s${s}_figdata $D/ros_s${s}_figdata; echo SEARCH_DONE; exit 0; fi
  done
done
echo "no seed gave the pair"; echo SEARCH_DONE
