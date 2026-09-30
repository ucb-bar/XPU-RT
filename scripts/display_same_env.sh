#!/usr/bin/env bash
# One scene for both arms: props and people drawn from the same layout seed, the same seed for
# everything else, each arm one recorded flight. Seeds are tried in order until XPU-RT completes
# the course and the baseline crashes after one or two gates in that very scene; every attempt
# is logged. Then the composite is rendered from the pair and verified.
. "$(dirname "$0")/env.sh"
set -u; cd "$(dirname "$0")/.."
WT=$PWD; D=$WT/results/codesign_feedback/campaign_v2/display_same; mkdir -p $D/tmp; export TMPDIR=$D/tmp
CRU="${CRUISE:-1.8}"; GAIN="${GAIN:-0.0055}"; SEEDS="${SEEDS:-1004 1001 1006 1010 1000 1002 1003 1005 1007 1008 1009 1011}"
# ROS_ACCEPT is the whole acceptance pattern for the baseline's flight, for a form whose baseline does
# not fail by crashing after a gate; left unset it is the crash-after-ROS_GATES rule the other forms use.
DENS="${DENS:-0.30}"; XLAT="${XLAT:-0}"; RLAT="${RLAT:-0}"; XHOLD="${XHOLD:-0}"; RHOLD="${RHOLD:-0}"; XGAIN="${XGAIN:-$GAIN}"; RGAIN="${RGAIN:-$GAIN}"   # the cell's scene and replay settings (latency, goal hold, per-arm gain)
XT=${XT:-$WT/results/codesign_feedback/ctrl_traces/xpu_a_cpsat_hard.csv}; RT=${RT:-$WT/results/codesign_feedback/ctrl_traces/ros_vanilla445.csv}
D=${OUTDIR:-$D}; mkdir -p $D/tmp
PY="$ISAAC_PY"; W=sims/models/warehouse/nav_fused_v12_cnn.pt
COMMON="--headless --controller rl --weights $W --sim_dt 0.01 --decimation 1 --obstacle_level 8 --prop_density $DENS --cruise_speed $CRU --episodes 1 --max_steps 1800 --keep_video"
gpu_room() { local free n; free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1); n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c .); [ "$free" -ge "${NEED_MB:-7000}" ] && [ "$n" -lt "${MAX_SIMS:-3}" ]; }
fly() { local tag=$1 gantt=$2; shift 2; until gpu_room; do sleep 60; done
  rm -rf $D/${tag}_figdata; echo "=== $(date +%H:%M:%S) $tag"
  (cd "$SIM_TREE" && timeout 2400 $PY sims/scripts/record_sensor_demo.py $COMMON "$@" --gantt_schedule $WT/$gantt --save_video $D/$tag.mp4 --dump_figure_data $D/${tag}_figdata > $D/$tag.log 2>&1)
  grep -h "outcome=" $D/$tag.log | tail -n 1; }
XD=""; RD=""
for s in $SEEDS; do
  x=$(fly xpu_s${s} schedules/measured_gantt_xpu.json --ctrl_trace $XT --percep_latency_ms $XLAT --percep_hold_ms $XHOLD --moment_scale $XGAIN --seed $s --layout_seed $s --post_success_steps 100); echo "  xpu $s: $x"
  echo "$x" | grep -q "outcome=success" || continue
  r=$(fly ros_s${s} schedules/measured_gantt_ros.json --ctrl_trace $RT --percep_latency_ms $RLAT --percep_hold_ms $RHOLD --moment_scale $RGAIN --seed $s --layout_seed $s); echo "  ros $s: $r"
  if echo "$r" | grep -qE "${ROS_ACCEPT:-outcome=crash .*gates=${ROS_GATES:-[12]}/4}"; then XD=$D/xpu_s${s}_figdata; RD=$D/ros_s${s}_figdata; echo "PAIR_SEED=$s"; break; fi   # ROS_GATES=2: the baseline must clear exactly two gates
done
[ "${RENDER:-1}" = 0 ] && { [ -n "$XD" ] && echo "PAIR $XD $RD"; echo SAME_ENV_DONE; exit 0; }
[ -n "$XD" ] || { echo "no seed gave the pair"; echo SAME_ENV_DONE; exit 0; }
SPEC=data/toplevel/wh_chain45_solve.json XPU_ARM=acpsat_hardr1 XPU_SCHED=schedules/fig_a_cpsat_hard_clamped.json XPU2_ARM=agreedyr1 XPU2_SCHED=schedules/fig_a_greedy_clamped.json ROS_TAG=45_vanilla4_r1 \
  LABEL_XPU="XPU-RT·CP-SAT" LABEL_XPU2="XPU-RT·greedy" LABEL_ROS="ROS 2 vanilla" WINDOW_MS=140 XPU_DIR=$XD ROS_DIR=$RD DPI=${DPI:-200} \
  scripts/render_showdown_measured.sh ${RENDER_OUT:-results/codesign_feedback/refined/warehouse_showdown_v2} 2>&1 | tail -n 3
echo SAME_ENV_DONE
