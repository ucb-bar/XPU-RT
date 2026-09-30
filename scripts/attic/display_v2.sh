#!/usr/bin/env bash
# The two displayed flights of the selected cell, recorded (video + figure data in one flight),
# each replaying its arm's measured cadence. The baseline is re-flown over the cell's two-gate
# crash seeds until an attempt reproduces the rule's flight; every attempt is logged. Then the
# composite is rendered from the same flights and verified.
. "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/scripts/env.sh"   # SIM_TREE, ISAAC_PY, RES: see scripts/env.local.sh.example
set -u; cd "$(dirname "$0")/.."
WT=$PWD; D=$WT/results/codesign_feedback/campaign_v2/display; mkdir -p $D/tmp; export TMPDIR=$D/tmp
CRU="${CRUISE:-1.8}"; GAIN="${GAIN:-0.0055}"; XS="${XPU_SEED:-1001}"; SEEDS="${ROS_SEEDS:?}"; MAX="${MAX_ATTEMPTS:-8}"
XT=$WT/results/codesign_feedback/ctrl_traces/xpu_a_cpsat_hard.csv; RT=$WT/results/codesign_feedback/ctrl_traces/ros_vanilla445.csv
XPU_ARM=acpsat_hardr1; XPU_SCHED=schedules/fig_a_cpsat_hard_clamped.json; XPU2_ARM=agreedyr1; XPU2_SCHED=schedules/fig_a_greedy_clamped.json; ROS_TAG=45_vanilla4_r1
# the measured tables the video strips and panel I draw
XL=results/codesign_feedback/xpurt_long; RTD=results/codesign_feedback/ros_traced
python3 scripts/make_measured_gantt_pair.py --arm "xpu:xpu:$XL/trace_${XPU_ARM}_other_run1.csv:$XL/cpu_${XPU_ARM}_other_run1.csv:$XL/manifest_${XPU_ARM}_other_run1.json:$XPU_SCHED" \
  --arm "xpu2:xpu:$XL/trace_${XPU2_ARM}_other_run1.csv:$XL/cpu_${XPU2_ARM}_other_run1.csv:$XL/manifest_${XPU2_ARM}_other_run1.json:$XPU2_SCHED" \
  --arm "ros:ros:$RTD/$ROS_TAG/trace.csv:$RTD/$ROS_TAG/cpu.csv:$RTD/$ROS_TAG/manifest.json" --window-ms 140 --skip-ms 400 --spec data/toplevel/wh_chain45_solve.json | tail -n 3
PY="$ISAAC_PY"; W=sims/models/warehouse/nav_fused_v12_cnn.pt
COMMON="--headless --controller rl --weights $W --sim_dt 0.01 --decimation 1 --obstacle_level 8 --prop_density 0.30 --moment_scale $GAIN --cruise_speed $CRU --episodes 1 --max_steps 1900 --keep_video"
gpu_room() { local free; free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1); [ "$free" -ge "${NEED_MB:-7000}" ]; }
fly() { local tag=$1 gantt=$2; shift 2; until gpu_room; do sleep 60; done
  rm -rf $D/${tag}_figdata; echo "=== $(date +%H:%M:%S) $tag"
  (cd "$SIM_TREE" && timeout 2400 $PY sims/scripts/record_sensor_demo.py $COMMON "$@" --gantt_schedule $WT/$gantt --save_video $D/$tag.mp4 --dump_figure_data $D/${tag}_figdata > $D/$tag.log 2>&1)
  grep -h "outcome=" $D/$tag.log | tail -n 1; }
fly xpu_s${XS} schedules/measured_gantt_xpu.json --ctrl_trace $XT --seed $XS --post_success_steps 100
k=0; RD=""
while [ $k -lt $MAX ] && [ -z "$RD" ]; do for s in $SEEDS; do k=$((k+1)); [ $k -gt $MAX ] && break
  line=$(fly ros_s${s}_a$k schedules/measured_gantt_ros.json --ctrl_trace $RT --seed $s); echo "  $line"
  if echo "$line" | grep -q "outcome=crash .*gates=2/4"; then RD=$D/ros_s${s}_a${k}_figdata; break; fi
done; done
[ -n "$RD" ] || { echo "no attempt reproduced a two-gate crash"; echo DISPLAY_V2_DONE; exit 0; }
echo "ROS_DISPLAY=$RD"
SPEC=data/toplevel/wh_chain45_solve.json XPU_ARM=$XPU_ARM XPU_SCHED=$XPU_SCHED XPU2_ARM=$XPU2_ARM XPU2_SCHED=$XPU2_SCHED ROS_TAG=$ROS_TAG \
  LABEL_XPU="XPU-RT·CP-SAT" LABEL_XPU2="XPU-RT·greedy" LABEL_ROS="ROS 2 vanilla" WINDOW_MS=140 XPU_DIR=$D/xpu_s${XS}_figdata ROS_DIR=$RD DPI=${DPI:-200} \
  scripts/render_showdown_measured.sh results/codesign_feedback/refined/warehouse_showdown_v2 2>&1 | tail -n 3
echo DISPLAY_V2_DONE
