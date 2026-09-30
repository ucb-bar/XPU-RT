#!/usr/bin/env bash
# The two displayed flights of the selected cell, recorded (video + figure data in one flight).
# The prop layout of an episode depends on how many steps the process has flown before it, so
# the display flights are flown the way the campaign flew them: one process, the campaign's seed
# sequence, and the recorder keeps the first episode that meets the arm's rule (XPU-RT: the first
# success, with a 1 s tail past the last gate; the baseline: the first crash after exactly two
# gates). Every episode's outcome is in the recorder's log. Then the composite is rendered from
# these flights and verified.
set -u; cd "$(dirname "$0")/.."
WT=$PWD; . "$WT/scripts/env.sh"
D=$RES/campaign_v2/display; mkdir -p $D/tmp; export TMPDIR=$D/tmp
CRU="${CRUISE:-1.8}"; GAIN="${GAIN:-0.0055}"; EPS="${EPISODES:-12}"; SEED0="${SEED0:-1000}"
XT=$WT/results/codesign_feedback/ctrl_traces/xpu_a_cpsat_hard.csv; RT=$WT/results/codesign_feedback/ctrl_traces/ros_vanilla445.csv
XPU_ARM=acpsat_hardr1; XPU_SCHED=schedules/fig_a_cpsat_hard_clamped.json; XPU2_ARM=agreedyr1; XPU2_SCHED=schedules/fig_a_greedy_clamped.json; ROS_TAG=45_vanilla4_r1
PY="$ISAAC_PY"; W=sims/models/warehouse/nav_fused_v12_cnn.pt
COMMON="--headless --controller rl --weights $W --sim_dt 0.01 --decimation 1 --obstacle_level 8 --prop_density 0.30 --moment_scale $GAIN --cruise_speed $CRU --episodes $EPS --seed $SEED0 --max_steps 1800"
gpu_room() { local free; free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1); [ "$free" -ge "${NEED_MB:-7000}" ]; }
fly() { local tag=$1 gantt=$2; shift 2; until gpu_room; do sleep 60; done
  rm -rf $D/${tag}_figdata; echo "=== $(date +%H:%M:%S) $tag"
  (cd "$SIM_TREE" && timeout 7200 $PY sims/scripts/record_sensor_demo.py $COMMON "$@" --gantt_schedule $WT/$gantt --save_video $D/$tag.mp4 --dump_figure_data $D/${tag}_figdata > $D/$tag.log 2>&1)
  grep -h "outcome=\|captured" $D/$tag.log | cut -c1-100; }
fly xpu_c${CRU} schedules/measured_gantt_xpu.json --ctrl_trace $XT --post_success_steps 100
fly ros_c${CRU} schedules/measured_gantt_ros.json --ctrl_trace $RT --keep_gates 2
XD=$D/xpu_c${CRU}_figdata; RD=$D/ros_c${CRU}_figdata
[ -f $XD/figure_data.npz ] && [ -f $RD/figure_data.npz ] || { echo "a display flight is missing"; echo DISPLAY_V3_DONE; exit 0; }
SPEC=data/toplevel/wh_chain45_solve.json XPU_ARM=$XPU_ARM XPU_SCHED=$XPU_SCHED XPU2_ARM=$XPU2_ARM XPU2_SCHED=$XPU2_SCHED ROS_TAG=$ROS_TAG \
  LABEL_XPU="XPU-RT·CP-SAT" LABEL_XPU2="XPU-RT·greedy" LABEL_ROS="ROS 2 vanilla" WINDOW_MS=140 XPU_DIR=$XD ROS_DIR=$RD DPI=${DPI:-200} \
  scripts/render_showdown_measured.sh results/codesign_feedback/refined/warehouse_showdown_v2 2>&1 | tail -n 3
echo DISPLAY_V3_DONE
