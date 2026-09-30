#!/usr/bin/env bash
# The displayed XPU-RT flight, re-flown over the cell's success seeds until an attempt completes
# the course (the simulator is not run-to-run deterministic); then the composite from that flight
# and the baseline flight display_v2.sh found.
. "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/scripts/env.sh"   # SIM_TREE, ISAAC_PY, RES: see scripts/env.local.sh.example
set -u; cd "$(dirname "$0")/.."
WT=$PWD; D=$WT/results/codesign_feedback/campaign_v2/display; export TMPDIR=$D/tmp
while ! grep -q DISPLAY_V2_DONE $D/../display_v2.log 2>/dev/null; do sleep 120; done
RD=$(grep -o "ROS_DISPLAY=.*" $D/../display_v2.log | cut -d= -f2); [ -n "$RD" ] || { echo "no baseline flight yet"; echo DISPLAY_V2_XPU_DONE; exit 0; }
CRU="${CRUISE:-1.8}"; GAIN="${GAIN:-0.0055}"; SEEDS="${XPU_SEEDS:?}"; MAX="${MAX_ATTEMPTS:-8}"
XT=$WT/results/codesign_feedback/ctrl_traces/xpu_a_cpsat_hard.csv
PY="$ISAAC_PY"; W=sims/models/warehouse/nav_fused_v12_cnn.pt
COMMON="--headless --controller rl --weights $W --sim_dt 0.01 --decimation 1 --obstacle_level 8 --prop_density 0.30 --moment_scale $GAIN --cruise_speed $CRU --episodes 1 --max_steps 1900 --keep_video"
gpu_room() { local free; free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1); [ "$free" -ge 7000 ]; }
XD=""; k=0
while [ $k -lt $MAX ] && [ -z "$XD" ]; do for s in $SEEDS; do k=$((k+1)); [ $k -gt $MAX ] && break
  until gpu_room; do sleep 60; done; tag=xpu_s${s}_a$k; rm -rf $D/${tag}_figdata; echo "=== $(date +%H:%M:%S) $tag"
  (cd "$SIM_TREE" && timeout 2400 $PY sims/scripts/record_sensor_demo.py $COMMON --ctrl_trace $XT --seed $s --post_success_steps 100 --gantt_schedule $WT/schedules/measured_gantt_xpu.json --save_video $D/$tag.mp4 --dump_figure_data $D/${tag}_figdata > $D/$tag.log 2>&1)
  line=$(grep -h "outcome=" $D/$tag.log | tail -n 1); echo "  $line"
  echo "$line" | grep -q "outcome=success" && XD=$D/${tag}_figdata && break
done; done
[ -n "$XD" ] || { echo "no attempt completed the course"; echo DISPLAY_V2_XPU_DONE; exit 0; }
echo "XPU_DISPLAY=$XD"
SPEC=data/toplevel/wh_chain45_solve.json XPU_ARM=acpsat_hardr1 XPU_SCHED=schedules/fig_a_cpsat_hard_clamped.json XPU2_ARM=agreedyr1 XPU2_SCHED=schedules/fig_a_greedy_clamped.json ROS_TAG=45_vanilla4_r1 \
  LABEL_XPU="XPU-RT·CP-SAT" LABEL_XPU2="XPU-RT·greedy" LABEL_ROS="ROS 2 vanilla" WINDOW_MS=140 XPU_DIR=$XD ROS_DIR=$RD DPI=${DPI:-200} \
  scripts/render_showdown_measured.sh results/codesign_feedback/refined/warehouse_showdown_v2 2>&1 | tail -n 3
echo DISPLAY_V2_XPU_DONE
