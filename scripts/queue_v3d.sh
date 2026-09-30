#!/usr/bin/env bash
# queue_v3's remaining stages, run after the baseline-first pair search (queue_v3c) so the search is not starved
# of the simulator slot: the eight-core ROS 2 flight arms with their latency, then the final renders, then the
# QUEUE_V3_DONE marker that releases the replicate queue (queue_v3b).
#   nohup bash scripts/queue_v3d.sh > results/codesign_feedback/queue_v3d.log 2>&1 &
set -u; cd "$(dirname "$0")/.."
R=$PWD/results/codesign_feedback; T=$R/ctrl_traces
say(){ echo "=== $(date +%H:%M:%S) $*"; }
until grep -q QUEUE_V3C_DONE $R/queue_v3c.log 2>/dev/null; do sleep 300; done
say "4. ROS 2 on all eight cores, flown with its latency (two YOLO nodes at 45 and 90 Hz, multi-threaded executor)"
OUT=$R/campaign_percep ARMS="ros_vanilla4x2:$T/ros_vanilla4x245.csv:37.0:0 ros_vanilla4x2_90:$T/ros_vanilla4x290.csv:254.3:19.6 ros_multi:$T/ros_multi45.csv:264.7:50.1" \
  SPEEDS="1.2 1.0 1.4 1.8 1.6" bash scripts/campaign_percep.sh 2>&1 | grep -E '^\[SWEEP\]|^=== ' | cut -c1-140
say "5. final renders"
for P in 0 1; do PAPER=$P CELL=tall1005 TUNED=all MAIN=1 bash scripts/render_showdown_v3.sh 2>&1 | grep -E 'FAIL|DONE'; done
S=$(grep -h '^PAIR_SEED=' $R/campaign_v2/display_v3s_c*.log 2>/dev/null | tail -n 1 | cut -d= -f2)
if [ -n "$S" ]; then
  L=$(grep -l "^PAIR_SEED=$S" $R/campaign_v2/display_v3s_c*.log | head -n 1); TC=$(basename "$L" .log | sed 's/.*_c//')
  for P in 0 1; do PAPER=$P CELL=tall1008 XPU_DIR=$R/campaign_v2/display_v3s_c$TC/xpu_s${S}_figdata ROS_DIR=$R/campaign_v2/display_v3s_c$TC/ros_s${S}_figdata SCENE_RECORDS=$R/campaign_scene/tall${S}s DISPLAY_CRUISE=$TC MAIN=1 bash scripts/render_showdown_v3.sh 2>&1 | grep -E 'FAIL|DONE'; done
fi
echo QUEUE_V3_DONE
