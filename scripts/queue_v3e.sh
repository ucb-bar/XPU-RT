#!/usr/bin/env bash
# A two-gate baseline pair at 1.2 m/s over the next seed block, baseline first (display_pair_search.sh), then its
# scene runs and renders as the tall1012 cell. Shares the simulator gate with the other queues.
#   nohup bash scripts/queue_v3e.sh > results/codesign_feedback/queue_v3e.log 2>&1 &
set -u; cd "$(dirname "$0")/.."
R=$PWD/results/codesign_feedback
say(){ echo "=== $(date +%H:%M:%S) $*"; }
say "pair search, baseline first, 1.2 m/s, seeds 1012-1023"
CRUISE=1.2 XLAT=56.8 RLAT=242 ROS_GATES=2 XPU_TRIES=3 SEEDS="1012 1013 1014 1015 1016 1017 1018 1019 1020 1021 1022 1023" OUTDIR=$R/campaign_v2/display_v3s_c1.2b bash scripts/display_pair_search.sh 2>&1 | tee $R/campaign_v2/display_v3s_c1.2b.log | grep -E 'ros 10|xpu 10|PAIR|no seed'
S=$(grep -h '^PAIR_SEED=' $R/campaign_v2/display_v3s_c1.2b.log | tail -n 1 | cut -d= -f2)
if [ -n "$S" ]; then
  say "scene runs for seed $S at 1.2 m/s"; CELL=tall${S}s LAYOUT_SEED=$S CRUISE=1.2 bash scripts/scene_runs.sh
  say "render"; for P in 0 1; do PAPER=$P CELL=tall1008 XPU_DIR=$R/campaign_v2/display_v3s_c1.2b/xpu_s${S}_figdata ROS_DIR=$R/campaign_v2/display_v3s_c1.2b/ros_s${S}_figdata SCENE_RECORDS=$R/campaign_scene/tall${S}s DISPLAY_CRUISE=1.2 bash scripts/render_showdown_v3.sh 2>&1 | grep -E 'FAIL|DONE'; done
fi
echo QUEUE_V3E_DONE
