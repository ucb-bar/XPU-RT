#!/usr/bin/env bash
# If queue_v3's display-pair stage ends without a pair, search the cheap way round (display_pair_search.sh:
# baseline first, XPU-RT retried on the seeds where the baseline clears exactly two gates), at 1.2 then 1.4 m/s,
# then the scene runs and the renders for the pair found. Waits for queue_v3 to pass its pair stage.
#   nohup bash scripts/queue_v3c.sh > results/codesign_feedback/queue_v3c.log 2>&1 &
set -u; cd "$(dirname "$0")/.."
R=$PWD/results/codesign_feedback
say(){ echo "=== $(date +%H:%M:%S) $*"; }
until grep -qE '^=== .* 3\. calibrated' $R/queue_v3.log 2>/dev/null; do sleep 300; done
if grep -q '^PAIR_SEED=' $R/campaign_v2/display_v3.log 2>/dev/null; then say "queue_v3 found its pair; nothing to do"; echo QUEUE_V3C_DONE; exit 0; fi
S=""
for cru in 1.2 1.4; do
  say "pair search, baseline first, $cru m/s"
  CRUISE=$cru XLAT=56.8 RLAT=242 ROS_GATES=2 XPU_TRIES=3 OUTDIR=$R/campaign_v2/display_v3s_c$cru bash scripts/display_pair_search.sh 2>&1 | tee $R/campaign_v2/display_v3s_c$cru.log | grep -E 'ros 10|xpu 10|PAIR|no seed'
  S=$(grep -h '^PAIR_SEED=' $R/campaign_v2/display_v3s_c$cru.log | tail -n 1 | cut -d= -f2); [ -n "$S" ] && { TC=$cru; break; }
done
if [ -n "$S" ]; then
  say "scene runs for seed $S at $TC m/s"; CELL=tall${S}s LAYOUT_SEED=$S CRUISE=$TC bash scripts/scene_runs.sh
  say "render"; CELL=tall1008 XPU_DIR=$R/campaign_v2/display_v3s_c$TC/xpu_s${S}_figdata ROS_DIR=$R/campaign_v2/display_v3s_c$TC/ros_s${S}_figdata SCENE_RECORDS=$R/campaign_scene/tall${S}s DISPLAY_CRUISE=$TC MAIN=1 bash scripts/render_showdown_v3.sh 2>&1 | grep -E 'FAIL|DONE'
fi
echo QUEUE_V3C_DONE
