#!/usr/bin/env bash
# One close_drawer rollout WITH VIDEO, so panel A can carry a drawer snapshot.
# No drawer mp4 exists anywhere in the tree; egg and spoon both have one, and a
# figure that shows frames for two workloads and a blank for the third reads as a
# missing measurement rather than a missing render.
# Kept OUT of traces_torque3/ so the per-tick arrays it needs are not reduced
# away and the seed sweep's cell count stays clean.
source /home/ubuntu/simpler/sim_eval/roselite/env.sh
cd /home/ubuntu/simpler/sim_eval/roselite/finegrain
OUT=/home/ubuntu/simpler/sim_eval/roselite/finegrain/runs_video/vid_drawer_p105
[ -d "$OUT" ] && [ -n "$(ls $OUT/*.mp4 2>/dev/null)" ] && { echo "skip drawer video"; exit 0; }
mkdir -p "$OUT" /home/ubuntu/simpler/logs
python trace_eval2.py --task google_robot_close_drawer --latency-ms 258.7 \
       --issue-period-ms 124.8 --init-rng 100 --n 8 --save-video-every 1 \
       --out "$OUT" > /home/ubuntu/simpler/logs/drvid.log 2>&1
echo "drawer video rc=$? : $(ls $OUT/*.mp4 2>/dev/null | wc -l) mp4"
