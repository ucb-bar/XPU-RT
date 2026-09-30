#!/usr/bin/env bash
# Second half of the board campaign. Waits for the first orchestrator to reach its replicate
# stage, takes the board over, and runs, in this order:
#   1  XPU-RT one-second chain at its designed 45 Hz camera, x3 SCHED_OTHER + x1 FIFO
#   2  XPU-RT one-second chain at a 25 Hz camera, x3
#   3  the single-frame coupled schedule with the sampler, x3
#   4  the STARVATION MATRIX: every ROS layout at 15 / 25 / 45 Hz camera -- which callbacks
#      share the control timer's thread decides how much of the control loop survives:
#        ship   yolo(1 hart)+nav+ctrl one thread      spin   yolo(4 harts)+nav+ctrl one thread
#        nproc  yolo(4 harts)+ctrl one thread         yproc  nav+ctrl one thread
#        p3     ctrl alone (nav alone, yolo alone)    multi/smte  multi-threaded executor
#   5  ROS replicates 2 and 3 at 5..30 Hz for every arm
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"
HOST="${MODELBLASTER_K1_HOST:-k1}"
LOG1="results/codesign_feedback/ros_traced/board_campaign.log"
say() { echo "=== $(date +%H:%M:%S) $*"; }

say "waiting for the pinned arms (replicate 1) to finish"
while ! grep -q "step 5" "$LOG1" 2>/dev/null; do sleep 20; done
pkill -f board_campaign.sh; sleep 1; pkill -f ros_traced_matrix.sh; sleep 1
ssh "$HOST" 'pkill -f cpu_sampler; pkill -f ros_mb_chain_traced; true'
say "board taken over"

LONG45=schedules/scheduled_wh_coupled_chain_long_greedy_periodic_profiled.json
LONG25=schedules/scheduled_wh_coupled_chain_long25_greedy_periodic_profiled.json
while [ ! -f "$LONG25" ]; do sleep 10; done

say "XPU-RT long, 45 Hz camera";  scripts/run_xpurt_long.sh "$LONG45" long45 3 2>&1 | grep -E "^===|trace rows|DONE"
say "XPU-RT long, 45 Hz, FIFO";   MODELBLASTER_K1_RT_PRIORITY=80 scripts/run_xpurt_long.sh "$LONG45" long45 1 2>&1 | grep -E "^===|trace rows|DONE"
say "XPU-RT long, 25 Hz camera";  scripts/run_xpurt_long.sh "$LONG25" long25 3 2>&1 | grep -E "^===|trace rows|DONE"
say "XPU-RT single-frame coupled"; scripts/run_xpurt_long.sh schedules/cmp_coupled_cpsat_board.json coupled 3 2>&1 | grep -E "^===|trace rows|DONE"

say "starvation matrix"
for arm in ship spin nproc yproc p3 multi smte; do
  RATES="15 25 45" scripts/ros_traced_matrix.sh $arm 1 2>&1 | grep -E "ROS_TRACED|MATRIX_DONE|error|incomplete"
done

say "ROS replicates"
for rep in 2 3; do for arm in ship multi spin smte p3 p8 nproc yproc; do
  RATES="5 8 10 12 15 20 25 30 45" scripts/ros_traced_matrix.sh $arm $rep 2>&1 | grep -E "MATRIX_DONE|error|incomplete"
done; done
say "BOARD_CAMPAIGN2_DONE"
