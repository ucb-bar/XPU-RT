#!/usr/bin/env bash
# After the ROS replicates: clean SCHED_OTHER runs of the 25 Hz layout (the earlier ones carried a
# stale stdout), a FIFO variant of the 45 Hz arm, and two more alt2 replicates.
set -u; cd "$(dirname "$0")/.."
say() { echo "=== $(date +%H:%M:%S) $*"; }
while ! grep -q "BEST_RUNS2_DONE" results/codesign_feedback/xpurt_long/best_runs2.log 2>/dev/null; do sleep 30; done
say "best25 p4, SCHED_OTHER"; scripts/run_xpurt_long.sh schedules/best25_p4.json best25p4 3 2>&1 | grep -E "^===|trace rows|retrying"
say "best45 alt2, FIFO";      MODELBLASTER_K1_RT_PRIORITY=80 scripts/run_xpurt_long.sh schedules/best45_alt2.json best45alt2 1 2>&1 | grep -E "^===|trace rows|retrying"
say "best45 alt2, 2 more";    scripts/run_xpurt_long.sh schedules/best45_alt2.json best45alt2b 2 2>&1 | grep -E "^===|trace rows|retrying"
say "XPURT_FINAL_DONE"
