#!/usr/bin/env bash
# Re-run the best-effort layouts that the ssh drops emptied, plus the two new 45 Hz layouts,
# then hand the board back to the ROS replicates.
set -u; cd "$(dirname "$0")/.."
say() { echo "=== $(date +%H:%M:%S) $*"; }
while ! grep -q "resume replicates" results/codesign_feedback/xpurt_long/best_runs.log 2>/dev/null; do sleep 10; done
pkill -f "^bash scripts/xpurt_best_run[s].sh"; sleep 1; pkill -f "^bash scripts/ros_traced_matri[x].sh"; sleep 1
ssh k1 'pkill -f cpu_sampler; pkill -f ros_mb_chain_traced; true'
say "best25 p4";     scripts/run_xpurt_long.sh schedules/best25_p4.json best25p4 3 2>&1 | grep -E "^===|trace rows|DONE|retrying"
say "best45 alt2";   scripts/run_xpurt_long.sh schedules/best45_alt2.json best45alt2 3 2>&1 | grep -E "^===|trace rows|DONE|retrying"
say "best45 alt4s";  scripts/run_xpurt_long.sh schedules/best45_alt4s.json best45alt4s 3 2>&1 | grep -E "^===|trace rows|DONE|retrying"
say "best45 alt4 (2 more)"; scripts/run_xpurt_long.sh schedules/best45_alt4.json best45alt4b 2 2>&1 | grep -E "^===|trace rows|DONE|retrying"
say "resume replicates"
for rep in 2 3; do for arm in cship cspin cp3 spin nproc yproc p3 ship multi; do
  RATES="15 25 45" scripts/ros_traced_matrix.sh $arm $rep 2>&1 | grep -E "MATRIX_DONE|error|incomplete"
done; done
say "BEST_RUNS2_DONE"
