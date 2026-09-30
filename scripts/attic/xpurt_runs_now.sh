#!/usr/bin/env bash
set -u; cd "$(dirname "$0")/.."
while pgrep -f "ros_traced_matri[x]" >/dev/null; do sleep 5; done
say() { echo "=== $(date +%H:%M:%S) $*"; }
say "long25 clamped"; scripts/run_xpurt_long.sh schedules/scheduled_wh_coupled_chain_long25_greedy_clamped.json long25 3 2>&1 | grep -E "^===|trace rows|DONE|Error|ValueError"
say "long45 clamped"; scripts/run_xpurt_long.sh schedules/scheduled_wh_coupled_chain_long_greedy_clamped.json long45 3 2>&1 | grep -E "^===|trace rows|DONE|Error|ValueError"
say "long25 FIFO";    MODELBLASTER_K1_RT_PRIORITY=80 scripts/run_xpurt_long.sh schedules/scheduled_wh_coupled_chain_long25_greedy_clamped.json long25 1 2>&1 | grep -E "^===|trace rows|DONE"
say "XPURT_RUNS_DONE"
