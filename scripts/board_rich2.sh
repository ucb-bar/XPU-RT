#!/usr/bin/env bash
set -u; cd "$(dirname "$0")/.."
say() { echo "=== $(date +%H:%M:%S) $*"; }
while ! grep -q "HIGHRATE2_DONE" results/codesign_feedback/xpurt_long/highrate2.log 2>/dev/null; do sleep 30; done
say "XPU-RT heavier stack"; scripts/run_xpurt_long.sh schedules/rich45_alt2.json rich45alt2 3 2>&1 | grep -E "^===|trace rows|retrying|Error"
scripts/run_xpurt_long.sh schedules/rich25_p4.json rich25p4 2 2>&1 | grep -E "^===|trace rows|retrying|Error"
say "RICH2_DONE"
