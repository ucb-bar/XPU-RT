#!/usr/bin/env bash
set -u; cd "$(dirname "$0")/.."
say() { echo "=== $(date +%H:%M:%S) $*"; }
while ! grep -q "HIGHRATE2B_DONE" results/codesign_feedback/xpurt_long/highrate2b.log 2>/dev/null; do sleep 30; done
say "XPU-RT heavier stack"; scripts/run_xpurt_long.sh schedules/rich45_alt2.json rich45alt2 3 2>&1 | grep -E "^===|trace rows|re-pulled|re-running"
scripts/run_xpurt_long.sh schedules/rich25_p4.json rich25p4 2 2>&1 | grep -E "^===|trace rows|re-pulled|re-running"
say "RICH3_DONE"
