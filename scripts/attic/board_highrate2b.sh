#!/usr/bin/env bash
set -u; cd "$(dirname "$0")/.."
say() { echo "=== $(date +%H:%M:%S) $*"; }
while ! grep -q "RICH2_DONE" results/codesign_feedback/xpurt_long/rich2.log 2>/dev/null; do sleep 30; done
for cfg in "best60_alt2:best60alt2" "best75_alt2:best75alt2" "best90_alt1:best90alt1"; do
  IFS=: read -r S L <<<"$cfg"; say "$L"; scripts/run_xpurt_long.sh schedules/$S.json $L 2 2>&1 | grep -E "^===|trace rows|re-pulled|re-running"
done
say "HIGHRATE2B_DONE"
