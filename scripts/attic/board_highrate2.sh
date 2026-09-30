#!/usr/bin/env bash
# Complete traces for the high-rate XPU-RT layouts (the first pass lost rows to ssh drops).
set -u; cd "$(dirname "$0")/.."
say() { echo "=== $(date +%H:%M:%S) $*"; }
while ! grep -q "RICH_DONE" results/codesign_feedback/xpurt_long/rich.log 2>/dev/null; do sleep 60; done
for cfg in "best60_alt2:best60alt2" "best60_alt1:best60alt1" "best75_alt2:best75alt2" "best90_alt1:best90alt1" "best45_alt1:best45alt1"; do
  IFS=: read -r S L <<<"$cfg"; say "$L"; scripts/run_xpurt_long.sh schedules/$S.json $L 2 2>&1 | grep -E "^===|trace rows|retrying"
done
say "HIGHRATE2_DONE"
