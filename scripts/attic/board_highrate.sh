#!/usr/bin/env bash
# Beyond the design rate: XPU-RT layouts and every ROS arm at 60 Hz and above.
set -u; cd "$(dirname "$0")/.."
say() { echo "=== $(date +%H:%M:%S) $*"; }
while pgrep -f "ros_traced_matri[x]|run_xpurt_lon[g]" >/dev/null; do sleep 10; done
for cfg in "best60_alt2:best60alt2" "best60_alt1:best60alt1" "best45_alt1:best45alt1" "best75_alt2:best75alt2" "best90_alt1:best90alt1"; do
  IFS=: read -r S L <<<"$cfg"; say "$L"; scripts/run_xpurt_long.sh schedules/$S.json $L 2 2>&1 | grep -E "^===|trace rows|retrying"
done
say "ROS arms at 60 / 75 / 90 Hz"
for arm in ship spin p3 multi cspin cp3 smte; do RATES="60 75 90" scripts/ros_traced_matrix.sh $arm 1 2>&1 | grep -E "ROS_TRACED|MATRIX_DONE|error|incomplete"; done
say "HIGHRATE_DONE"
