#!/usr/bin/env bash
# The shard-costed XPU-RT arm over the course/density cells the 8-core ROS arm covers, so the
# two can be compared seed for seed. Same cells, speeds and seeds as campaign_x2cells.sh.
set -u
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${OUT:-$WT/results/codesign_feedback/campaign_shardcells}"
TRACE="${TRACE:-$WT/results/codesign_feedback/ctrl_traces/xpu_shard45.csv}"
LAT="${LAT:-36.0}"
CELLS="${CELLS:-a:0.30 a:0.20 a:0.40 b:0.30 b:0.20 b:0.40 c:0.30 c:0.40}"
for cell in $CELLS; do
  co=${cell%%:*}; de=${cell#*:}
  echo "=== $(date +%H:%M:%S) course $co density $de"
  ARMS="xpu_shard:$TRACE:$LAT" SPEEDS="${SPEEDS:-1.0 1.2 1.4 1.6 1.8}" \
    COURSE=$co DENS=$de GAIN=0.0055 WALK=0.0 SEEDS=12 SEED0=1000 OUT="$OUT" \
    bash "$WT/scripts/campaign_percep.sh"
done
echo "CAMPAIGN_SHARDCELLS_DONE $(date +%H:%M:%S)"
