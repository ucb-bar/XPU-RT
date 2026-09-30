#!/usr/bin/env bash
# The baseline's half of the breaking-point campaign, run as its own simulator so the ROS 2 curve lands while
# campaign_break.sh is still flying the XPU-RT speeds (campaign_percep.sh skips any cell already in the CSV, so
# the two scripts share campaign_break/campaign.csv without re-flying each other's cells; a cell both start
# within the same hour is flown twice and the figure's equal-replicate rule keeps the first). Waits for a
# simulator slot (< 3 simulators, >= 7 GB free), then 1.2 -> 1.6 -> 1.0 -> 1.4 -> 2.0 -> 0.8 -> 1.8 m/s.
#   nohup bash scripts/campaign_break_ros.sh > results/codesign_feedback/campaign_break_ros.log 2>&1 &
. "$(dirname "$0")/env.sh"
set -u; cd "$(dirname "$0")/.."
R=$PWD/results/codesign_feedback; T=$R/ctrl_traces
gpu_room() { local free n; free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1); n=$(pgrep -fc 'sweep_rate_dem[o]\.py|record_sensor_dem[o]\.py'); [ "$free" -ge 7000 ] && [ "$n" -lt 6 ]; }
until gpu_room; do sleep 120; done
echo "=== $(date +%H:%M:%S) slot free, ROS 2 vanilla in the 1.7 m scene with its latency"
export WAREHOUSE_PERSON_H=1.7
OUT=$R/campaign_break ARMS="ros_vanilla4:$T/ros_vanilla445.csv:242:0" SPEEDS="1.2 1.6 1.0 1.4 2.0 0.8 1.8" bash scripts/campaign_percep.sh 2>&1 | grep -E '^\[SWEEP\]|^=== |^skip' | cut -c1-150
echo CAMPAIGN_BREAK_ROS_DONE
