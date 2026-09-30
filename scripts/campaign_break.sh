#!/usr/bin/env bash
# The breaking-point campaign: people 1.7 m (overflown, so the crates set the difficulty), both arms replaying their
# board cadence AND camera-to-control latency, cruise 0.8-2.0 m/s, 12 seeds a cell. The hypothesis it tests: at low
# speed a 242 ms decision delay is tolerable and both arms complete; at some speed the baseline's blind distance
# exceeds its margin while 57 ms still fits; at the top both fail.  One simulator, GPU-room gated; cells skipped once flown.
#   nohup bash scripts/campaign_break.sh > results/codesign_feedback/campaign_break.log 2>&1 &
. "$(dirname "$0")/env.sh"
set -u; cd "$(dirname "$0")/.."
R=$PWD/results/codesign_feedback; T=$R/ctrl_traces
export WAREHOUSE_PERSON_H=1.7
OUT=$R/campaign_break ARMS="xpu_cpsat:$T/xpu_a_cpsat_hard.csv:56.8:0 ros_vanilla4:$T/ros_vanilla445.csv:242:0" SPEEDS="1.2 1.6 1.0 1.4 2.0 0.8 1.8" bash scripts/campaign_percep.sh 2>&1 | grep -E '^\[SWEEP\]|^=== ' | cut -c1-150
OUT=$R/campaign_break ARMS="xpu_greedy:$T/xpu_a_greedy.csv:748:0" SPEEDS="1.2 1.6 1.0" bash scripts/campaign_percep.sh 2>&1 | grep -E '^\[SWEEP\]|^=== ' | cut -c1-150
echo CAMPAIGN_BREAK_DONE
