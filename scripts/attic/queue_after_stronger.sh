#!/usr/bin/env bash
# After the stronger-story queue: the slower end of the speed axis in the tall-people scene (0.8 m/s,
# both main arms, cadence-only and with each arm's measured latency), and twelve more seeds on the
# latency-replay cell at 1.0 m/s — the completion counts at 1.0 m/s vary from 1/12 to 6/12 between
# near-identical cells, so the display cell needs more flights behind it.
set -u; cd "$(dirname "$0")/.."
R=results/codesign_feedback; T=$R/ctrl_traces; X=$T/xpu_a_cpsat_hard.csv; V=$T/ros_vanilla445.csv
until grep -q QUEUE_STRONGER_DONE $R/queue_stronger.log 2>/dev/null; do sleep 300; done
say(){ echo "=== $(date +%H:%M:%S) $*"; }
say "0.8 m/s, tall scene"
OUT=$R/campaign_percep ARMS="xpu_cpsat:$X:0 ros_vanilla4:$V:0 xpu_cpsat:$X:56.8 ros_vanilla4:$V:242" SPEEDS="0.8" bash scripts/campaign_percep2.sh
say "seeds 1012-1023, latency replay at 1.0"
OUT=$R/campaign_seeds24 SEED0=1012 ARMS="xpu_cpsat:$X:56.8 ros_vanilla4:$V:242" SPEEDS="1.0" bash scripts/campaign_percep2.sh
say "QUEUE_AFTER_STRONGER_DONE"
