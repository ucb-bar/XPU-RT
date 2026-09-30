#!/usr/bin/env bash
# The 30 Hz camera census, each arm flown at the gain its own control rate calls for.
#
# WHY. scripts/campaign_rate30.sh flies both arms at one fixed moment_scale (0.0055). The controller's
# moment gain is a per-step impulse, so the authority it applies per second scales with how often it
# runs: a rate-independent constant over-drives the fast arm or under-drives the slow one. The law the
# gain-controlled grid uses (results/codesign_feedback/gain_controlled/) is moment_scale = 0.5 / eff_hz,
# i.e. each arm is given the gain its measured command rate calls for. Under that law the baseline at
# 30.0 Hz is flown at 0.0167 rather than 0.0055 -- three times the authority the shared constant gave
# it -- so any separation that survives is not an artefact of gain.
#
# The grid that established the law reports, at each rate's own gain: 0/60 at 20 Hz, 0/60 at 25 Hz,
# 0/48 at 33 Hz, 9/60 at 50 Hz and 16/60 at 100 Hz, so the control-rate floor is a property of the
# closed loop rather than of the constant.
#
# Each arm still replays its own board-measured camera->control latency and holds the goal for one
# camera period (33.3 ms), exactly as campaign_rate30.sh does.
#
#   scripts/campaign_rate30_gain.sh       env ARMLIST SPEEDS SEEDS MAX_SIMS NEED_MB OUT
#
# ARMLIST entries are name:trace:latency_ms:hold_ms:gain -- the gain is explicit per arm rather than
# derived here, so the value flown is the value recorded in this file and in the campaign CSV.
set -u
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${OUT:-$WT/results/codesign_feedback/campaign_rate30_gain}"
T="$WT/results/codesign_feedback/ctrl_traces"
# ros_vanilla4x230.csv measures eff_hz 30.00 -> 0.5/30.00 = 0.01667
# xpu_a30_cpsat.csv    measures eff_hz 100.01 -> 0.5/100.01 = 0.00500
ARMLIST="${ARMLIST:-ros_x2_30g:$T/ros_vanilla4x230.csv:31.4:33.3:0.01667 xpu_a30g:$T/xpu_a30_cpsat.csv:55.2:33.3:0.00500}"
CELLS="${CELLS:-a:0.30}"
# 1.6 is kept so the upper edge of the envelope is shown rather than cropped; the gain grid finds
# every arm failing at 1.8, so that speed separates nothing and is not flown here.
SPEEDS="${SPEEDS:-1.2 1.0 1.4 1.6}"
for cell in $CELLS; do
  co=${cell%%:*}; de=${cell#*:}
  for a in $ARMLIST; do
    IFS=: read -r nm tr lat hold gain <<< "$a"
    MAX_SIMS="${MAX_SIMS:-3}" NEED_MB="${NEED_MB:-10000}" ARMS="$nm:$tr:$lat:$hold" SPEEDS="$SPEEDS" \
      COURSE=$co DENS=$de GAIN="$gain" WALK=0.0 SEEDS="${SEEDS:-12}" SEED0=1000 OUT="$OUT" \
      bash "$WT/scripts/campaign_percep.sh" &
  done
  wait
done
echo "CAMPAIGN_RATE30_GAIN_DONE $(date +%H:%M:%S)"
