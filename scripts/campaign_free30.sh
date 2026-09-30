#!/usr/bin/env bash
# The 30 Hz census for the figure's two arms, each flown at the gain its own command rate calls for.
#
# The baseline is ROS 2 partitioned by hand the way a deployment has to be: perception on a 4-hart
# pool over the P cores, navigation sharded across the E cluster, control sharing those harts and
# still running in the goal callback (scripts/ros_traced_matrix.sh cp3n4). The scheduled arm is given
# the same machine with NO placement instruction at all -- no allowed_machines, no machine_width --
# and the solver chooses (data/toplevel/wh_chain30_free.json). Neither arm leaves a hart idle.
#
# Both replay their own board-measured camera-to-control latency and hold the goal for one camera
# period. moment_scale = 0.5 / eff_hz per arm, so a separation that survives is not a gain artefact.
set -u
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
T="$WT/results/codesign_feedback/ctrl_traces"
OUT="${OUT:-$WT/results/codesign_feedback/campaign_free30}"
# ros_cp3n430.csv measures 30.0 outputs/s -> 0.5/30.0 = 0.01667
# xpu_p30free.csv measures 100.2 outputs/s -> 0.5/100.2 = 0.00500
ARMLIST="${ARMLIST:-ros_cp3n4_30:$T/ros_cp3n430.csv:30.1:33.3:0.01667 xpu_free30:$T/xpu_p30free.csv:26.8:33.3:0.00500}" \
  SPEEDS="${SPEEDS:-1.2 1.0 1.4 1.6}" SEEDS="${SEEDS:-12}" MAX_SIMS="${MAX_SIMS:-3}" OUT="$OUT" \
  bash "$WT/scripts/campaign_rate30_gain.sh"
echo "CAMPAIGN_FREE30_DONE $(date +%H:%M:%S)"
