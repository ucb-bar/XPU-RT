#!/usr/bin/env bash
# Panel D's power-versus-command-rate curve, each rate flown at the gain that rate calls for.
#
# WHY. results/codesign_feedback/flight_energy_r30_rates.csv flies every rate at one fixed moment_scale
# (0.0055). The controller's moment gain is a per-step impulse, so a rate-independent constant under-drives
# a slow arm: the tracking error it fails to correct accumulates, the next command is large, and the
# commanded moment grows for a reason that is partly the constant rather than the rate. That file therefore
# cannot separate "the baseline commands at 30 Hz" from "the baseline was flown at our gain".
#
# This sweep applies the same law the gain-controlled grid uses, moment_scale = 0.5 / eff_hz, with eff_hz
# measured from each condition's own board trace rather than from its nominal name (the 60 Hz arm measures
# 63.0 Hz on the board, the 45 Hz arm 45.3). 36 and 40 Hz are flown as well as 25/30/45/60 because the
# fixed-gain curve turns between 30 and 45, and a turn stated from two endpoints is an assertion rather
# than a measurement.
#
#   scripts/energy_rate_sweep_gain.sh        env CRUISE SEEDS ER OUTCSV MAX_SIMS
#
# CONDLIST entries are name:trace:gain. The gain is written here rather than derived at run time, so the
# value flown is the value recorded in this file and in the per-flight logs.
set -u
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; RES="$WT/results/codesign_feedback"
T="$RES/ctrl_traces"
ER="${ER:-$RES/energy_runs_r30_gain}"
OUTCSV="${OUTCSV:-$RES/flight_energy_r30_rates_gain.csv}"
# measured eff_hz -> 0.5/eff_hz
CONDLIST="${CONDLIST:-\
xpu_cpsat_r30:$T/xpu_a30_cpsat.csv:0.00499 \
ros_x2_60:$T/ros_vanilla4x260.csv:0.00794 \
ros_x2_45:$T/ros_vanilla4x245.csv:0.01105 \
ros_x2_40:$T/ros_vanilla4x240.csv:0.01244 \
ros_x2_36:$T/ros_vanilla4x236.csv:0.01386 \
ros_x2_30:$T/ros_vanilla4x230.csv:0.01667 \
ros_x2_25:$T/ros_vanilla4x225.csv:0.01998}"
for c in $CONDLIST; do
  nm=${c%%:*}; rest=${c#*:}; tr=${rest%:*}; gn=${rest##*:}
  echo "=== $(date +%H:%M:%S) energy-rate $nm  trace=$(basename "$tr")  gain=$gn ==="
  CONDS="$nm:$tr" CRUISE="${CRUISE:-1.8}" SEEDS="${SEEDS:-1000 1001 1002 1003 1004 1005}" \
    GAIN="$gn" ER="$ER" OUTCSV="$OUTCSV" MAX_SIMS="${MAX_SIMS:-3}" NEED_MB="${NEED_MB:-6500}" \
    bash "$WT/scripts/run_energy_pair.sh"
done
echo "ENERGY_RATE_SWEEP_GAIN_DONE $(date +%H:%M:%S) -> $OUTCSV"
