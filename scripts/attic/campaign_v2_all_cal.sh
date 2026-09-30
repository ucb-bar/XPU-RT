#!/usr/bin/env bash
# The same campaign under the calibrated gain law: each arm's controller gain set by the cadence
# it replays (moment_scale = 0.5 / outputs-per-second over the loop), so no cadence is flown at
# an authority tuned for another. Runs after the fixed-gain drivers.
set -u; cd "$(dirname "$0")/.."
L=results/codesign_feedback/campaign_v2; SP="1.8 1.6 1.5 1.4 1.3 1.2 1.1 1.0"
while ! grep -q ALL_DRIVERS_DONE $L/all_drivers.log 2>/dev/null; do sleep 600; done
gain_of() { python3 - "$1" <<'PY'
import sys
ts=[]; span=None
for line in open(sys.argv[1]):
    line=line.strip()
    if line.startswith("#"):
        for tok in line[1:].split():
            if tok.startswith("span_ms="): span=float(tok.split("=")[1])
        continue
    if line and not line.startswith("t_ms"): ts.append(float(line))
hz = 1000.0*len(ts)/span
print(f"{0.5/hz:.5f}")
PY
}
T=results/codesign_feedback/ctrl_traces
( for arm in "xpu_cpsat_hard:$T/xpu_a_cpsat_hard.csv" "ros_vanilla:$T/ros_vanilla45.csv"; do GAIN=$(gain_of ${arm#*:}) ARMS="$arm" SPEEDS="$SP" bash scripts/campaign_v2b.sh; done; echo CAL1_DONE ) > $L/cal_driver1.log 2>&1 &
( for arm in "ros_vanilla4t:$T/ros_vanilla4t45.csv"; do GAIN=$(gain_of ${arm#*:}) ARMS="$arm" SPEEDS="$SP" bash scripts/campaign_v2b.sh; done; echo CAL2_DONE ) > $L/cal_driver2.log 2>&1 &
( for arm in "ros_vanilla4:$T/ros_vanilla445.csv" "xpu_greedy:$T/xpu_a_greedy.csv"; do GAIN=$(gain_of ${arm#*:}) ARMS="$arm" SPEEDS="$SP" bash scripts/campaign_v2b.sh; done; echo CAL3_DONE ) > $L/cal_driver3.log 2>&1 &
wait; echo CAL_ALL_DONE
