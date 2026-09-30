#!/usr/bin/env bash
# The 20 Hz control-rate cell of the flight envelope (the out-of-the-box ROS 2 cadence), under
# both gain policies, appended to the same CSVs the envelope panel reads. Runs after the
# calibrated-gain grid that is in flight has finished with the GPU.
set -u; WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; . "$WT/scripts/env.sh"; cd "$SIM_TREE"
R="$RES"; PY="$ISAAC_PY"; W=sims/models/warehouse/nav_fused_v12_cnn.pt
mkdir -p $R/gain_controlled/tmp; export TMPDIR=$R/gain_controlled/tmp
while pgrep -f "[g]ain_controlled_grid.sh" > /dev/null; do sleep 120; done
for cru in 1.0 1.2 1.4 1.6 1.8; do
  n=$(python3 -c "import csv; print(sum(1 for r in csv.DictReader(open('$R/hil_ablation.csv')) if r['sched_latency_ms']=='48.0' and abs(float(r['cruise_speed'])-$cru)<1e-9))")
  [ "$n" -ge 12 ] && { echo "skip fixed 20 Hz cruise $cru ($n rows)"; continue; }
  echo "=== $(date +%H:%M:%S) fixed-gain 20 Hz cruise=$cru ==="
  timeout 3000 $PY sims/scripts/sweep_rate_demo.py --headless --controller rl --weights $W --sim_dt 0.01 --decimation 1 --obstacle_level 8 --prop_density 0.30 \
    --sched_latency_ms 48 --percep_hold_ms 0 --moment_scale 0.0055 --cruise_speed $cru --walk_speed 0.0 --episodes 12 --seed 1000 --max_steps 1800 \
    --sweep-csv $R/hil_ablation.csv 2>&1 | grep -E "\[SWEEP\]|Traceback"
done
echo "=== $(date +%H:%M:%S) calibrated-gain 20 Hz cells ==="
bash "$WT/scripts/gain_controlled_grid.sh" 2>&1 | grep -E "cal_20Hz|SWEEP" | head -20
echo ENVELOPE_20HZ_DONE
