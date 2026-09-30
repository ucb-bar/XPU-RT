#!/usr/bin/env bash
# Panel D's mechanism runs for any pair of deployments: one recorded flight per seed per condition with the
# commanded wrench logged, each condition replaying the control-output cadence measured on the K1 for it, at
# the display cruise speed, no latency or goal hold (the panel isolates what the control cadence alone does).
# Totals via scripts/flight_energy_model.py.
#
#   CONDS="xpu_cpsat:<trace> ros_8core:<trace>" CRUISE=1.0 ER=<dir> OUTCSV=<csv> scripts/run_energy_pair.sh
set -u
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; RES="$WT/results/codesign_feedback"
. "$WT/scripts/env.sh"
PY="$ISAAC_PY"; W=$WT/sims/models/warehouse/nav_fused_v12_cnn.pt
CONDS="${CONDS:?name:trace ...}"; CRU="${CRUISE:?}"; ER="${ER:?}"; OUTCSV="${OUTCSV:?}"
SEEDS="${SEEDS:-1000 1001 1002 1003 1004 1005}"; DENS="${DENS:-0.30}"; GAIN="${GAIN:-0.0055}"
mkdir -p "$ER/tmp"; export TMPDIR="$ER/tmp"; LOG=$ER/ENERGY.log
say(){ echo "=== $(date +%H:%M:%S) $* ===" | tee -a "$LOG"; }
gpu_room() { local free n; free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1); n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c .); [ "$free" -ge "${NEED_MB:-6500}" ] && [ "$n" -lt "${MAX_SIMS:-5}" ]; }
for c in $CONDS; do
  cond=${c%%:*}; trace=${c#*:}; [ -f "$trace" ] || { echo "no trace $trace"; exit 2; }
  for s in $SEEDS; do
    out="$ER/${cond}_s${s}"
    if [ -f "$out/figure_data.npz" ]; then say "have $cond seed $s"; continue; fi
    # the same serialised admission as the flight campaigns (scripts/campaign_percep.sh)
    exec 8>"$RES/gpu_admit.lock"; flock 8
    until gpu_room; do sleep 30; done
    ( sleep "${ADMIT_SETTLE:-150}"; flock -u 8 ) & settle=$!
    say "energy: $cond seed $s (trace $(basename $trace), cruise $CRU)"
    (cd "$SIM_TREE" && timeout 900 $PY sims/scripts/sweep_rate_demo.py --headless --controller rl --weights "$W" \
      --sim_dt 0.01 --decimation 1 --obstacle_level 8 --prop_density $DENS --percep_hold_ms 0 --moment_scale $GAIN \
      --cruise_speed "$CRU" --walk_speed 0.0 --episodes 1 --seed "$s" --max_steps 1800 --ctrl_trace "$trace" \
      --dump_figure_data "$out" > "$ER/${cond}_s${s}.log" 2>&1 8>&-)
    kill $settle 2>/dev/null; flock -u 8
    if grep -qE "PhysX Internal CUDA error|Simulation cannot continue|GPU is out of memory|CUDA error, code" "$ER/${cond}_s${s}.log"; then
      say "GPU fault in $cond seed $s: dump removed, flown again next pass"; rm -rf "$out"
    fi
  done
done
say "modeled propulsive energy from the logged wrench"
$PY "$WT/scripts/flight_energy_model.py" --glob "$ER/*_s*/figure_data.npz" --out "$OUTCSV" 2>&1 | tee -a "$LOG"
say "ENERGY_PAIR_DONE -> $OUTCSV"
