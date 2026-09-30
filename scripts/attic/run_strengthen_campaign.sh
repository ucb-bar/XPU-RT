#!/usr/bin/env bash
# HIL data-strengthening campaign — runs the CANONICAL sim (has --walk_speed etc.), writes into the
# paper worktree's results dir, into NEW files (never overwrites committed data; the v1 120-flight
# CSV is already archived as hil_ablation_v1_120flights_fixedgain.csv). Single GPU -> sequential.
# Fast-first order: A showdown (+12 seeds, ~1h) -> B fresh envelope grid @12/cell (~8h) ->
#   C course-B envelope @6/cell (~4h). Each stage logs per cell; NO figure regen/swap here (done
#   after review). Re-runnable: showdown appends; the two grid CSVs are removed+rebuilt per launch.
set -u
. "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/scripts/env.sh"   # SIM_TREE, ISAAC_PY: see scripts/env.local.sh.example
CAN="${CAN:-$SIM_TREE}"             # canonical repo = the real sim + course-B edit
WT="${WT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"   # this repo, wherever it is checked out                    # paper worktree = committed results live here
RES=$WT/results/codesign_feedback; CV=$RES/crash_verify
PY="$ISAAC_PY"
W=$CAN/sims/models/warehouse/nav_fused_v12_cnn.pt
LOG=$RES/strengthen_campaign; mkdir -p "$LOG/tmp"; export TMPDIR="$LOG/tmp"
SUM=$LOG/CAMPAIGN.log; : > "$SUM"
cd "$CAN" || { echo "canonical cwd missing" | tee -a "$SUM"; exit 1; }
say(){ echo "=== $(date +%H:%M:%S) $* ===" | tee -a "$SUM"; }

# one flight cell: fly(csv, cruise, sched_lat_ms, moment, seed0, episodes [, extra args...])
fly(){ local csv=$1 cru=$2 lat=$3 mom=$4 s0=$5 eps=$6; shift 6
  timeout 2600 $PY sims/scripts/sweep_rate_demo.py --headless --controller rl --weights "$W" \
    --sim_dt 0.01 --decimation 1 --obstacle_level 8 --prop_density 0.30 \
    --sched_latency_ms "$lat" --percep_hold_ms 0 --moment_scale "$mom" \
    --cruise_speed "$cru" --walk_speed 0.0 --episodes "$eps" --seed "$s0" --max_steps 1800 \
    --sweep-csv "$csv" "$@" >> "$LOG/$(basename "$csv" .csv).log" 2>&1
}

# ---- A. SHOWDOWN +12 seeds (seeds 1006..1017), APPEND to the existing 6-seed CSVs -------------
say "A showdown +12 seeds -> new_xpu/new_ros50/new_ros (append, 6->18)"
fly "$CV/new_xpu.csv"   1.4 6.30  0.0055 1006 12; say "A xpu done   (now $(( $(wc -l < "$CV/new_xpu.csv")-1 )) rows)"
fly "$CV/new_ros33.csv" 1.4 24.94 0.0055 1006 12; say "A ros50 done (now $(( $(wc -l < "$CV/new_ros50.csv")-1 )) rows)"
fly "$CV/new_ros17.csv" 1.4 53.22 0.0055 1006 12; say "A ros25 done (now $(( $(wc -l < "$CV/new_ros.csv")-1 )) rows)"

# ---- B. FRESH FULL ENVELOPE GRID @12 seeds/cell -> hil_ablation_v2.csv (supersedes v1) ---------
V2=$RES/hil_ablation_v2.csv; rm -f "$V2"
say "B fresh envelope grid 5 speeds x 4 rates x 12 seeds -> $(basename "$V2")"
for cru in 1.0 1.2 1.4 1.6 1.8; do for lat in 8 18 28 38; do
  say "B cell cruise=$cru lat=${lat}ms"; fly "$V2" "$cru" "$lat" 0.0055 1000 12
done; done
say "B done ($(( $(wc -l < "$V2")-1 )) rows in $(basename "$V2"))"

# ---- C. COURSE B (env-var) envelope @6 seeds/cell -> hil_ablation_courseB.csv ------------------
CB=$RES/hil_ablation_courseB.csv; rm -f "$CB"
say "C course-B envelope 5x4x6 (WAREHOUSE_COURSE=b) -> $(basename "$CB")"
export WAREHOUSE_COURSE=b
for cru in 1.0 1.2 1.4 1.6 1.8; do for lat in 8 18 28 38; do
  say "C cell cruise=$cru lat=${lat}ms"; fly "$CB" "$cru" "$lat" 0.0055 1000 6
done; done
unset WAREHOUSE_COURSE
say "C done ($(( $(wc -l < "$CB")-1 )) rows in $(basename "$CB"))"
say "CAMPAIGN COMPLETE — review, then swap hil_ablation.csv<-v2 + regen figures"
