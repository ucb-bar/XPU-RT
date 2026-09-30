#!/usr/bin/env bash
# IDEMPOTENT resume of the strengthening campaign (survives OOM kills on the shared box: each relaunch
# skips cells already complete and continues). Stage A (showdown +seeds) is already done (18/18/18) so
# it is skipped. Stage B fills hil_ablation_v2.csv to 12 seeds/cell WITHOUT wiping existing rows; stage
# C fills hil_ablation_courseB.csv to 6 seeds/cell. Re-run this script as many times as needed.
set -u
WT="${WT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"   # this repo, wherever it is checked out
. "$WT/scripts/env.sh"                                                        # SIM_TREE, ISAAC_PY, RES
CAN="${CAN:-$SIM_TREE}"
PY="$ISAAC_PY"
W=$CAN/sims/models/warehouse/nav_fused_v12_cnn.pt
LOG=$RES/strengthen_campaign; mkdir -p "$LOG/tmp"; export TMPDIR="$LOG/tmp"
SUM=$LOG/CAMPAIGN.log
cd "$CAN" || { echo "canonical cwd missing" | tee -a "$SUM"; exit 1; }
say(){ echo "=== $(date +%H:%M:%S) $* ===" | tee -a "$SUM"; }

# rows already in $csv for a given cruise (col2) + sched_latency (col6, matches int or float form)
have(){ local csv=$1 cru=$2 lat=$3
  [ -f "$csv" ] || { echo 0; return; }
  awk -F, -v c="$cru" -v l="$lat" 'NR>1 && ($2+0==c+0) && ($6+0==l+0){n++} END{print n+0}' "$csv"; }

fly(){ local csv=$1 cru=$2 lat=$3 mom=$4 s0=$5 eps=$6
  timeout 2600 $PY sims/scripts/sweep_rate_demo.py --headless --controller rl --weights "$W" \
    --sim_dt 0.01 --decimation 1 --obstacle_level 8 --prop_density 0.30 \
    --sched_latency_ms "$lat" --percep_hold_ms 0 --moment_scale "$mom" \
    --cruise_speed "$cru" --walk_speed 0.0 --episodes "$eps" --seed "$s0" --max_steps 1800 \
    --sweep-csv "$csv" >> "$LOG/$(basename "$csv" .csv).log" 2>&1; }

say "RESUME: stage A skipped (showdown already 18/18/18)"

# ---- B. envelope grid @12 seeds/cell -> hil_ablation_v2.csv (idempotent) -----------------------
V2=$RES/hil_ablation_v2.csv
say "B resume envelope grid -> $(basename "$V2") (skip cells already >=12 rows)"
for cru in 1.0 1.2 1.4 1.6 1.8; do for lat in 8 18 28 38; do
  h=$(have "$V2" "$cru" "$lat")
  if [ "$h" -ge 12 ]; then say "B skip cruise=$cru lat=${lat}ms (have $h)"; continue; fi
  say "B cell cruise=$cru lat=${lat}ms (have $h, need 12)"; fly "$V2" "$cru" "$lat" 0.0055 1000 12
done; done
say "B done ($(( $(wc -l < "$V2" 2>/dev/null || echo 1)-1 )) rows)"

# ---- C. course-B envelope @6 seeds/cell -> hil_ablation_courseB.csv (idempotent) ---------------
CB=$RES/hil_ablation_courseB.csv
say "C resume course-B (WAREHOUSE_COURSE=b) -> $(basename "$CB") (skip cells already >=6 rows)"
export WAREHOUSE_COURSE=b
for cru in 1.0 1.2 1.4 1.6 1.8; do for lat in 8 18 28 38; do
  h=$(have "$CB" "$cru" "$lat")
  if [ "$h" -ge 6 ]; then say "C skip cruise=$cru lat=${lat}ms (have $h)"; continue; fi
  say "C cell cruise=$cru lat=${lat}ms (have $h, need 6)"; fly "$CB" "$cru" "$lat" 0.0055 1000 6
done; done
unset WAREHOUSE_COURSE
say "C done ($(( $(wc -l < "$CB" 2>/dev/null || echo 1)-1 )) rows)"
say "RESUME COMPLETE — review, then finalize_v2.sh"
