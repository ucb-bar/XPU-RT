#!/usr/bin/env bash
# The 30 Hz census against the baseline the SUBMITTED figure drew (docs/Baselines/ros_baseline_tiers.md).
#
# That figure's panel I is labelled "static - 6 cores": four dense lanes of YOLO, navigation and
# control on one further lane each, and two lanes marked "idle - core unused". Its title reads
# "ROS serial on 1 hart"; this census flies the arrangement the lanes draw. `cp3` is that
# deployment, measured: percep pinned 0x1 with a YOLO pool of 4 over harts 0-3, nav pinned 0x10
# (hart 4), control pinned 0x20 (hart 5), harts 6 and 7 never touched -- the board's sampler puts
# them at 0.7 % and 0.6 % busy over the run.
#
# Both arms fly ONE gain, 0.00500, so only the command rate differs; the scheduled arm's 48 rows are
# the rate-correct census's own and are carried over rather than re-flown.
set -u
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
T="$WT/results/codesign_feedback/ctrl_traces"
OUT="${OUT:-$WT/results/codesign_feedback/campaign_static6}"
ARMLIST="${ARMLIST:-ros_cp3_30:$T/ros_cp330.csv:30.7:33.3:0.00500}" \
  SPEEDS="${SPEEDS:-1.6 1.4 1.2 1.0}" SEEDS="${SEEDS:-12}" MAX_SIMS="${MAX_SIMS:-3}" OUT="$OUT" \
  bash "$WT/scripts/campaign_rate30_gain.sh"

.venv/bin/python - "$OUT/campaign.csv" "$WT/results/codesign_feedback/campaign_free30/campaign.csv" <<'PY'
import csv, sys
dst, src = sys.argv[1], sys.argv[2]
have = list(csv.DictReader(open(dst))); cols = have[0].keys() if have else None
add = [r for r in csv.DictReader(open(src))
       if r["ctrl_trace"].endswith("xpu_p30free.csv") and abs(float(r["moment_scale"]) - 0.005) < 1e-9]
seen = {(r["seed"], r["cruise_speed"], r["ctrl_trace"]) for r in have}
add = [r for r in add if (r["seed"], r["cruise_speed"], r["ctrl_trace"]) not in seen]
with open(dst, "a", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(cols))
    for r in add:
        w.writerow({k: r.get(k, "") for k in cols})
print(f"carried over {len(add)} scheduled-arm rows at gain 0.00500")
PY
echo "CAMPAIGN_STATIC6_DONE $(date +%H:%M:%S)"
