#!/usr/bin/env bash
# The per-node-pinning configuration at ITS OWN camera rate.
#
# The Tier A panel I is titled "measured at the 45 Hz camera" and draws "static - 6 cores": a
# four-hart YOLO pool, nav and control each pinned to one more hart, two harts marked idle. `cp3` is
# that deployment; at 45 Hz the board puts its pool at 84-98 % busy and its two spare harts at 0.2 %
# and 0.6 %, and control -- chained to the goal -- comes out at 38.8 Hz, right at the rate floor
# rather than far below it. That is the rate at which the baseline survives into the course, which is
# what the Tier A showdown drew.
#
# Only the baseline is flown: the scheduled arm's rows at this gain, latency and hold are already in
# campaign_submitted and are carried over unchanged, so the counts stay equal and nothing is re-flown.
set -u
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${OUT:-$WT/results/codesign_feedback/campaign_static6_45}"; T="$WT/results/codesign_feedback/ctrl_traces"
ARMS="${ARMS:-ros_static6:$T/ros_cp345.csv:56.2:0}"
MAX_SIMS="${MAX_SIMS:-3}" NEED_MB="${NEED_MB:-10000}" ARMS="$ARMS" SPEEDS="${SPEEDS:-1.4 1.0 1.8 1.2 1.6}" \
  COURSE=a DENS=0.30 GAIN=0.0055 WALK=0.0 SEEDS=12 SEED0=1000 OUT="$OUT" bash "$WT/scripts/campaign_percep.sh"

.venv/bin/python - "$OUT/campaign.csv" "$WT/results/codesign_feedback/campaign_submitted/campaign.csv" <<'PY'
import csv, os, sys
dst, src = sys.argv[1], sys.argv[2]
have = list(csv.DictReader(open(dst))); cols = list(have[0].keys()) if have else None
add = [r for r in csv.DictReader(open(src))
       if os.path.basename(r["ctrl_trace"]) == "xpu_a_cpsat_hard.csv"
       and abs(float(r["moment_scale"]) - 0.0055) < 1e-9 and abs(float(r["percep_hold_ms"])) < 1e-9]
seen = {(r["seed"], r["cruise_speed"], os.path.basename(r["ctrl_trace"])) for r in have}
add = [r for r in add if (r["seed"], r["cruise_speed"], os.path.basename(r["ctrl_trace"])) not in seen]
with open(dst, "a", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols)
    for r in add:
        w.writerow({k: r.get(k, "") for k in cols})
print(f"carried over {len(add)} scheduled-arm rows at gain 0.0055, hold 0")
PY
echo "CAMPAIGN_STATIC6_45_DONE $(date +%H:%M:%S)"
