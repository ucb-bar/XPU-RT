#!/usr/bin/env bash
# The 45 Hz census the earlier one should have been: the solver-derived arm against the static
# 6-core ROS 2 partition.
#
# campaign_static6_45 flew `cp3` against xpu_a_cpsat_hard, which is 58.0 ms camera->control -- the
# same latency as cp3's own 56.2 ms. With no latency difference the comparison rested on command
# rate alone, and the rate-injected envelope is flat above the control-rate floor, so it measured
# nothing; the baseline came out ahead at both cruise speeds flown. p45free is the same chain solved
# with nothing pinned at a 22.2 ms camera period: 28.3 ms camera->control at 100.7 Hz, 1012 IME
# dispatches, placement derived rather than instructed.
#
# Only the scheduled arm is flown here; cp3's rows are campaign_static6_45's own and are carried over
# once that census finishes, so both arms keep the same cells, gain and hold.
set -u
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${OUT:-$WT/results/codesign_feedback/campaign_free45}"; T="$WT/results/codesign_feedback/ctrl_traces"
ARMS="${ARMS:-xpu_p45free:$T/xpu_p45free.csv:28.3:0}"
MAX_SIMS="${MAX_SIMS:-3}" NEED_MB="${NEED_MB:-10000}" ARMS="$ARMS" SPEEDS="${SPEEDS:-1.4 1.0 1.8 1.2 1.6}" \
  COURSE=a DENS=0.30 GAIN=0.0055 WALK=0.0 SEEDS=12 SEED0=1000 OUT="$OUT" bash "$WT/scripts/campaign_percep.sh"

SRC="$WT/results/codesign_feedback/campaign_static6_45/campaign.csv"
.venv/bin/python - "$OUT/campaign.csv" "$SRC" <<'PY'
import csv, os, sys
dst, src = sys.argv[1], sys.argv[2]
have = list(csv.DictReader(open(dst))); cols = list(have[0].keys()) if have else None
add = [r for r in csv.DictReader(open(src)) if os.path.basename(r["ctrl_trace"]) == "ros_cp345.csv"] if os.path.exists(src) else []
seen = {(r["seed"], r["cruise_speed"], os.path.basename(r["ctrl_trace"])) for r in have}
add = [r for r in add if (r["seed"], r["cruise_speed"], os.path.basename(r["ctrl_trace"])) not in seen]
with open(dst, "a", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols)
    for r in add:
        w.writerow({k: r.get(k, "") for k in cols})
print(f"carried over {len(add)} baseline rows from campaign_static6_45")
PY
echo "CAMPAIGN_FREE45_DONE $(date +%H:%M:%S)"
