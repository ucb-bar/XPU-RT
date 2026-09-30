#!/usr/bin/env bash
# The displayed pair for the 30 Hz figure whose placement the solver chooses, at the gain each arm's own
# command rate calls for.
#
# Arms as the census flew them (scripts/campaign_free30.sh): the hand-partitioned ROS 2 baseline,
# 30.1 ms camera->goal with control in the goal callback at 30 Hz, against the scheduled arm given no
# placement instruction, 26.75 ms with control on its own 100 Hz slot. Each replays its own gain --
# 0.01667 at 30 Hz, 0.00500 at 100 Hz -- so the drawn pair is flown exactly as the census was.
#
# What the baseline does at this gain is not what it does at the equal gain: across 96 flights it never
# reaches the first gate, ending the episode either against the floor or still short of it when the step
# budget runs out. The pair is accepted on that -- the baseline does not enter the course while the
# scheduled arm completes it -- rather than on a crash after a gate, which is the equal-gain form's rule
# (scripts/display_search_free30_eq.sh).
#
# display_same_env.sh flies ONE episode per seed with layout_seed = seed, so a census cell is a
# different scene and a census outcome never carries over: the pair is accepted on what the display
# flight itself does, and every attempt is logged. The census is used only to ORDER the search, which
# is why a cell that fails here is simply passed over rather than reported as a disagreement.
#
#   scripts/display_search_free30.sh          env CRUISES OD MAX_SIMS NEED_MB
set -u
cd "$(dirname "$0")/.."
OD=${OD:-$PWD/results/codesign_feedback/campaign_free30/display}; mkdir -p "$OD"
T=$PWD/results/codesign_feedback/ctrl_traces
C=results/codesign_feedback/campaign_free30/campaign.csv

# Seed order per cruise, from the census: cells where the baseline never reached the first gate come
# first -- at this gain that is how it fails, in all 96 of its flights -- and among those the ones where
# the scheduled arm completed the course.
order_for() {
  .venv/bin/python - "$1" <<'PY'
import csv, sys, os, collections
cru=sys.argv[1]; p="results/codesign_feedback/campaign_free30/campaign.csv"
allseeds=[str(s) for s in range(1000,1012)]
if not os.path.exists(p): print(" ".join(allseeds)); raise SystemExit
rows=[r for r in csv.DictReader(open(p)) if abs(float(r["cruise_speed"])-float(cru))<1e-6]
by=collections.defaultdict(dict)
for r in rows: by[r["seed"]][r["ctrl_trace"]]=r
def rank(s):
    v=by.get(s,{})
    ros=next((r for k,r in v.items() if k.startswith("ros")), None)
    xpu=next((r for k,r in v.items() if k.startswith("xpu")), None)
    if not ros or not xpu: return 3
    rg=float(ros["gates_passed"]); rn=ros["outcome"]!="success"; xs=xpu["outcome"]=="success"
    if rn and rg==0 and xs: return 0
    if rn and rg==0:        return 1
    if rn:                  return 2
    return 3
print(" ".join(sorted(allseeds, key=rank)))
PY
}

for cru in ${CRUISES:-1.4 1.6 1.2 1.0}; do
  echo "=== $(date +%H:%M:%S) search cruise $cru"
  SEEDS="$(order_for "$cru")"
  echo "    seed order from the census: $SEEDS"
  CRUISE=$cru SEEDS="$SEEDS" DENS=0.30 \
    XT=$T/xpu_p30free.csv RT=$T/ros_cp3n430.csv XLAT=26.8 RLAT=30.1 XHOLD=33.3 RHOLD=33.3 \
    XGAIN=0.00500 RGAIN=0.01667 \
    ROS_ACCEPT='outcome=(crash|timeout) .*gates=0/4' RENDER=0 MAX_SIMS=${MAX_SIMS:-3} NEED_MB=${NEED_MB:-10000} \
    OUTDIR="$OD/search_c$cru" bash scripts/display_same_env.sh > "$OD/search_c$cru.log" 2>&1
  grep -E "^  (xpu|ros) |PAIR" "$OD/search_c$cru.log" | tail -20
  grep -q PAIR_SEED "$OD/search_c$cru.log" && break
done
echo "DISPLAY_FREE30_SEARCH_DONE $(date +%H:%M:%S)"
