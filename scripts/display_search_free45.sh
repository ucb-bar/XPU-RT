#!/usr/bin/env bash
# The displayed pair for the per-node-pinning configuration at its own 45 Hz camera.
#
# Baseline `cp3`: the static 6-core partition the Tier A panel I drew, 56.2 ms camera->goal with
# control chained to the goal at 38.8 Hz -- at the rate floor rather than far below it, which is why
# this is the configuration in which the baseline survives into the course. Scheduled arm: the CP-SAT
# schedule at 56.8 ms with control on its own 100 Hz slot. Both fly moment_scale 0.0055, the gain the
# controller is deployed with, so neither arm is favoured by the number.
#
# display_same_env.sh flies ONE episode per seed with layout_seed = seed, so a census cell is a
# different scene and a census outcome never carries over: the pair is accepted on what the display
# flight itself does, and every attempt is logged. The census is used only to ORDER the search, which
# is why a cell that fails here is simply passed over rather than reported as a disagreement.
#
#   scripts/display_search_static6.sh          env CRUISES OD MAX_SIMS NEED_MB
set -u
cd "$(dirname "$0")/.."
OD=${OD:-$PWD/results/codesign_feedback/campaign_free45/display}; mkdir -p "$OD"
T=$PWD/results/codesign_feedback/ctrl_traces
C=results/codesign_feedback/campaign_free45/campaign.csv

# Seed order per cruise, from the census: cells where the baseline crashed having entered the course
# come first, and among those the ones where the scheduled arm also completed.
order_for() {
  .venv/bin/python - "$1" <<'PY'
import csv, sys, os, collections
cru=sys.argv[1]; p="results/codesign_feedback/campaign_free45/campaign.csv"
allseeds=[str(s) for s in range(1000,1012)]
if not os.path.exists(p): print(" ".join(allseeds)); raise SystemExit
rows=[r for r in csv.DictReader(open(p)) if abs(float(r["cruise_speed"])-float(cru))<1e-6]
by=collections.defaultdict(dict)
for r in rows: by[r["seed"]][r["ctrl_trace"]]=r
def rank(s):
    v=by.get(s,{})
    ros=next((r for k,r in v.items() if k.startswith("ros")), None)
    xpu=next((r for k,r in v.items() if k.startswith("xpu")), None)
    if not ros or not xpu: return 6
    rg=float(ros["gates_passed"]); rc=ros["outcome"]=="crash"; xs=xpu["outcome"]=="success"
    if rc and rg>=2 and xs:    return 0
    if rc and rg>=2:           return 1
    if rc and rg==1 and xs:    return 2
    if rc and rg==1:           return 3
    if rc:                     return 4
    return 5
print(" ".join(sorted(allseeds, key=rank)))
PY
}

for cru in ${CRUISES:-1.4 1.6 1.2 1.0}; do
  echo "=== $(date +%H:%M:%S) search cruise $cru"
  SEEDS="$(order_for "$cru")"
  echo "    seed order from the census: $SEEDS"
  CRUISE=$cru SEEDS="$SEEDS" DENS=0.30 \
    XT=$T/xpu_p45free.csv RT=$T/ros_cp345.csv XLAT=28.3 RLAT=56.2 XHOLD=0 RHOLD=0 \
    XGAIN=0.0055 RGAIN=0.0055 \
    ROS_GATES='[12]' RENDER=0 MAX_SIMS=${MAX_SIMS:-3} NEED_MB=${NEED_MB:-10000} \
    OUTDIR="$OD/search_c$cru" bash scripts/display_same_env.sh > "$OD/search_c$cru.log" 2>&1
  grep -E "^  (xpu|ros) |PAIR" "$OD/search_c$cru.log" | tail -20
  grep -q PAIR_SEED "$OD/search_c$cru.log" && break
done
echo "DISPLAY_FREE45_SEARCH_DONE $(date +%H:%M:%S)"
