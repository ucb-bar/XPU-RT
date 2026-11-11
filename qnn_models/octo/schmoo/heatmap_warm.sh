#!/usr/bin/env bash
# Schmoo grid, WARM-STARTED CP-SAT, keeping cpsat's own emitted schedules.
#
# Supersedes heatmap_sweep.sh, which ran cpsat COLD (no --seed-solver) and only kept its
# status. Two consequences that this fixes:
#   1. Cold cpsat gives worse solutions at the same budget. Warm start from a
#      deadline-aware heuristic is what the earlier targeted searches used.
#   2. The heatmap coloured by GREEDY's latency/cadence, and greedy ignores
#      window_duration entirely -- so every column was uniform down the deadline axis
#      and no 2-D structure could appear. cpsat DOES respond to the deadline, so its own
#      schedule is what the heatmap should show.
# The per-cell schedule JSON is what carries latency/cadence, so tags must be unique --
# NOTE the separate `local` statements: bash evaluates every RHS of one `local` before
# assigning, which silently produced an EMPTY tag last time and made 108 cells overwrite
# a single file.
set -u
cd ~/xpurt_sched
source ~/miniforge3/etc/profile.d/conda.sh; conda activate sched
export XPURT_CPSAT_PYTHON=$(which python)
export XPURT_CPSAT_WORKERS=12
BASE=data/toplevel/networks_octo_pipe3fast10_qrb5165.json
mkdir -p gridw
cell () {
  local p=$1
  local w=$2
  local tag="GW_p${p}_w${w}"
  local json=data/toplevel/networks_octo_pareto_${tag}.json
  python3 - "$BASE" "$json" "$p" "$w" <<'PY'
import json,sys
b,o,p,w=sys.argv[1],sys.argv[2],int(sys.argv[3]),int(sys.argv[4])
d=json.load(open(b)); d["networks"]["octo"]["period"]=p; d["networks"]["octo"]["window_duration"]=w
d["_comment"]=f"schmoo grid (warm cpsat): period={p}, window={w}, 10 instances."
json.dump(d,open(o,"w"),indent=1)
PY
  timeout -s KILL 600 python scripts/run_xpurt_schedule.py --networks-json "$json" \
      --solver cpsat --seed-solver heft_edf --cpsat-time-limit 120 --profiled \
      > gridw/${tag}.log 2>&1
  local st
  st=$(grep -aoE 'cpsat status=[A-Z]+' gridw/${tag}.log | tail -1 | cut -d= -f2)
  echo "p=$p w=$w -> ${st:-NOSTATUS}"
}
export -f cell; export BASE
for p in 100 110 120 130 140 150 165 180 200 220 250 283; do
  for w in 240 260 275 290 305 320 350 400 450; do echo "$p $w"; done
done | xargs -P 8 -n 2 bash -c 'cell "$0" "$1"'
echo GRIDW_DONE
