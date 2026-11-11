#!/usr/bin/env bash
# Re-solve the 13 cells still UNKNOWN after the 900 s deep pass, at a 3600 s budget.
# UNKNOWN is budget exhaustion, NOT a proof of infeasibility.  All 13 sit at
# p=100/110/120 -- the fastest cadences, where the measured cadence effect is strongest.
# 13 cells run concurrently at 7 CP-SAT workers each (91 of 96 cores).
set -u
cd ~/xpurt_sched
source ~/miniforge3/etc/profile.d/conda.sh; conda activate sched
export XPURT_CPSAT_PYTHON=$(which python)
export XPURT_CPSAT_WORKERS=7
BASE=data/toplevel/networks_octo_pipe3fast10_qrb5165.json
mkdir -p gridx
cell () {
  local p=$1
  local w=$2
  local tag="GX_p${p}_w${w}"
  local json=data/toplevel/networks_octo_pareto_${tag}.json
  python3 - "$BASE" "$json" "$p" "$w" <<'PY'
import json,sys
b,o,p,w=sys.argv[1],sys.argv[2],int(sys.argv[3]),int(sys.argv[4])
d=json.load(open(b)); d["networks"]["octo"]["period"]=p; d["networks"]["octo"]["window_duration"]=w
d["_comment"]=f"schmoo grid (3600 s warm cpsat): period={p}, window={w}, 10 instances."
json.dump(d,open(o,"w"),indent=1)
PY
  timeout -s KILL 4500 python scripts/run_xpurt_schedule.py --networks-json "$json" \
      --solver cpsat --seed-solver heft_edf --cpsat-time-limit 3600 --profiled \
      > gridx/${tag}.log 2>&1
  local st
  st=$(grep -aoE 'cpsat status=[A-Z]+' gridx/${tag}.log | tail -1 | cut -d= -f2)
  echo "p=$p w=$w -> ${st:-NOSTATUS}"
}
export -f cell; export BASE
cat > /tmp/unk13.txt <<'CELLS'
100 305
100 320
100 350
100 400
100 450
110 305
110 320
110 350
110 400
120 290
120 305
120 320
120 350
CELLS
xargs -P 13 -n 2 bash -c 'cell "$0" "$1"' < /tmp/unk13.txt
echo GRIDX_DONE
