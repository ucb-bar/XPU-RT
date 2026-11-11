#!/usr/bin/env bash
# usage: 03_validation_gate.sh <task> <seed> [outdir]
#          task = spoon | egg | drawer | coke
#
# THE GATE THAT MUST PASS BEFORE ANY SWEEP.  Runs the same 24 episode configs
# twice at ZERO modelled latency:
#   A) the stock harness   sim_eval/run_eval.py            (the reference)
#   B) the fine harness    finegrain/finegrain_eval.py --latency-ms 0
# and reports the difference in episodes.  If the fine harness cannot reproduce
# the stock rate at zero latency it is not modelling latency, it is modelling a
# bug -- exactly what happened on google_robot, where the fine harness scored
# 0/24 against stock's 10/24 (see ../GOOGLE_ROBOT_PORT.md, two separate bugs).
#
# PASS CRITERION (the one ../GOOGLE_ROBOT_PORT.md used): every arm within TWO
# episodes of stock.  Larger than that and something is wrong with the port.
#
# BOTH arms are pinned to $CKPT.  run_eval.py DEFAULTS to octo-small-1.5 and the
# fine harness to octo-small (1.0); leaving that to the defaults silently compares
# two different policies.
#
# Time: ~15 min for the pair on an idle TITAN RTX (24 episodes each).
# Produces: <outdir>/{stock,fine}_<task>_rng<seed>/summary.json + .log,
#           and a PASS/FAIL line on stdout.

set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/common.sh"
[ $# -lt 2 ] && { sed -n '2,26p' "$0"; exit 2; }

TASK_SHORT="$1"; SEED="$2"; OUTDIR="${3:-$FINE/repro_gate}"
T=$(task_id "$TASK_SHORT")
[ "$T" = BADTASK ] && { echo "bad task '$TASK_SHORT' (want spoon|egg|drawer|coke)"; exit 2; }
mkdir -p "$OUTDIR"
octo_env

S_OUT="$OUTDIR/stock_${TASK_SHORT}_rng${SEED}"
F_OUT="$OUTDIR/fine_${TASK_SHORT}_rng${SEED}"

echo "=== A) stock run_eval.py  $T  ckpt=$CKPT  rng=$SEED ==="
cd "$SIMROOT"
python run_eval.py --task "$T" --ckpt "$CKPT" --init-rng "$SEED" --n 24 \
       --save-video-every 0 --out "$S_OUT" > "$S_OUT.log" 2>&1
echo "  $(grep -h 'SUCCESS RATE' "$S_OUT.log" | tail -1)"

echo "=== B) finegrain_eval.py --latency-ms 0  $T  rng=$SEED ==="
# --actuation is left at 'auto': native for google_robot (its controllers are
# planner-interpolated and CANNOT be driven at a fine tick), fine for widowx.
cd "$FINE"
python finegrain_eval.py --task "$T" --ckpt "$CKPT" --latency-ms 0 \
       --init-rng "$SEED" --n 24 --out "$F_OUT" > "$F_OUT.log" 2>&1
echo "  $(grep -h 'SUCCESS RATE' "$F_OUT.log" | tail -1)"

python - "$S_OUT/summary.json" "$F_OUT/summary.json" <<'PY'
import json, sys
s = json.load(open(sys.argv[1])); f = json.load(open(sys.argv[2]))
ks, kf = s["n_success"], f["n_success"]; n = s["n_episodes"]
ss = {e["episode_id"] for e in s["episodes"] if e["success"]}
fs = {e["episode_id"] for e in f["episodes"] if e["success"]}
print(f"\n  stock {ks}/{n} = {100*ks/n:.1f}%   fine {kf}/{n} = {100*kf/n:.1f}%   "
      f"delta {kf-ks:+d} episodes")
print(f"  success sets differ on: {sorted(ss ^ fs)}")
print("  GATE PASSED" if abs(kf-ks) <= 2 else
      "  GATE FAILED (>2 episodes) -- do NOT sweep, see ../GOOGLE_ROBOT_PORT.md")
PY
