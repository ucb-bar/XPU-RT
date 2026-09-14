#!/usr/bin/env bash
# usage: 04_run_arm_local.sh <task> <arm|all> <seed> [outdir]
#          task = spoon | egg | drawer | coke
#          arm  = lat0 | pipe110 | pipe200 | serial283 | fp32_555 | cpu685 | all
#
# Run ONE fine-grain latency arm (or the whole 6-arm ladder) on THIS box, using
# exactly the latency/cadence table the AWS sweep used (g5fine/job.sh).  This is
# the "one task, locally" path -- the AWS scripts (05-08) fan the same job out.
#
# Latency and cadence are two DIFFERENT numbers and are never collapsed:
#   --latency-ms      how old the observation behind an applied result is
#   --issue-period-ms how often a fresh result arrives
# Serial arms have cadence == latency; the pipelined arms do not.  Both are
# MEASURED on the QRB5165 -- see XPU-RT/qnn_models/octo/REPRODUCE_SCHEDULES.md.
#
# THIS BOX IS SHARED.  The script prints nvidia-smi / free before starting and
# refuses to launch 'all' with more than $CONC concurrent runs (default 3; 3-4
# fit on the 24 GB TITAN RTX alongside other users).  Check the numbers it prints
# before you trust them -- someone else's job can appear between the check and
# the launch.
#
# Time: ~7 min per 24-episode widowx run uncontended; google_robot is longer
# (113 native steps vs 60).  'all' at CONC=3 is ~20-30 min.
# Produces: <outdir>/<task>_<arm>_rng<seed>/summary.json  and  logs/<...>.log
# It worked if every run prints a SUCCESS RATE line and summary.json exists.

set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/common.sh"
[ $# -lt 3 ] && { sed -n '2,26p' "$0"; exit 2; }

TASK_SHORT="$1"; ARM="$2"; SEED="$3"; OUTDIR="${4:-$FINE/runs}"
CONC="${CONC:-3}"
T=$(task_id "$TASK_SHORT")
[ "$T" = BADTASK ] && { echo "bad task '$TASK_SHORT'"; exit 2; }
[ "$ARM" = all ] || [ "$(arm_lat "$ARM")" != BADARM ] || { echo "bad arm '$ARM'"; exit 2; }

echo "=== shared box check (do not assume the GPU is yours) ==="
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader || true
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader || true
free -g | sed -n '1,2p'
echo "  running $CONC concurrent at most"

mkdir -p "$OUTDIR" "$FINE/logs"
octo_env
cd "$FINE"

one() {  # $1 = arm
  local a="$1" lat per extra out log
  lat=$(arm_lat "$a"); per=$(arm_per "$a")
  extra=""; [ "$per" != auto ] && extra="--issue-period-ms $per"
  out="$OUTDIR/${TASK_SHORT}_${a}_rng${SEED}"
  log="$FINE/logs/${TASK_SHORT}_${a}_rng${SEED}.log"
  # No --horizon-ms / --action-dt-ms / --tick-hz: all THREE are derived from the
  # env.  Passing eggplant's 24000 ms horizon to spoon (a 60-step task) hands it
  # double the wall-clock the stock baseline gets and inflates its success.
  python finegrain_eval.py --task "$T" --ckpt "$CKPT" --latency-ms "$lat" $extra \
         --init-rng "$SEED" --n 24 --out "$out" > "$log" 2>&1
  echo "done ${TASK_SHORT} ${a} rng${SEED} rc=$? : $(grep -h 'SUCCESS RATE' "$log" | tail -1)"
}

if [ "$ARM" = all ]; then
  n=0
  for a in $ARMS_ALL; do
    one "$a" &
    n=$((n+1)); [ "$n" -ge "$CONC" ] && { wait -n 2>/dev/null || wait; n=$((n-1)); }
  done
  wait
else
  one "$ARM"
fi
echo "ARM RUN COMPLETE -> $OUTDIR"
