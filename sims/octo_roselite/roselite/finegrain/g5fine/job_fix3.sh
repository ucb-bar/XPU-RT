#!/usr/bin/env bash
# The two schedules found by the refinement pass. Argument: "TASK:ARM:SEED".
# Both are BOARD-MEASURED, ungated, over multiple processes.
#
#   p105w300   latency 258.7 ms, cadence 124.8 ms
#              greedy against the CORRECTED profile, period 105 / window 300.
#              Best on wall (1444.5), cadence and median latency. Its plan violates
#              REL (finish within window of scheduled release) on 7/10 instances, which
#              matters only for a tick-driven deployment -- the board runtime is
#              ungated, so REL is a property of the plan, not of these measurements.
#   p130w275   latency 272.3 ms, cadence 130.1 ms
#              cpsat against the corrected profile. The only board-validated schedule
#              satisfying REL 10/10; lowest stall rate (3.8%) and tightest cadence band
#              (130-131 ms). The pick if jitter matters more than rate.
source /home/ubuntu/simpler/sim_eval/roselite/env.sh
cd /home/ubuntu/simpler/sim_eval/roselite/finegrain
IFS=: read -r TASK ARM S <<< "$1"
case "$TASK" in
  spoon)  T=widowx_spoon_on_towel ;;
  egg)    T=widowx_put_eggplant_in_basket ;;
  drawer) T=google_robot_close_drawer ;;
  coke)   T=google_robot_pick_coke_can ;;
  *) echo "bad task $TASK"; exit 2 ;;
esac
case "$ARM" in
  p105w300) LAT=258.7; PER=124.8 ;;
  p130w275) LAT=272.3; PER=130.1 ;;
  *) echo "bad arm $ARM"; exit 2 ;;
esac
OUT=/home/ubuntu/simpler/sim_eval/roselite/finegrain/runs/${TASK}_${ARM}_rng${S}
LOG=/home/ubuntu/simpler/sim_eval/roselite/finegrain/logs/${TASK}_${ARM}_rng${S}.log
python finegrain_eval.py --task "$T" --latency-ms "$LAT" --issue-period-ms "$PER" \
       --init-rng "$S" --n 24 --out "$OUT" > "$LOG" 2>&1
echo "done ${TASK} ${ARM} seed ${S} rc=$? : $(grep -h 'SUCCESS RATE' "$LOG" | tail -1)"
