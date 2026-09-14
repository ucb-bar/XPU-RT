#!/usr/bin/env bash
# The CP-SAT-found, BOARD-VALIDATED operating point. Argument: "TASK:ARM:SEED".
#
#   p150w300   latency 281.6 ms, cadence 150.4 ms
#
# Provenance: schedule from --solver cpsat (status OPTIMAL) at period 150 /
# window_duration 300, then RUN ON THE BOARD (repro_runs/p150w300_20260906-164100.log,
# 710/710 dispatched, wall 1517.7 vs predicted 1585.4, so measured/predicted 0.958 --
# it beats its own prediction and therefore cannot be a busy-wait gate artifact).
# Latency is the MEASURED median per-instance span, matching how pipe110fix (385.1)
# and pipe200fix (260.5) were parameterised; cadence is the measured release period.
#
# This point has BETTER latency than pipe110fix and BETTER cadence than pipe200fix,
# so it tests the cadence-dominates-latency finding those two arms produced.
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
  p150w300) LAT=281.6; PER=150.4 ;;
  *) echo "bad arm $ARM"; exit 2 ;;
esac
OUT=/home/ubuntu/simpler/sim_eval/roselite/finegrain/runs/${TASK}_${ARM}_rng${S}
LOG=/home/ubuntu/simpler/sim_eval/roselite/finegrain/logs/${TASK}_${ARM}_rng${S}.log
python finegrain_eval.py --task "$T" --latency-ms "$LAT" --issue-period-ms "$PER" \
       --init-rng "$S" --n 24 --out "$OUT" > "$LOG" 2>&1
echo "done ${TASK} ${ARM} seed ${S} rc=$? : $(grep -h 'SUCCESS RATE' "$LOG" | tail -1)"
