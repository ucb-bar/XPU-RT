#!/usr/bin/env bash
# Corrected PIPELINED arms. Argument: "TASK:ARM:SEED".
#
# The original pipe110/pipe200 arms used the board's THROUGHPUT (wall / n_instances)
# as the latency parameter. For a pipelined schedule that is wrong: a fresh result
# arrives every ~111 ms, but each result is ~385 ms old because 4 inferences are in
# flight (385 ~= 4 x 111). Latency here is the MEASURED per-inference span from the
# board trace; cadence is the measured throughput.
#
#   arm         latency ms   cadence ms   provenance (MEASURED on the QRB5165)
#   pipe110fix     385.1        111.4     pipe110x10 trace: median instance span
#                                         385.14, wall 1113.86 / 10 instances
#   pipe200fix     260.5        219.2     pipe200_gated trace: median span 260.49,
#                                         wall 1096.04 / 5 instances
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
  pipe110fix) LAT=385.1; PER=111.4 ;;
  pipe200fix) LAT=260.5; PER=219.2 ;;
  *) echo "bad arm $ARM"; exit 2 ;;
esac
OUT=/home/ubuntu/simpler/sim_eval/roselite/finegrain/runs/${TASK}_${ARM}_rng${S}
LOG=/home/ubuntu/simpler/sim_eval/roselite/finegrain/logs/${TASK}_${ARM}_rng${S}.log
python finegrain_eval.py --task "$T" --latency-ms "$LAT" --issue-period-ms "$PER" \
       --init-rng "$S" --n 24 --out "$OUT" > "$LOG" 2>&1
echo "done ${TASK} ${ARM} seed ${S} rc=$? : $(grep -h 'SUCCESS RATE' "$LOG" | tail -1)"
