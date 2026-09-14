#!/usr/bin/env bash
# One fine-grain run. Argument: "TASK:ARM:SEED".
#
# Unlike the coarse g5wide/job.sh this does NOT pass a horizon or an action dt --
# finegrain_eval.py now derives both from the env (native control_freq x the
# registered max_episode_steps), which is what keeps spoon at 12000 ms instead of
# silently inheriting eggplant's 24000 ms, and puts the google_robot tasks on
# their 3 Hz / 513 Hz timebase at a 27 Hz tick.
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

# name  latency_ms  cadence_ms   provenance (all MEASURED on the QRB5165)
case "$ARM" in
  lat0)      LAT=0;     PER=auto  ;;   # ideal reference, no compute latency
  pipe110)   LAT=117.7; PER=117.6 ;;   # pipelined 110 ms cadence, 8.50 inf/s
  pipe200)   LAT=231.8; PER=203.0 ;;   # pipelined 200 ms cadence
  serial283) LAT=283.4; PER=283.4 ;;   # 3-way serial chain, one in flight
  fp32_555)  LAT=555.0; PER=555.0 ;;   # fp32 numerically-valid path (CPU only)
  cpu685)    LAT=684.8; PER=684.8 ;;   # CPU-only int8 monolith
  *) echo "bad arm $ARM"; exit 2 ;;
esac
EXTRA=""; [ "$PER" != auto ] && EXTRA="--issue-period-ms $PER"

OUT=/home/ubuntu/simpler/sim_eval/roselite/finegrain/runs/${TASK}_${ARM}_rng${S}
LOG=/home/ubuntu/simpler/sim_eval/roselite/finegrain/logs/${TASK}_${ARM}_rng${S}.log
python finegrain_eval.py --task "$T" --latency-ms "$LAT" $EXTRA \
       --init-rng "$S" --n 24 --out "$OUT" > "$LOG" 2>&1
rc=$?
echo "done ${TASK} ${ARM} seed ${S} rc=${rc} : $(grep -h 'SUCCESS RATE' "$LOG" | tail -1)"
