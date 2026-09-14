#!/usr/bin/env bash
# PHASE 2: control-rate parity sweep. Originals in runs/ and traces_torque2/ are NEVER
# touched -- everything here lands in runs_phase2/ and traces_phase2/.
#
# Argument: "KEY:TASK:ARM:MODE"
#   MODE=native  -- the embodiment's OWN control rate (widowx 5 Hz / 200 ms,
#                   google 3 Hz / 333 ms). The faithful model for both, and the only
#                   configuration validated for google (GOOGLE_FINE_MECHANISMS.md).
#   MODE=fine    -- widowx 25 Hz / 40 ms; google refused (see that doc).
#   MODE=trace   -- as native, but through trace_eval.py for the energy metrics.
source /home/ubuntu/simpler/sim_eval/roselite/env.sh
cd /home/ubuntu/simpler/sim_eval/roselite/finegrain
IFS=: read -r KEY TASK ARM MODE <<< "$1"
case "$TASK" in
  egg)    T=widowx_put_eggplant_in_basket ;;
  spoon)  T=widowx_spoon_on_towel ;;
  coke)   T=google_robot_pick_coke_can ;;
  drawer) T=google_robot_close_drawer ;;
  *) echo "bad task $TASK"; exit 2 ;;
esac
case "$ARM" in
  lat0)       LAT=0;     PER=auto   ;;
  pipe110fix) LAT=385.1; PER=111.4  ;;
  p105w300)   LAT=258.7; PER=124.8  ;;
  p130w275)   LAT=272.3; PER=130.1  ;;
  p150w300)   LAT=281.6; PER=150.4  ;;
  pipe200fix) LAT=260.5; PER=219.2  ;;
  serial283)  LAT=283.4; PER=283.4  ;;
  fp32_555)   LAT=555.0; PER=555.0  ;;
  cpu685)     LAT=684.8; PER=684.8  ;;
  *) echo "bad arm $ARM"; exit 2 ;;
esac
EXTRA=""; [ "$PER" != auto ] && EXTRA="--issue-period-ms $PER"
case "$MODE" in
  native) ACT="--actuation native"; OUTDIR=runs_phase2;    BIN=finegrain_eval.py ;;
  fine)   ACT="--actuation fine";   OUTDIR=runs_phase2;    BIN=finegrain_eval.py ;;
  trace)  ACT="--actuation native"; OUTDIR=traces_phase2;  BIN=trace_eval.py ;;
  *) echo "bad mode $MODE"; exit 2 ;;
esac
OUT=/home/ubuntu/simpler/sim_eval/roselite/finegrain/${OUTDIR}/${KEY}
mkdir -p "$(dirname "$OUT")"
python "$BIN" --task "$T" --latency-ms "$LAT" $EXTRA $ACT \
       --init-rng "${RNG:-100}" --n "${NEP:-24}" --out "$OUT" \
       > /home/ubuntu/simpler/logs/p2_${KEY}.log 2>&1
echo "done ${KEY} rc=$? : $(grep -h 'SUCCESS RATE' /home/ubuntu/simpler/logs/p2_${KEY}.log | tail -1)"
