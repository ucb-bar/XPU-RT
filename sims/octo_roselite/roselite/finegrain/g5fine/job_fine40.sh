#!/usr/bin/env bash
# PHASE 2: rerun the original success-rate / energy sweeps with the google envs at the
# ~40 ms (27 Hz) control grid, so both embodiments are refined alike.
#
# Writes to runs_fine40/ and traces_fine40/ -- the originals in runs/ and traces_torque2/
# are NEVER touched, so every published number remains reproducible from its own tree.
#
# Argument: "KEY:TASK:ARM:MODE" where MODE selects the google actuation config.
#   native   the validated 3 Hz configuration (what the original sweep used)
#   fine27   27 Hz actuation; the exact controller/sim-freq combination is filled in
#            from GOOGLE_FINE_CFG below once Phase 1 settles which one reproduces stock
# widowx is always 'fine' at 25 Hz and is unaffected by MODE.
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
case "$ARM" in                      # MEASURED latency / cadence, ungated
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
# Filled in by Phase 1. Empty MEANS NOT YET VALIDATED -- the script refuses rather than
# guessing, because an unvalidated actuation config silently produces a 0/24 floor.
GOOGLE_FINE_CFG=""
CFG=""
case "$T" in
  google_robot_*)
    if [ "$MODE" = "fine27" ]; then
      [ -z "$GOOGLE_FINE_CFG" ] && {
        echo "REFUSING $KEY: no validated google fine27 config yet (see GOOGLE_FINE_MECHANISMS.md)"
        exit 3; }
      CFG="$GOOGLE_FINE_CFG"
    fi ;;
esac
EXTRA=""; [ "$PER" != auto ] && EXTRA="--issue-period-ms $PER"
OUT=/home/ubuntu/simpler/sim_eval/roselite/finegrain/runs_fine40/${KEY}
mkdir -p "$(dirname "$OUT")"
python finegrain_eval.py --task "$T" --latency-ms "$LAT" $EXTRA $CFG \
       --init-rng "${RNG:-100}" --n 24 --out "$OUT" \
       > /home/ubuntu/simpler/logs/f40_${KEY}.log 2>&1
echo "done ${KEY} rc=$? : $(grep -h 'SUCCESS RATE' /home/ubuntu/simpler/logs/f40_${KEY}.log | tail -1)"
