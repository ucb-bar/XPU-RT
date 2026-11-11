#!/usr/bin/env bash
# Closed-loop arm for one cell of the warm-started CP-SAT (release period x deadline)
# grid.  Argument: "TASK:ARM:SEED".  ARM is g<P>_<W> for grid cell p=P, w=W.
#
# LAT/PER below are PREDICTED from the schedule JSON (extract_gw.py): LAT is the
# median job span (observation age), PER the median inter-release gap (cadence).
# They are NOT board-measured.  Board error is known and not one-directional
# (-4.3% at p150/w300, +31% on a 4-deep pipeline), so these rank correctly but
# must never be reported as measured numbers.
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
  g130_275) LAT=271.4; PER=131.1 ;;
  g130_290) LAT=286.0; PER=130.2 ;;
  g130_305) LAT=297.3; PER=130.0 ;;
  g130_320) LAT=315.1; PER=130.0 ;;
  g130_350) LAT=339.4; PER=130.0 ;;
  g110_450) LAT=389.8; PER=110.0 ;;
  g120_400) LAT=381.1; PER=120.0 ;;
  g120_450) LAT=387.4; PER=120.0 ;;
  g130_400) LAT=356.3; PER=130.0 ;;
  g130_450) LAT=395.6; PER=130.0 ;;
  g140_290) LAT=283.7; PER=140.0 ;;
  g140_305) LAT=297.6; PER=140.0 ;;
  g140_320) LAT=312.4; PER=140.0 ;;
  g140_350) LAT=339.2; PER=140.0 ;;
  g140_450) LAT=391.6; PER=140.0 ;;
  g140_400) LAT=328.3; PER=140.9 ;;
  g150_290) LAT=281.3; PER=150.0 ;;
  g150_305) LAT=303.1; PER=150.0 ;;
  g150_320) LAT=312.4; PER=150.0 ;;
  g150_450) LAT=317.6; PER=150.0 ;;
  g150_350) LAT=319.0; PER=150.0 ;;
  g165_290) LAT=277.6; PER=165.0 ;;
  g165_305) LAT=297.1; PER=165.0 ;;
  g165_400) LAT=311.0; PER=165.0 ;;
  g165_320) LAT=312.9; PER=165.0 ;;
  g165_350) LAT=326.3; PER=165.0 ;;
  g165_450) LAT=327.2; PER=165.0 ;;
  g165_275) LAT=270.3; PER=165.2 ;;
  g180_275) LAT=269.6; PER=180.0 ;;
  g180_290) LAT=282.3; PER=180.0 ;;
  g180_305) LAT=292.6; PER=180.0 ;;
  g180_350) LAT=294.7; PER=180.0 ;;
  g180_320) LAT=303.1; PER=180.0 ;;
  g180_400) LAT=352.6; PER=180.0 ;;
  g180_450) LAT=357.7; PER=182.3 ;;
  g200_260) LAT=246.0; PER=200.0 ;;
  g200_400) LAT=257.9; PER=200.0 ;;
  g200_275) LAT=264.8; PER=200.0 ;;
  g200_290) LAT=269.0; PER=200.0 ;;
  g200_320) LAT=280.7; PER=200.0 ;;
  g200_450) LAT=369.2; PER=200.0 ;;
  g220_260) LAT=232.7; PER=220.0 ;;
  g250_260) LAT=232.7; PER=250.0 ;;
  g283_260) LAT=232.7; PER=283.0 ;;
  *) echo "bad arm $ARM"; exit 2 ;;
esac
OUT=/home/ubuntu/simpler/sim_eval/roselite/finegrain/runs/${TASK}_${ARM}_rng${S}
LOG=/home/ubuntu/simpler/sim_eval/roselite/finegrain/logs/${TASK}_${ARM}_rng${S}.log
python finegrain_eval.py --task "$T" --latency-ms "$LAT" --issue-period-ms "$PER" \
       --init-rng "$S" --n 24 --out "$OUT" > "$LOG" 2>&1
echo "done ${TASK} ${ARM} seed ${S} rc=$? : $(grep -h 'SUCCESS RATE' "$LOG" | tail -1)"
