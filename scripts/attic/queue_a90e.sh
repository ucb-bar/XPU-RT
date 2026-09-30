#!/usr/bin/env bash
# The execution-mean calibration variant on the 90 Hz chain's 200 ms table (a90e), sharing round 0 with a90h,
# started once the 120 Hz variant (study3) has finished so the host is not over-subscribed with solves.
set -u; cd "$(dirname "$0")/.."
X=results/codesign_feedback/xpurt_long; H=results/codesign_feedback/hil_feedback; L=results/codesign_feedback/solver_v2
until grep -q HIL_FEEDBACK_STUDY_DONE $H/study3.log 2>/dev/null; do sleep 300; done
until grep -q BOARD_STAGE2_DONE $L/board_stage2_fba90hr0.log 2>/dev/null; do sleep 60; done
for s in cpsat_hard cpsat_soft greedy; do for k in 1 2 3; do for kind in trace cpu manifest; do ext=csv; [ $kind = manifest ] && ext=json
  [ -f $X/${kind}_fba90hr0${s}r${k}_other_run1.$ext ] && ln -sf ${kind}_fba90hr0${s}r${k}_other_run1.$ext $X/${kind}_fba90er0${s}r${k}_other_run1.$ext; done; done; done
cp $L/stage2_fba90hr0.log $L/stage2_fba90er0.log; cp $L/board_stage2_fba90hr0.log $L/board_stage2_fba90er0.log
CALARGS="--execution-only --stat mean" ROUNDS="0 1 2" PAIRS="a90e:wh_chain90_solve_h200" LIMIT=9000 bash scripts/hil_feedback_study_cal.sh
