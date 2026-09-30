#!/usr/bin/env bash
# Is the 45 Hz free placement something the solver hands you, or something that happened once?
#
# The board number the figure draws -- 28.30 ms camera->control, 27.45 ms in the Gantt window -- is
# measured on the K1, but WHERE the kernels run is the solver's answer: wh_chain45_free.json carries
# no allowed_machines, no machine_width, no preferred_hw. That table was produced by an 8-worker
# CP-SAT solve, and xpu-rt/cpsat_scheduler.py is explicit that CP-SAT is reproducible only with
# workers=1: with several search workers the result depends on thread interleaving, measured spread
# about +/-1.5 ms on a 46 ms schedule.
#
# So two single-worker solves, each on its own copy of the spec so nothing already on disk moves:
#
#   det1  seed 42, the same seed the 8-worker solve used -- does one worker reach the same placement?
#   det2  seed 7                                         -- is that placement a property of the
#                                                           problem, or of one random draw?
#
# A recipe is reliable if it gives the same answer to both. Each solve is one core; two is the whole
# solve budget, so nothing else heavy runs alongside.
#
#   scripts/solve45_determinism.sh              env LIMIT=3600
set -u
cd "$(dirname "$0")/.."
LIMIT="${LIMIT:-3600}"
R=results/codesign_feedback/solver_det45
mkdir -p "$R"
CAL=results/codesign_feedback/k1_board_calibration_yolo110.json
export XPURT_CPSAT_PYTHON=$PWD/.venv/bin/python XPURT_UNIFORM_PACKED_WIDTH=1 XPURT_NO_COMPACT=1
export XPURT_CPSAT_WORKERS=1

for stem in det1 det2; do
  spec="data/toplevel/wh_chain45_free_${stem}.json"
  echo "=== $(date +%H:%M:%S) solve $stem (workers=1, limit ${LIMIT}s)"
  .venv/bin/python scripts/run_xpurt_schedule.py --networks-json "$spec" \
      --solver cpsat --max-periodic-iters 1 --cpsat-time-limit "$LIMIT" \
      --use-profiled --board-calibration "$CAL" > "$R/${stem}.log" 2>&1 &
done
wait
echo "=== $(date +%H:%M:%S) both solves returned"
for stem in det1 det2; do
  s="schedules/scheduled_wh_chain45_free_${stem}_cpsat_profiled.json"
  [ -f "$s" ] && echo "  $stem -> $s" || echo "  $stem produced NO table (see $R/${stem}.log)"
done
echo "SOLVE45_DET_DONE $(date +%H:%M:%S)"
