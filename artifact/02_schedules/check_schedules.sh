#!/usr/bin/env bash
# The two host-side gates a schedule has to pass before a board run means anything, run over the
# schedule tables the current figures draw:
#
#   scripts/check_schedule_feasibility.py   can the board's walker execute this table as written?
#                                           (no double-booking, no forward edge, every target on the
#                                           board, every implementation available where it was placed)
#   xpu-rt/codegen_contract.py              can ModelBlaster generate code for it? (a packed-weight
#                                           dispatch must take one core width across its instances)
#
# Both are host-side and cost no board time. This is a thin driver: it calls those two scripts and
# nothing else. Pass schedule paths to check others.
#
#   artifact/02_schedules/check_schedules.sh
#   artifact/02_schedules/check_schedules.sh schedules/fig_a_greedy_clamped.json
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$REPO"
PY="${HOST_PY:-$REPO/.venv/bin/python}"

# The XPU-RT row of each current figure's panel I, as its Gantt sidecar's `xpu_schedule` names it:
#   fig_w2pg36_greedy_clamped   -> refined/rate36/measured_gantt_r36_xpu_metrics.json  (warehouse_showdown_paper_r36)
#   fig_a_cpsat_hard_clamped    -> schedules/measured_gantt_v3_xpu_metrics.json        (the showdown_45hz_pinned_vs_rosdefault_s1003 family)
DEFAULT="schedules/fig_w2pg36_greedy_clamped.json schedules/fig_a_cpsat_hard_clamped.json"
SCHEDS="${*:-$DEFAULT}"

rc=0
for s in $SCHEDS; do
  echo "=== $s"
  "$PY" scripts/check_schedule_feasibility.py --schedule "$s" > /tmp/.sched_feas.$$ 2>&1 || rc=1
  tail -3 /tmp/.sched_feas.$$
  "$PY" xpu-rt/codegen_contract.py "$s" > /tmp/.sched_ctr.$$ 2>&1 || rc=1
  tail -2 /tmp/.sched_ctr.$$
  rm -f /tmp/.sched_feas.$$ /tmp/.sched_ctr.$$
  echo
done
exit $rc
