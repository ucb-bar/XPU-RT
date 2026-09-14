#!/usr/bin/env bash
# usage: 10_plot.sh [all|four|tasks|comparison|completion]
#
# Regenerate the sweep figures.  Every one of these recomputes its numbers from
# g5fine/runs/ at plot time -- nothing is hardcoded -- so a figure is only as
# current as the last 08_aws_fetch.sh.
#
#   four        finegrain_four_tasks.png        4 envs, 2 embodiments: marginal
#                                               success (design-corrected) and the
#                                               PAIRED penalty.  The headline.
#   tasks       finegrain_tasks.png             the two widowx envs, same encoding
#   comparison  finegrain_tasks_comparison.png  funnel / sensor->actuation age /
#                                               duty cycle -- WHERE latency breaks it
#   completion  completion_time.png             completion time over SUCCESSFUL
#                                               episodes only (a selection effect;
#                                               the script states it on the figure)
#
# CAVEAT THE FIGURES CARRY AND YOU MUST REPEAT: the widowx arms actuate on a fine
# 40 ms grid, the google_robot arms on their native 333.3 ms grid, so a latency
# change only alters the google_robot command sequence when an arrival crosses a
# control boundary.  That damps their measured sensitivity; the grasp-vs-push
# reading is better supported than a raw embodiment comparison.
#
# Time: ~20 s each.  Produces the PNGs listed above, in finegrain/.
# It worked if each script prints "[ok] <path>".

set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/common.sh"
WHICH="${1:-all}"
[ "$WHICH" = -h ] || [ "$WHICH" = --help ] && { sed -n '2,28p' "$0"; exit 0; }
octo_env
cd "$FINE"
run() { echo "=== $1 ==="; python "$1"; }
case "$WHICH" in
  all)        run plot_four_tasks.py; run plot_tasks_finegrain.py
              run plot_tasks_comparison.py; run plot_completion_time.py ;;
  four)       run plot_four_tasks.py ;;
  tasks)      run plot_tasks_finegrain.py ;;
  comparison) run plot_tasks_comparison.py ;;
  completion) run plot_completion_time.py ;;
  *) sed -n '2,28p' "$0"; exit 2 ;;
esac
