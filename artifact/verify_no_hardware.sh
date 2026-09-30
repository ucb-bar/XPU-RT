#!/usr/bin/env bash
# The whole no-hardware verification suite of docs/Artifact/artifact_checklist.md §3, in one command.
#
# Nothing here needs the K1 board or a GPU, and nothing here writes into the results tree: every
# check re-derives a recorded number from the raw file it was derived from and compares. The board
# line of §3 (scripts/board_source_snapshot.sh --verify) is deliberately NOT run here -- it needs
# the board -- and is named in artifact/03_measure/README.md instead.
#
# This is a thin driver. Every check below is an existing script, run exactly as the docs run it.
#
#   artifact/verify_no_hardware.sh                 # the whole suite
#   FIGURES="warehouse_showdown_paper_r36" artifact/verify_no_hardware.sh
#   SKIP_FIGURES=1 artifact/verify_no_hardware.sh  # the repo-wide checks only
#
# Exit 0 if every check passed, 1 otherwise. The per-check output is kept in $LOGDIR.
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"
PY="${HOST_PY:-$REPO/.venv/bin/python}"
LOGDIR="${LOGDIR:-$(mktemp -d "${TMPDIR:-/tmp}/artifact_verify.XXXXXX")}"; mkdir -p "$LOGDIR"

# The six figures whose sidecars re-derive today (docs/Artifact/artifact_checklist.md §5). `--all` covers
# every sidecar in refined/ and reports the backlog as well; it is the wider check, not this one.
FIGURES="${FIGURES:-warehouse_showdown_cam30_solver_placed warehouse_showdown_cam30_solver_placed_rate_gain warehouse_showdown_paper_r30 warehouse_showdown_paper_submitted warehouse_showdown_paper_r36 showdown_45hz_pinned_vs_rosdefault_s1003 warehouse_showdown_paper_ladder warehouse_showdown_paper_allcores warehouse_showdown_paper_allcores_merged showdown_45hz_pinned_vs_rosdefault_s1000 warehouse_showdown_cam30_static6 warehouse_showdown_cam45_solver_placed warehouse_showdown_cam45_static6 warehouse_showdown_cam30_threeway warehouse_showdown_cam45_unpinned_best_v2 warehouse_showdown_cam36_allcores_s1003 warehouse_showdown_cam36_allcores_s1007 warehouse_showdown_cam36_allcores_s1009 showdown_36hz_solver_vs_rosallhart_s1006 showdown_45hz_pinned_vs_rospinned_s1011 showdown_45hz_solver_vs_rospinned_s1007 warehouse_showdown_cam30_solver_placed_v2 warehouse_showdown_cam30_solver_placed_rate_gain_v2 showdown_36hz_solver_vs_rosallhart_s1006_ladder warehouse_showdown_cam36_allcores_s1003_ladder warehouse_showdown_cam36_allcores_s1007_ladder warehouse_showdown_cam36_allcores_s1009_ladder warehouse_showdown_cam45_unpinned_best_s1007}"

PASS=0; FAIL=0
run() {  # run <name> <command...>
  local name="$1"; shift
  local log="$LOGDIR/${name//[^A-Za-z0-9_]/_}.log"
  printf '  %-58s ' "$name"
  if "$@" > "$log" 2>&1; then echo "PASS"; PASS=$((PASS+1));
  else echo "FAIL  (see $log)"; FAIL=$((FAIL+1)); fi
}

echo "no-hardware verification -- docs/Artifact/artifact_checklist.md §3"
echo "repo   $REPO"
echo "python $PY"
echo "logs   $LOGDIR"
echo

run "pytest tests xpu-rt/tests" \
    "$PY" -m pytest tests xpu-rt/tests -q
run "measured_timing.py --verify (board constants)" \
    "$PY" scripts/measured_timing.py --verify
# the same guarantee for the analytical (Tier A) ROS 2 per-node-pinning baseline: --verify re-runs
# the model from its stated inputs and re-reads those inputs from the schedules it was built from,
# so a drift in either is caught here
run "ros_pinning_model.py --verify (analytical baseline)" \
    "$PY" scripts/ros_pinning_model.py --verify
run "flight_quarantine.py (faulted batches)" \
    "$PY" scripts/flight_quarantine.py
run "figure_constants.check_registry (display literals)" \
    "$PY" -c "import sys;sys.path.insert(0,'scripts');import figure_constants as F;e=F.check_registry();print(e or 'clean');sys.exit(1 if e else 0)"
run "executed_tables.py (schedule-table ledger)" \
    "$PY" scripts/executed_tables.py
# the display dumps are too large to track, so the check that matters is that the archives hold the
# same bytes the renders read -- without it a clean clone plus the archives cannot redraw a figure
# ARCHIVES lets a clone point these two at the archive directory a reviewer has beside it: archive_v3/
# is gitignored, so a clone has none of its own and both would otherwise fail on every tar, which says
# nothing about the artifact.
ARCHIVES="${ARCHIVES:-$REPO/results/codesign_feedback/archive_v3}"
run "verify_archived_dumps.py (untracked inputs are archived)" \
    "$PY" scripts/verify_archived_dumps.py --archives "$ARCHIVES"
# the same guarantee one level down: --verify re-derives the board constants from traces that are
# ignored by pattern, so a clean clone that lacks them reports "no runs" rather than failing
run "verify_schedule_evolution.py (paper Fig 4, every panel re-composed)" \
    "$PY" scripts/verify_schedule_evolution.py

run "verify_loop_overview.py (paper Fig 1, both bands re-generated)" \
    "$PY" scripts/verify_loop_overview.py

# The figures that are not showdown composites each have their own producer, so each is checked by
# re-running it and comparing the sidecar it emits -- the same shape as the two above. Without these
# thirteen sidecars recorded numbers that nothing recomputed.
run "verify_hil_feedback.py (the 6 feedback-study figures, every round)" \
    "$PY" scripts/verify_hil_feedback.py

run "verify_control_rate_response.py (3 renders, every drawn point)" \
    "$PY" scripts/verify_control_rate_response.py

run "verify_ros_effort_ladder.py (every rung's harts, rate, latency and flights)" \
    "$PY" scripts/verify_ros_effort_ladder.py
# the paper's Fig 3 had no sidecar and no producer in this repository; cores_yolo_service.py derives
# its points from the schedules and this checks every one of them back against those schedules
run "verify_cores_yolo.py (paper Fig 3, every point against its schedule)" \
    "$PY" scripts/verify_cores_yolo.py
run "verify_ros_model_fidelity.py (ROS 2 baseline at three tiers, 7 stems)" \
    "$PY" scripts/verify_ros_model_fidelity.py
run "verify_showdown_improved.py (the showdown_improved_* forms, every census re-paired)" \
    "$PY" scripts/verify_showdown_improved.py
# this one recomputes its own output from the campaign CSVs and exits non-zero when the contrast the
# envelope figures annotate is not a positive, significant step
run "audit_showdown_claims.py (the envelope's rate contrasts)" \
    "$PY" scripts/audit_showdown_claims.py
# a reproduction page is only a recipe if its commands resolve; --strict also checks the pages that
# document components this checkout does not contain, which is how the skip list is audited
run "verify_doc_commands.py (every script a page tells a reader to run)" \
    "$PY" scripts/verify_doc_commands.py

run "verify_board_traces.py (traces a check opens are archived)" \
    "$PY" scripts/verify_board_traces.py --archives "$ARCHIVES"
# Isaac is the reviewer's to install; everything a flight LOADS is ours to supply, and a flight
# leaves no sidecar, so nothing else looks at the weights, the cadence traces or the task package.
run "verify_flight_inputs.py (a reviewer with Isaac has the rest)" \
    "$PY" scripts/verify_flight_inputs.py
# the same guarantee for the half of the artifact that has no figure sidecars: the ModelBlaster pin is
# unpushed, so what makes it reachable is a bundle whose tip must not drift from the pin, and the IME
# numbers the docs state must still follow from the tracked tables and schedules behind them. With the
# submodule not checked out its content checks report SKIP, not PASS, and say so on the summary line
run "verify_modelblaster_inputs.py (the ModelBlaster side is complete)" \
    "$PY" scripts/verify_modelblaster_inputs.py
# docs/Artifact/run_index.md is generated from the registries; --check fails when it and they disagree, so the
# one page that says what differs between two runs cannot quietly stop being true
run "run_index.py --check (the run index matches the registries)" \
    "$PY" scripts/run_index.py --check
# the per-result bundles are generated from the sidecars too; --check fails when a figure has moved
# on and its directory still describes a superseded render
run "build_implementation_bundles.py --check (a page per figure, ROS deployment and XPU-RT arm)" \
    "$PY" scripts/build_implementation_bundles.py --check

if [ -z "${SKIP_FIGURES:-}" ]; then
  echo
  for f in $FIGURES; do
    m="results/codesign_feedback/refined/${f}_metrics.json"
    if [ ! -f "$m" ]; then printf '  %-58s %s\n' "verify_showdown_figure $f" "SKIP (no sidecar)"; continue; fi
    run "verify_showdown_figure --metrics $f" "$PY" scripts/verify_showdown_figure.py --metrics "$m"
  done
fi

echo
echo "$PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ]
