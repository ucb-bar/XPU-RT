#!/usr/bin/env bash
# usage: 11_traces.sh ee            # 1-episode traces for the end-effector figures
#        11_traces.sh energy        # 24-episode traces for the motion-cost figure
#        11_traces.sh plot          # draw both figure sets from existing traces
#
# trace_eval.py is finegrain_eval.py plus per-tick state dumps: ep*_ee_xyz.npy,
# ep*_ee_quat.npy, ep*_qvel.npy, ep*_link_com.npy, ep*_bodies.json,
# ep*_action_age_ms.npy, ep*_bg.png.  Those files, not the summaries, are what
# plot_ee_traces.py and energy_analysis.py read.
#
# There was NO driver script for this step -- the traces were produced by hand.
# This is that step written down; the parameters below were read back out of the
# existing traces' own summary.json (seed 100, ckpt octo-small 1.0, n=1 for the
# ee traces, n=24 for the energy traces), so re-running reproduces the same
# CONFIGURATION.  It will NOT reproduce the same trajectories: the harness is not
# run-to-run deterministic (../NONDETERMINISM.md).
#
# ONE SOURCE PER FIGURE.  energy_analysis.py reads traces_energy2/ if
# traces_energy2/_COMPLETE exists, else traces_energy/.  Never half-populate one
# directory from two sessions: mixing arms across independent runs of a
# non-deterministic harness puts run-to-run noise into the arm differences.  This
# script writes into a fresh directory and only touches _COMPLETE at the end.
#
# Time: ee ~10 min (8 short runs); energy ~3-4 h (12 x 24 episodes at CONC=3).
# Produces: traces/{egg_<arm>}/, traces_energy2/{drw,egg}_<arm>/, and on `plot`
#   eetrace_{A_time,B_staleness,C_overlay,D_ladder}.png, energy_by_schedule.png
# It worked if every requested arm directory contains ep00_qvel.npy and a
# summary.json, and `plot` prints an [ok] line per figure.

set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/common.sh"
MODE="${1:-}"; SEED="${SEED:-100}"; CONC="${CONC:-3}"
[ -z "$MODE" ] || [ "$MODE" = -h ] || [ "$MODE" = --help ] && { sed -n '2,30p' "$0"; exit 0; }
octo_env
cd "$FINE"

# $1 out-root  $2 key  $3 task-short  $4 arm  $5 n-episodes  [$6 extra args]
trace_one() {
  local root="$1" key="$2" ts="$3" a="$4" n="$5"; shift 5
  local t lat per extra d
  t=$(task_id "$ts"); lat=$(arm_lat "$a"); per=$(arm_per "$a")
  extra=""; [ "$per" != auto ] && extra="--issue-period-ms $per"
  d="$root/${key}_${a}"
  if [ -f "$d/summary.json" ]; then echo "skip $d (already present)"; return 0; fi
  mkdir -p "$root"
  python trace_eval.py --task "$t" --ckpt "$CKPT" --latency-ms "$lat" $extra \
         --init-rng "$SEED" --n "$n" "$@" --out "$d" > "$root/${key}_${a}.log" 2>&1
  echo "done ${key} ${a} rc=$? : $(grep -h 'SUCCESS RATE' "$root/${key}_${a}.log" | tail -1)"
}

case "$MODE" in
  ee)
    # plot_ee_traces.py reads traces/egg_<arm>, episode 0 only, four arms.
    for a in lat0 pipe110 serial283 cpu685; do
      trace_one "$FINE/traces" egg egg "$a" 1 --ep-start 0
    done
    ;;
  energy)
    # energy_analysis.py reads {drw,egg}_<arm> over all six arms, 24 episodes,
    # and scores SUCCESSFUL episodes only.
    R="$FINE/traces_energy2"
    rm -f "$R/_COMPLETE"
    n=0
    for key_task in "drw drawer" "egg egg"; do
      set -- $key_task
      for a in $ARMS_ALL; do
        trace_one "$R" "$1" "$2" "$a" 24 &
        n=$((n+1)); [ "$n" -ge "$CONC" ] && { wait -n 2>/dev/null || wait; n=$((n-1)); }
      done
    done
    wait
    miss=0
    for key in drw egg; do for a in $ARMS_ALL; do
      [ -f "$R/${key}_${a}/summary.json" ] || { echo "MISSING $R/${key}_${a}"; miss=1; }
    done; done
    [ "$miss" -eq 0 ] && { touch "$R/_COMPLETE"; echo "all 12 arms present -> touched _COMPLETE"; } \
                      || echo "NOT marking _COMPLETE: energy_analysis.py will fall back to traces_energy/"
    ;;
  plot)
    echo "=== plot_ee_traces.py ==="; python plot_ee_traces.py
    echo "=== energy_analysis.py ==="; python energy_analysis.py
    ;;
  *) sed -n '2,30p' "$0"; exit 2 ;;
esac
