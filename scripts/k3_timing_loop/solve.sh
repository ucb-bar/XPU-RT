#!/usr/bin/env bash
# Solve one workload config with this XPU-RT checkout and a chosen profile tree (pinned `decomposed` solver).
# usage (from anywhere): solve.sh <config .json> <profile tree> <out dir>
# By default the schedule picture is not drawn (run_schedule_noplot.py; the plan file is the same).
# K3TL_PLOT=1 draws it. Outputs in <out dir>: schedules/scheduled_<config>_decomposed_profiled.json, plots/, logs/.
# Env: K3TL_MODELBLASTER (default <repo>/zephyr-chipyard-sw/modelblaster): its cores/codegen_contract.json is the
#      packed-weight width contract the K3 kernels need (XPURT_CODEGEN_CONTRACT, XPURT_UNIFORM_PACKED_WIDTH=1).
set -eu
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; xp="$(cd "$HERE/../.." && pwd)"
cfg="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"; root="$(cd "$2" && pwd)"; out="$3"
mb="$(cd "${K3TL_MODELBLASTER:-$xp/zephyr-chipyard-sw/modelblaster}" && pwd)"
[[ -d "$root/profile" ]] || { echo "no profile tree at $root" >&2; exit 2; }
[[ -f "$mb/cores/codegen_contract.json" ]] || { echo "no cores/codegen_contract.json in the ModelBlaster checkout $mb" >&2; exit 2; }
mkdir -p "$out/schedules" "$out/plots" "$out/logs"; cd "$out"
name="$(basename "$cfg" .json)"
export XPURT_GEN_ROOT="$root"
export XPURT_CODEGEN_CONTRACT="$mb/cores/codegen_contract.json"
export XPURT_UNIFORM_PACKED_WIDTH=1 PYTHONHASHSEED=0 PYTHONUNBUFFERED=1 MPLBACKEND=Agg
log="logs/solve_${name}_decomposed.log"
start=$(date +%s)
echo "xpurt=$(git -C "$xp" rev-parse --short HEAD) branch=$(git -C "$xp" branch --show-current) dirty=$(git -C "$xp" status --porcelain --untracked-files=no | wc -l | tr -d ' ') gen_root=$XPURT_GEN_ROOT config=$cfg" > "$log"
set +e
if [[ "${K3TL_PLOT:-0}" == 1 ]]; then
  python3 "$xp/scripts/run_xpurt_schedule.py" --networks-json "$cfg" --solver decomposed >> "$log" 2>&1
else                                       # skip the schedule picture (most of the time for big workloads); same plan file
  python3 "$HERE/run_schedule_noplot.py" "$xp" --networks-json "$cfg" --solver decomposed >> "$log" 2>&1
fi
rc=$?
set -e
echo "exit=$rc seconds=$(( $(date +%s) - start ))" >> "$log"
grep -a -E "Hand-off cost|Sender delay|misses=|Scheduled JSON saved|exit=" "$log" | tail -5 | cut -c1-240
exit $rc
