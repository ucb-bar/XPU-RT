#!/usr/bin/env bash
# Before/after comparison of walker policies and scheduler feedback on the K3,
# with the board locked for the whole campaign (k3_board_lock.sh).
#
#   run_before_after_campaign.sh <out_dir>
#
# Workload: data/toplevel/networks_k3_mlp1600_dronet480_yolo64_singletons.json
# (mlp_control 1600 Hz, dronet 480 Hz, yolov8_nano 64 Hz, singletons, greedy).
# Every condition uses the same build; runs are interleaved so board drift hits
# all conditions alike. Run directories (one per board run) under <out_dir>:
#   before_static_walker_runN        the schedule as solved, static walker
#   ratio_feedback_r1_runN           duration_feedback.py (mean measured/predicted), 1 round
#   calibrated_feedback_r1_runN      calibration_feedback.py, round 1
#   calibrated_feedback_r2_runN      calibration_feedback.py, round 2
#   global_pool_yolo_runN            static walker, yolov8_nano in the global pool
#   dynamic_walker_runN              ModelBlaster --dynamic-walker, safety 1.0
# Then: score_k3_runs.py <out_dir>/*_run*   and   plot_predicted_vs_actual.py --runs <out_dir>
#
# Env: as run_schedule_on_k3.sh, plus PY_XPURT (python with XPU-RT's deps).
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
X="$(cd "${here}/../.." && pwd)"
R="$(realpath -m "${1:?usage: run_before_after_campaign.sh <out_dir>}")"; mkdir -p "${R}"
MB="$(realpath "${MODELBLASTER_ROOT:-${X}/../ModelBlaster}")"
PY_XPURT="${PY_XPURT:-python3}"
SPEC=data/toplevel/networks_k3_mlp1600_dronet480_yolo64_singletons.json
export XPURT_CODEGEN_CONTRACT="${MB}/cores/codegen_contract.json" XPURT_UNIFORM_PACKED_WIDTH=1

# shellcheck source=k3_board_lock.sh
source "${here}/k3_board_lock.sh"
k3_lock_acquire || exit 3
export K3_LOCK_HELD=1

run() {   # run <name> <schedule> [VAR=value ...]
    local name=$1 sched=$2; shift 2
    echo "[$(date +%T)] ${name}"
    "${here}/run_schedule_on_k3.sh" "${sched}" "${R}/${name}" "$@" >/dev/null 2>&1 \
        && [[ -s "${R}/${name}/$(basename "${sched}" .json)_trace.csv" ]] || echo "  FAILED ${name}"
}
loop() {  # loop <feedback> <round> [--trace T]
    local fb=$1 r=$2; shift 2
    ( cd "${X}" && "${PY_XPURT}" scripts/calibrated_feedback_loop.py --spec "${SPEC}" --execute trace \
          --round "${r}" --out "${R}/feedback_${fb}" --feedback "${fb}" --truth "${BASE}:${T0}" "$@" ) | tail -2
}

cd "${X}"
"${PY_XPURT}" scripts/run_xpurt_schedule.py --networks-json "${SPEC}" --solver greedy > "${R}/solve_base.log" 2>&1
BASE="${X}/schedules/scheduled_$(basename "${SPEC}" .json)_greedy_profiled.json"
cp "${BASE}" "${R}/"; BASE="${R}/$(basename "${BASE}")"

run before_static_walker_run1 "${BASE}" MB_XPURT_DYNAMIC=0
T0="$(ls "${R}"/before_static_walker_run1/*_trace.csv)"
for fb in calibrated ratio; do
    loop "${fb}" 0 --trace "${T0}"
    loop "${fb}" 1
done
C1="$(ls "${R}"/feedback_calibrated/round1/scheduled_*.json)"
R1="$(ls "${R}"/feedback_ratio/round1/scheduled_*.json)"
for i in 1 2 3; do
    run calibrated_feedback_r1_run$i "${C1}" MB_XPURT_DYNAMIC=0
    run global_pool_yolo_run$i "${BASE}" MB_XPURT_DYNAMIC=0 MB_XPURT_POOL=yolov8_nano:15.625
    [[ $i -le 2 ]] && run ratio_feedback_r1_run$i "${R1}" MB_XPURT_DYNAMIC=0
    [[ $i -le 2 ]] && run dynamic_walker_run$i "${BASE}" MB_XPURT_DYNAMIC=1 MB_XPURT_SAFETY_BOUNDARY=1.0
    [[ $i -ge 2 ]] && run before_static_walker_run$i "${BASE}" MB_XPURT_DYNAMIC=0
done
loop calibrated 1 --trace "$(ls "${R}"/calibrated_feedback_r1_run1/*_trace.csv)"
loop calibrated 2
C2="$(ls "${R}"/feedback_calibrated/round2/scheduled_*.json)"
for i in 1 2; do run calibrated_feedback_r2_run$i "${C2}" MB_XPURT_DYNAMIC=0; done
echo "[$(date +%T)] done: ${R}"
