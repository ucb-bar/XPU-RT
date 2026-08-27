#!/usr/bin/env bash
# End-to-end reproduction of the mlp_control + dronet + yolov8_nano XPU-RT
# schedule on spike (no FireSim). Automates every step documented in
# docs/mlp_dronet_yolo_spike_reproduction.md sections 0/1/2/3/5/6/7 in
# order. Read that doc for the "why" behind each step and the bugs each
# workaround here is standing in for; this script is the "just run it"
# path, not a replacement for understanding the flow.
#
# Usage:
#   bash scripts/repro_mlp_dronet_yolo_spike.sh [flags]
#
# Flags:
#   --skip-deps         skip installing xpu-rt/modelblaster's own deps
#                        (scripts/install_xpurt_deps.sh) into the active
#                        conda env. Use once you've already run that once
#                        for this env and just want to re-run the pipeline.
#   --skip-profile      skip step 1 (re-profiling all 3 models on spike).
#                        Use once gen/profile/{scalar,RVV}/spike/<model>
#                        already holds fresh data (e.g. re-running just the
#                        scheduler/build after an unrelated change).
#   --skip-dispatch     skip step 2 (dispatch-graph emit + basename rename).
#   --skip-schedule     skip step 4 (schedule generation). Requires the
#                        target schedules/scheduled_*.json to already exist.
#   --skip-build        skip step 5 (xpurt_demo build + spike run + verify).
#   --trace             enable XPURT_TRACE=1 on the build/run step and
#                        render the real-execution timeline afterward via
#                        plot_xpurt_trace.py (doc section 7).
#   --networks-json PATH   workload spec (default:
#                        data/toplevel/networks_mlp_dronet_yolo_spike.json)
#   --solver NAME       scheduler algorithm (default: greedy_periodic)
#   -h, --help          show this help and exit
#
# Example (full from-scratch run with a timeline plot):
#   bash scripts/repro_mlp_dronet_yolo_spike.sh --trace
#
# Example (schedule/logic changed, profile data + dispatch graphs still
# fresh -- just regenerate the schedule and rebuild):
#   bash scripts/repro_mlp_dronet_yolo_spike.sh --skip-profile --skip-dispatch

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOP_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Resolve submodule locations from .gitmodules rather than hardcoding
# "zephyr-chipyard-sw"/"modelblaster" as literal path components -- these
# are submodule *names*, which are the stable identifier; the checkout
# path is whatever .gitmodules currently says it is.
_zcs_path="$(git -C "${TOP_ROOT}" config -f .gitmodules --get submodule.zephyr-chipyard-sw.path)" || {
    echo "ERROR: no 'zephyr-chipyard-sw' submodule entry in ${TOP_ROOT}/.gitmodules" >&2
    exit 1
}
ZCS_ROOT="${TOP_ROOT}/${_zcs_path}"
if [[ ! -d "${ZCS_ROOT}" ]]; then
    echo "ERROR: zephyr-chipyard-sw submodule not checked out at ${ZCS_ROOT}" >&2
    echo "  run: git -C ${TOP_ROOT} submodule update --init ${_zcs_path}" >&2
    exit 1
fi

_mb_path="$(git -C "${ZCS_ROOT}" config -f .gitmodules --get submodule.modelblaster.path)" || {
    echo "ERROR: no 'modelblaster' submodule entry in ${ZCS_ROOT}/.gitmodules" >&2
    exit 1
}
MB_ROOT="${ZCS_ROOT}/${_mb_path}"
if [[ ! -d "${MB_ROOT}" ]]; then
    echo "ERROR: modelblaster submodule not checked out at ${MB_ROOT}" >&2
    echo "  run: git -C ${ZCS_ROOT} submodule update --init ${_mb_path}" >&2
    exit 1
fi

SKIP_DEPS=""
SKIP_PROFILE=""
SKIP_DISPATCH=""
SKIP_SCHEDULE=""
SKIP_BUILD=""
TRACE=""
NETWORKS_JSON="${TOP_ROOT}/data/toplevel/networks_mlp_dronet_yolo_spike.json"
SOLVER="greedy_periodic"

while (( $# )); do
    case "$1" in
        --skip-deps)      SKIP_DEPS=1 ;;
        --skip-profile)   SKIP_PROFILE=1 ;;
        --skip-dispatch)  SKIP_DISPATCH=1 ;;
        --skip-schedule)  SKIP_SCHEDULE=1 ;;
        --skip-build)     SKIP_BUILD=1 ;;
        --trace)          TRACE=1 ;;
        --networks-json)  shift; NETWORKS_JSON="$1" ;;
        --solver)         shift; SOLVER="$1" ;;
        -h|--help)
            sed -n '2,38p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *) echo "unknown flag: $1" >&2; exit 1 ;;
    esac
    shift
done

case "${NETWORKS_JSON}" in
    /*) : ;;
    *) NETWORKS_JSON="${TOP_ROOT}/${NETWORKS_JSON}" ;;
esac

log() { echo "[repro_mlp_dronet_yolo] $*"; }

# --- Step 0: environment (doc section 0) -----------------------------------
log "activating zephyr toolchain env"
# shellcheck disable=SC1091
source "${ZCS_ROOT}/tools/miniforge3/etc/profile.d/conda.sh"
conda activate zephyr
# shellcheck disable=SC1091
source "${ZCS_ROOT}/scripts/set_envvars_sdk.sh"
export PATH="/usr/bin:${PATH}"
export PATH="${CONDA_PREFIX}/bin:${PATH}"   # re-promote conda's python/pip (see doc Bug 13)
export PYTHONPATH="${ZCS_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

if [[ -z "${SKIP_DEPS}" ]]; then
    log "installing xpu-rt/modelblaster deps into this env"
    bash "${TOP_ROOT}/scripts/install_xpurt_deps.sh"
else
    log "skipping dependency install (--skip-deps)"
fi

# Model, quant pairs this reproduction covers. Order matters: it's the
# order used for MODELS=/QUANTS= in step 5 below (must match).
MODEL_QUANTS=(mlp_control:fp32 dronet:int8 yolov8_nano:int8)

# --- Step 1: profile each model on spike (doc section 1) -------------------
if [[ -z "${SKIP_PROFILE}" ]]; then
    log "step 1/5: profiling each model on spike (scalar + rvv)"
    cd "${ZCS_ROOT}"
    for mq in "${MODEL_QUANTS[@]}"; do
        model="${mq%%:*}"; quant="${mq##*:}"
        curated_env=()
        if [[ "${quant}" == "int8" ]]; then
            # Without this, generate_kernels.py's curated-kernel swap never
            # fires and every int8 op silently falls back to the slow,
            # non-bit-exact reference implementation (see run.sh comments).
            curated_env=(GLOBAL_CURATED_DIR="${MB_ROOT}/kernels")
        fi
        for target in scalar rvv; do
            log "  profiling ${model} (${quant}, TARGET=${target})"
            env PROFILE_OUT_ROOT=gen/profile PROFILE_CPU=spike PROFILE_CORES=0 \
                PROFILE_CLOCK_MHZ=1000.0 QUANT="${quant}" TARGET="${target}" \
                BACKEND=reference RUNNER=spike "${curated_env[@]}" \
                bash "modelblaster/examples/${model}/run.sh" >/dev/null
        done
    done
else
    log "step 1/5: skipped (--skip-profile)"
fi

# --- Step 2: dispatch graphs + basename fixup (doc section 2) --------------
VMFB_ROOT="${ZCS_ROOT}/gen/vmfb"
if [[ -z "${SKIP_DISPATCH}" ]]; then
    log "step 2/5: emitting dispatch graphs"
    cd "${MB_ROOT}"
    for mq in "${MODEL_QUANTS[@]}"; do
        model="${mq%%:*}"; quant="${mq##*:}"
        for hw in scalar RVV; do
            python3 pipeline/emit_dispatch_graph.py \
                --ir "examples/${model}/${quant}/generated/graph.json" \
                --out-root "${VMFB_ROOT}" --target generic_riscv64 --hw "${hw}"
        done
        # Profile CSVs are always tagged .fp32 regardless of QUANT (a
        # modelblaster quirk, see doc section 2), but emit_dispatch_graph.py
        # tags its output with the real quant -- rename int8 outputs to
        # .fp32 so the scheduler's basename match against the profile CSV
        # succeeds. mlp_control is already fp32, nothing to rename.
        if [[ "${quant}" != "fp32" ]]; then
            for hw in scalar RVV; do
                d="${VMFB_ROOT}/${model}/generic_riscv64/${hw}"
                if [[ -d "${d}/${model}.${quant}" && ! -d "${d}/${model}.fp32" ]]; then
                    mv "${d}/${model}.${quant}" "${d}/${model}.fp32"
                    mv "${d}/${model}.fp32/${model}.${quant}_dispatch_graph.json" \
                       "${d}/${model}.fp32/${model}.fp32_dispatch_graph.json"
                fi
            done
        fi
    done
else
    log "step 2/5: skipped (--skip-dispatch)"
fi

# --- Step 3: bridge profile data into the top-level repo (doc section 3) --
log "step 3/5: bridging profile data (gen_root symlink workaround)"
cd "${TOP_ROOT}"
mkdir -p gen/profile/RVV/spike gen/profile/scalar/spike
for mq in "${MODEL_QUANTS[@]}"; do
    model="${mq%%:*}"
    for hwdir in RVV scalar; do
        link="gen/profile/${hwdir}/spike/${model}"
        if [[ ! -L "${link}" ]]; then
            ln -s "${MB_ROOT}/gen/profile/${hwdir}/spike/${model}" "${link}"
        fi
    done
done

# --- Step 4: generate the schedule (doc section 5) --------------------------
case "${SOLVER}" in
    greedy)          solver_tag="_greedy" ;;
    greedy_periodic) solver_tag="_greedy_periodic" ;;
    decomposed)      solver_tag="_decomposed" ;;
    *)               solver_tag="" ;;
esac
networks_base="$(basename "${NETWORKS_JSON}" .json)"
SCHEDULE_JSON="${TOP_ROOT}/schedules/scheduled_${networks_base}${solver_tag}_profiled.json"

if [[ -z "${SKIP_SCHEDULE}" ]]; then
    log "step 4/5: generating schedule (solver=${SOLVER})"
    cd "${TOP_ROOT}"
    python3 scripts/run_xpurt_schedule.py \
        --networks-json "${NETWORKS_JSON}" \
        --solver "${SOLVER}" --profiled
else
    log "step 4/5: skipped (--skip-schedule)"
fi

if [[ ! -f "${SCHEDULE_JSON}" ]]; then
    echo "ERROR: expected schedule at ${SCHEDULE_JSON} but it doesn't exist" >&2
    echo "  (re-run without --skip-schedule, or check --solver/--networks-json match)" >&2
    exit 1
fi

# --- Step 5: build + run the combined binary on spike (doc section 6) ------
mkdir -p "${TOP_ROOT}/logs"
LOG_PATH="${TOP_ROOT}/logs/xpurt_demo_$(date +%Y%m%d_%H%M%S).log"

if [[ -z "${SKIP_BUILD}" ]]; then
    log "step 5/5: building + running xpurt_demo on spike (log: ${LOG_PATH})"
    cd "${ZCS_ROOT}"
    set +e
    SCHEDULE_JSON="${SCHEDULE_JSON}" \
        MODELS=mlp_control,dronet,yolov8_nano \
        QUANTS=fp32,int8,int8 \
        BACKENDS=scalar,rvv \
        FORCE_REGEN=1 \
        RUNNER=spike \
        XPURT_TRACE="${TRACE:-0}" \
        bash modelblaster/examples/xpurt_demo/run.sh 2>&1 | tee "${LOG_PATH}"
    xpurt_status=${PIPESTATUS[0]}
    set -e
else
    log "step 5/5: skipped (--skip-build)"
    xpurt_status=0
fi

echo
log "=== summary ==="
log "networks-json: ${NETWORKS_JSON}"
log "schedule-json: ${SCHEDULE_JSON}"
if [[ -z "${SKIP_BUILD}" ]]; then
    log "run log:       ${LOG_PATH}"
    grep -E "^(mlp_control|dronet|yolov8_nano|OVERALL):" "${LOG_PATH}" || true
fi

# --- Step 6 (optional): real-execution timeline (doc section 7) ------------
if [[ -n "${TRACE}" && -z "${SKIP_BUILD}" ]]; then
    log "rendering execution timeline (--trace)"
    cd "${MB_ROOT}"
    trace_tag="${networks_base}${solver_tag}"
    python3 -m modelblaster.scripts.plot_xpurt_trace \
        "${LOG_PATH}" \
        --out "${TOP_ROOT}/plots/xpurt_trace_${trace_tag}.png" \
        --csv "${TOP_ROOT}/schedules/xpurt_trace_${trace_tag}.csv" \
        --source spike
    log "timeline plot: plots/xpurt_trace_${trace_tag}.png"
    log "timeline csv:  schedules/xpurt_trace_${trace_tag}.csv"
fi

if [[ -z "${SKIP_BUILD}" && "${xpurt_status}" -ne 0 ]]; then
    log "xpurt_demo/run.sh exited ${xpurt_status} -- see ${LOG_PATH}"
    exit "${xpurt_status}"
fi

log "done."
