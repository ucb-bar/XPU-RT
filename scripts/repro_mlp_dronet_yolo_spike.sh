#!/usr/bin/env bash
# End-to-end reproduction of an XPU-RT schedule on spike (no FireSim),
# driven entirely by a workload spec JSON. Automates every step documented
# in docs/mlp_dronet_yolo_spike_reproduction.md sections 0/1/2/3/5/6/7 in
# order. Read that doc for the "why" behind each step and the bugs each
# workaround here is standing in for; this script is the "just run it"
# path, not a replacement for understanding the flow.
#
# The spec passed via --networks-json is the single source of truth for
# WHAT gets run: which models are profiled, which quant tree each one
# builds from, which HW backends are emitted/linked, and which profile
# target the data is read back from. Defaults to the mlp_control + dronet
# + yolov8_nano spec this doc was written against, but any spike workload
# spec works (e.g. data/toplevel/networks_dronet_only_spike.json).
#
# Usage:
#   bash scripts/repro_mlp_dronet_yolo_spike.sh [flags]
#
# Flags:
#   --networks-json PATH   workload spec (default:
#                        data/toplevel/networks_mlp_dronet_yolo_spike.json).
#                        Models come from its "networks" map (each entry's
#                        "identifier", falling back to the key), per-model
#                        quant from each entry's optional "quant" field
#                        (default fp32), HW backends from
#                        hardware.profile_hw, and the profile target from
#                        hardware.profile.target (must be spike here).
#   --quants LIST       comma list overriding the spec's per-model quants,
#                        one entry per model, in spec order. Use to rebuild
#                        the same topology at a different precision without
#                        editing the spec.
#   --skip-deps         skip installing xpu-rt/modelblaster's own deps
#                        (scripts/install_xpurt_deps.sh) into the active
#                        conda env. Use once you've already run that once
#                        for this env and just want to re-run the pipeline.
#   --skip-profile      skip step 1 (re-profiling every model on spike).
#                        Use once gen/profile/<hw>/spike/<model>
#                        already holds fresh data (e.g. re-running just the
#                        scheduler/build after an unrelated change).
#   --skip-dispatch     skip step 2 (dispatch-graph emit + basename rename).
#   --skip-schedule     skip step 4 (schedule generation). Requires the
#                        target schedules/scheduled_*.json to already exist.
#   --skip-build        skip step 5 (xpurt_demo build + spike run + verify).
#   --trace             enable XPURT_TRACE=1 on the build/run step and
#                        render the real-execution timeline afterward via
#                        plot_xpurt_trace.py (doc section 7).
#   --solver NAME       scheduler algorithm (default: greedy_periodic)
#   -h, --help          show this help and exit
#
# Example (full from-scratch run with a timeline plot):
#   bash scripts/repro_mlp_dronet_yolo_spike.sh --trace
#
# Example (a different workload spec entirely):
#   bash scripts/repro_mlp_dronet_yolo_spike.sh \
#       --networks-json data/toplevel/networks_dronet_only_spike.json
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
QUANTS_OVERRIDE=""
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
        --quants)         shift; QUANTS_OVERRIDE="$1" ;;
        --solver)         shift; SOLVER="$1" ;;
        -h|--help)
            awk 'NR>1 { if ($0 !~ /^#/) exit; sub(/^# ?/, ""); print }' "${BASH_SOURCE[0]}"
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

if [[ ! -f "${NETWORKS_JSON}" ]]; then
    echo "ERROR: workload spec not found: ${NETWORKS_JSON}" >&2
    exit 1
fi

log() { echo "[repro_spike] $*"; }

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

# --- Read the workload out of the spec (drives every step below) ----------
# Emits shell assignments (shlex-quoted) so the spec stays the single
# source of truth for models/quants/backends instead of this script
# carrying a hardcoded copy of one particular workload. Runs after the
# conda activate above (so python3 is guaranteed) but before the slow
# dependency install, so a malformed spec fails in seconds.
_spec_vars="$(python3 - "${NETWORKS_JSON}" <<'PY'
import json
import shlex
import sys

path = sys.argv[1]
try:
    spec = json.load(open(path))
except Exception as exc:  # noqa: BLE001 - surface any parse failure verbatim
    sys.exit(f"ERROR: cannot parse {path}: {exc}")

hardware = spec.get("hardware") or {}
profile = hardware.get("profile") or {}
profile_target = str(profile.get("target") or "").strip()

# Models + per-model quant, in first-appearance order. Periodic workloads
# repeat the same identifier across instances/windows, so dedupe.
models: list[str] = []
quants: dict[str, str] = {}
for key, net in (spec.get("networks") or {}).items():
    if key.startswith("_") or not isinstance(net, dict):
        continue
    model = str(net.get("identifier") or key).strip()
    if not model:
        sys.exit(f"ERROR: network {key!r} in {path} has no identifier")
    # The dispatch graphs are renamed to .fp32 for the scheduler's
    # profile-CSV basename match (see step 2), so dispatch_deps_path can't
    # tell us the real precision -- it has to be declared.
    quant = str(net.get("quant") or "fp32").strip()
    if model in quants:
        if quants[model] != quant:
            sys.exit(
                f"ERROR: {path} gives model {model!r} two different quants "
                f"({quants[model]!r} and {quant!r}); one binary can only "
                f"build one quant tree per model"
            )
        continue
    quants[model] = quant
    models.append(model)

if not models:
    sys.exit(f"ERROR: {path} has no entries under \"networks\"")

# HW backend directory names, as spelled by hardware.profile_hw -- these
# are the <hw> path component of every network's dispatch_deps_path and of
# gen/profile/<hw>/<target>/<model>/.
hws: list[str] = []
for name in (hardware.get("profile_hw") or {}).values():
    name = str(name).strip()
    if name and name not in hws:
        hws.append(name)
if not hws:
    sys.exit(f"ERROR: {path} has no hardware.profile_hw entries")

# The IREE target triple lives in the middle of dispatch_deps_path:
#   <...>/gen/vmfb/<model>/<target>/<hw>/<model>.<quant>/<...>.json
targets = set()
for key, net in (spec.get("networks") or {}).items():
    if key.startswith("_") or not isinstance(net, dict):
        continue
    parts = str(net.get("dispatch_deps_path") or "").split("/")
    if len(parts) >= 4:
        targets.add(parts[-4])
if len(targets) > 1:
    sys.exit(
        f"ERROR: {path} mixes IREE targets {sorted(targets)} across its "
        f"dispatch_deps_path entries; this flow builds one target"
    )
vmfb_target = targets.pop() if targets else "generic_riscv64"

out = [
    f"PROFILE_TARGET={shlex.quote(profile_target)}",
    f"VMFB_TARGET={shlex.quote(vmfb_target)}",
    "MODELS_ARR=({})".format(" ".join(shlex.quote(m) for m in models)),
    "QUANTS_ARR=({})".format(" ".join(shlex.quote(quants[m]) for m in models)),
    "HW_DIRS=({})".format(" ".join(shlex.quote(h) for h in hws)),
]
print("\n".join(out))
PY
)" || exit 1
eval "${_spec_vars}"

# Per-model quant override (--quants), parallel to the spec's model order.
if [[ -n "${QUANTS_OVERRIDE}" ]]; then
    IFS=',' read -ra QUANTS_ARR <<< "${QUANTS_OVERRIDE}"
    if (( ${#QUANTS_ARR[@]} != ${#MODELS_ARR[@]} )); then
        echo "ERROR: --quants needs one entry per model (got ${#QUANTS_ARR[@]}" \
             "for ${#MODELS_ARR[@]} models: ${MODELS_ARR[*]})" >&2
        exit 1
    fi
fi

# This script is the spike (no-RTL) path; a firesim spec needs a different
# board target and profiling runner, so fail loudly rather than silently
# profiling the wrong thing.
if [[ "${PROFILE_TARGET}" != "spike" ]]; then
    echo "ERROR: ${NETWORKS_JSON} has hardware.profile.target='${PROFILE_TARGET}'," \
         "but this script only reproduces the spike flow" >&2
    exit 1
fi

# profile_hw names double as modelblaster TARGET= values (lowercased).
# Only the two software backends are reachable without RTL.
for hw in "${HW_DIRS[@]}"; do
    case "${hw,,}" in
        scalar|rvv) ;;
        *)
            echo "ERROR: hardware.profile_hw names '${hw}', which isn't buildable" \
                 "on spike (expected scalar/RVV); use the firesim flow instead" >&2
            exit 1
            ;;
    esac
done

# Canonical backend order (scalar before rvv), independent of the order
# profile_hw happens to list them in: BACKENDS= order picks xpurt_demo's
# build-dir tag, so a stable order keeps builds from this script and from
# the doc's manual commands landing in the same tree.
PROFILE_TARGETS=()
for t in scalar rvv; do
    for hw in "${HW_DIRS[@]}"; do
        if [[ "${hw,,}" == "${t}" ]]; then
            PROFILE_TARGETS+=("${t}")
            break
        fi
    done
done

MODELS_CSV="$(IFS=,; echo "${MODELS_ARR[*]}")"
QUANTS_CSV="$(IFS=,; echo "${QUANTS_ARR[*]}")"
BACKENDS_CSV="$(IFS=,; echo "${PROFILE_TARGETS[*]}")"

for model in "${MODELS_ARR[@]}"; do
    if [[ ! -f "${MB_ROOT}/examples/${model}/run.sh" ]]; then
        echo "ERROR: no modelblaster example for model '${model}'" \
             "(expected ${MB_ROOT}/examples/${model}/run.sh)" >&2
        exit 1
    fi
done

log "workload spec:  ${NETWORKS_JSON}"
log "models/quants:  ${MODELS_CSV} / ${QUANTS_CSV}"
log "backends:       ${BACKENDS_CSV} (profile dirs: ${HW_DIRS[*]}, target: ${VMFB_TARGET})"

if [[ -z "${SKIP_DEPS}" ]]; then
    log "installing xpu-rt/modelblaster deps into this env"
    bash "${TOP_ROOT}/scripts/install_xpurt_deps.sh"
else
    log "skipping dependency install (--skip-deps)"
fi

# --- Step 1: profile each model on spike (doc section 1) -------------------
if [[ -z "${SKIP_PROFILE}" ]]; then
    log "step 1/5: profiling each model on ${PROFILE_TARGET} (${BACKENDS_CSV})"
    cd "${ZCS_ROOT}"
    for i in "${!MODELS_ARR[@]}"; do
        model="${MODELS_ARR[$i]}"; quant="${QUANTS_ARR[$i]}"
        curated_env=()
        if [[ "${quant}" != "fp32" ]]; then
            # Without this, generate_kernels.py's curated-kernel swap never
            # fires and every int8 op silently falls back to the slow,
            # non-bit-exact reference implementation (see run.sh comments).
            curated_env=(GLOBAL_CURATED_DIR="${MB_ROOT}/kernels")
        fi
        for target in "${PROFILE_TARGETS[@]}"; do
            log "  profiling ${model} (${quant}, TARGET=${target})"
            env PROFILE_OUT_ROOT=gen/profile PROFILE_CPU="${PROFILE_TARGET}" PROFILE_CORES=0 \
                PROFILE_CLOCK_MHZ=1000.0 QUANT="${quant}" TARGET="${target}" \
                BACKEND=reference RUNNER="${PROFILE_TARGET}" "${curated_env[@]}" \
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
    for i in "${!MODELS_ARR[@]}"; do
        model="${MODELS_ARR[$i]}"; quant="${QUANTS_ARR[$i]}"
        for hw in "${HW_DIRS[@]}"; do
            python3 pipeline/emit_dispatch_graph.py \
                --ir "examples/${model}/${quant}/generated/graph.json" \
                --out-root "${VMFB_ROOT}" --target "${VMFB_TARGET}" --hw "${hw}"
        done
        # Profile CSVs are always tagged .fp32 regardless of QUANT (a
        # modelblaster quirk, see doc section 2), but emit_dispatch_graph.py
        # tags its output with the real quant -- rename non-fp32 outputs to
        # .fp32 so the scheduler's basename match against the profile CSV
        # succeeds. An fp32 model is already right, nothing to rename.
        if [[ "${quant}" != "fp32" ]]; then
            for hw in "${HW_DIRS[@]}"; do
                d="${VMFB_ROOT}/${model}/${VMFB_TARGET}/${hw}"
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
for hwdir in "${HW_DIRS[@]}"; do
    mkdir -p "gen/profile/${hwdir}/${PROFILE_TARGET}"
    for model in "${MODELS_ARR[@]}"; do
        link="gen/profile/${hwdir}/${PROFILE_TARGET}/${model}"
        if [[ ! -L "${link}" ]]; then
            ln -s "${MB_ROOT}/gen/profile/${hwdir}/${PROFILE_TARGET}/${model}" "${link}"
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
    log "step 5/5: building + running xpurt_demo on ${PROFILE_TARGET} (log: ${LOG_PATH})"
    cd "${ZCS_ROOT}"
    set +e
    SCHEDULE_JSON="${SCHEDULE_JSON}" \
        MODELS="${MODELS_CSV}" \
        QUANTS="${QUANTS_CSV}" \
        BACKENDS="${BACKENDS_CSV}" \
        FORCE_REGEN=1 \
        RUNNER="${PROFILE_TARGET}" \
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
    summary_re="^($(IFS='|'; echo "${MODELS_ARR[*]}")|OVERALL):"
    grep -E "${summary_re}" "${LOG_PATH}" || true
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
        --source "${PROFILE_TARGET}"
    log "timeline plot: plots/xpurt_trace_${trace_tag}.png"
    log "timeline csv:  schedules/xpurt_trace_${trace_tag}.csv"
fi

if [[ -z "${SKIP_BUILD}" && "${xpurt_status}" -ne 0 ]]; then
    log "xpurt_demo/run.sh exited ${xpurt_status} -- see ${LOG_PATH}"
    exit "${xpurt_status}"
fi

log "done."
