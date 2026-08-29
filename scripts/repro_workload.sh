#!/usr/bin/env bash
# One command, one workload spec: profile -> dispatch graphs -> schedule ->
# build -> run, doing whatever the spec describes. The JSON is the single
# source of truth for WHAT runs; the flags below only say which stages to
# skip and how loudly to report.
#
# Usage:
#   bash scripts/repro_workload.sh [SPEC.json] [flags]
#
# The spec drives every stage:
#   hardware.profile.target     spike -> the no-RTL flow documented in
#                               docs/mlp_dronet_yolo_spike_reproduction.md;
#                               anything else is treated as a FireSim
#                               target (docs/end_to_end_xpurt_firesim.md)
#                               and profiled/run under RUNNER=firesim.
#   hardware.profile.topo_tag   topo_0 -> PROFILE_CORES=0, topo_0_1 -> 0,1
#   hardware.profile_hw         the HW backends to profile, emit dispatch
#                               graphs for, and link into the binary. Each
#                               value is the on-disk profile dir name AND
#                               (mapped: *_rvv/RVV -> rvv, gemmini,
#                               gemmini_q31, scalar) the modelblaster
#                               TARGET=. cpu_p/cpu_e also pick the
#                               registry core kinds for the build.
#   networks[*].identifier      which models get profiled and built
#   networks[*].quant           each model's modelblaster quant tree. If
#                               absent it is inferred from the basename in
#                               dispatch_deps_path (<model>.<quant>/),
#                               falling back to fp32.
#   networks[*].dispatch_deps_path
#                               per network: the IREE target its graph is
#                               emitted under, the hw dir it lives in, and
#                               the basename step 2 renames the emitted
#                               graph to (the spike specs point at .fp32
#                               even for int8 models, because profile CSVs
#                               are always tagged .fp32 -- see the doc).
#                               Networks may name different targets.
#   scheduler.*                 read by run_xpurt_schedule.py as usual.
#   flow.*                      OPTIONAL, read only by this script and
#                               ignored by the scheduler:
#     flow.solver               scheduler algorithm (default greedy_periodic)
#     flow.trace                true -> XPURT_TRACE=1 + timeline plot
#     flow.profile.out_root     where profile CSVs land (default
#                               gen/profile under modelblaster for spike,
#                               <zcs>/gen/profile/sweep_v8 for firesim)
#     flow.profile.clock_mhz    PROFILE_CLOCK_MHZ (default 1000.0 on spike,
#                               unset elsewhere)
#     flow.build.registry       modelblaster/cores/*.json for the build --
#                               this is the one thing the spec can't
#                               otherwise say, since it encodes the
#                               bitstream. Without it: xpurt_demo's default
#                               on spike, else the single registry whose
#                               core kinds are exactly cpu_p/cpu_e (and an
#                               error listing candidates if several match).
#     flow.build.backends       explicit BACKENDS= list, overriding the
#                               canonical scalar,rvv,gemmini,gemmini_q31
#                               ordering derived from profile_hw
#     flow.build.firesim_conf   per-bitstream overlay (harness/backends/*.conf)
#     flow.build.firesim_timeout  FIRESIM_TIMEOUT seconds
#     flow.build.force_regen    FORCE_REGEN for xpurt_demo (default 1)
#
# Flags (all optional; they override the spec, never the reverse):
#   --networks-json PATH  same as the positional SPEC argument
#   --quants LIST       comma list overriding the spec's per-model quants,
#                        one entry per model, in spec order
#   --solver NAME       scheduler algorithm (overrides flow.solver)
#   --trace / --no-trace  force the execution-timeline plot on/off
#   --dry-run           print the resolved plan (models, quants, backends,
#                        core kinds, registry, paths) and exit without
#                        running anything
#   --skip-deps         skip installing xpu-rt/modelblaster's own deps
#                        (scripts/install_xpurt_deps.sh) into the active
#                        conda env. Use once you've already run that once
#                        for this env and just want to re-run the pipeline.
#   --skip-profile      skip step 1 (re-profiling every model).
#   --skip-dispatch     skip step 2 (dispatch-graph emit + basename fixup).
#   --skip-schedule     skip step 4 (schedule generation). Requires the
#                        target schedules/scheduled_*.json to already exist.
#   --skip-build        skip step 5 (xpurt_demo build + run + verify).
#   -h, --help          show this help and exit
#
# Examples:
#   # the spike three-network repro, with a timeline plot
#   bash scripts/repro_workload.sh \
#       data/toplevel/networks_mlp_dronet_yolo_spike.json --trace
#
#   # a single-network spike spec -- same command, different JSON
#   bash scripts/repro_workload.sh data/toplevel/networks_dronet_only_spike.json
#
#   # a FireSim spec: gemmini + RVV, RUNNER=firesim end to end
#   bash scripts/repro_workload.sh \
#       data/toplevel/networks_periodic_dronet_yolov8_firesim.json
#
#   # see exactly what a spec would do, without doing it
#   bash scripts/repro_workload.sh <spec.json> --dry-run
#
#   # schedule/logic changed, profile data + dispatch graphs still fresh
#   bash scripts/repro_workload.sh <spec.json> --skip-profile --skip-dispatch

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
DRY_RUN=""
TRACE_FLAG=""          # unset -> take flow.trace from the spec
NETWORKS_JSON=""
QUANTS_OVERRIDE=""
SOLVER_FLAG=""         # unset -> take flow.solver from the spec

while (( $# )); do
    case "$1" in
        --skip-deps)      SKIP_DEPS=1 ;;
        --skip-profile)   SKIP_PROFILE=1 ;;
        --skip-dispatch)  SKIP_DISPATCH=1 ;;
        --skip-schedule)  SKIP_SCHEDULE=1 ;;
        --skip-build)     SKIP_BUILD=1 ;;
        --dry-run)        DRY_RUN=1 ;;
        --trace)          TRACE_FLAG=1 ;;
        --no-trace)       TRACE_FLAG=0 ;;
        --networks-json)  shift; NETWORKS_JSON="$1" ;;
        --quants)         shift; QUANTS_OVERRIDE="$1" ;;
        --solver)         shift; SOLVER_FLAG="$1" ;;
        -h|--help)
            awk 'NR>1 { if ($0 !~ /^#/) exit; sub(/^# ?/, ""); print }' "${BASH_SOURCE[0]}"
            exit 0
            ;;
        -*) echo "unknown flag: $1" >&2; exit 1 ;;
        *)
            if [[ -n "${NETWORKS_JSON}" ]]; then
                echo "ERROR: more than one spec given ('${NETWORKS_JSON}' and '$1')" >&2
                exit 1
            fi
            NETWORKS_JSON="$1"
            ;;
    esac
    shift
done

# Historical default: the spec this flow was first written against.
NETWORKS_JSON="${NETWORKS_JSON:-${TOP_ROOT}/data/toplevel/networks_mlp_dronet_yolo_spike.json}"
case "${NETWORKS_JSON}" in
    /*) : ;;
    *) NETWORKS_JSON="${TOP_ROOT}/${NETWORKS_JSON}" ;;
esac

if [[ ! -f "${NETWORKS_JSON}" ]]; then
    echo "ERROR: workload spec not found: ${NETWORKS_JSON}" >&2
    exit 1
fi

log() { echo "[repro] $*"; }

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

# --- Read the whole workload out of the spec (drives every step below) -----
# Emits shell assignments (shlex-quoted) so the spec stays the single
# source of truth instead of this script carrying a hardcoded copy of one
# particular workload. Runs after the conda activate above (so python3 is
# guaranteed) but before the slow dependency install, so a malformed spec
# fails in seconds rather than minutes.
_spec_vars="$(python3 - "${NETWORKS_JSON}" "${MB_ROOT}" "${ZCS_ROOT}" <<'PY'
import json
import os
import shlex
import sys

path, mb_root, zcs_root = sys.argv[1], sys.argv[2], sys.argv[3]
try:
    spec = json.load(open(path))
except Exception as exc:  # noqa: BLE001 - surface any parse failure verbatim
    sys.exit(f"ERROR: cannot parse {path}: {exc}")


def die(msg):
    sys.exit(f"ERROR: {msg}")


hardware = spec.get("hardware") or {}
profile = hardware.get("profile") or {}
flow = spec.get("flow") or {}
flow_profile = flow.get("profile") or {}
flow_build = flow.get("build") or {}

profile_target = str(profile.get("target") or "").strip()
if not profile_target:
    die(f"{path} has no hardware.profile.target; nothing says where to profile")
# spike is the no-RTL path; everything else is an RTL/FPGA target reached
# through RUNNER=firesim (firesim_rocket_saturn, ...).
runner = "spike" if profile_target == "spike" else "firesim"

# topo_0 -> PROFILE_CORES=0, topo_0_1 -> PROFILE_CORES=0,1. The topo tag is
# what profile_loader looks the CSV up under, so the capture has to be run
# with the matching core list.
topo_tag = str(profile.get("topo_tag") or "topo_0").strip()
if not topo_tag.startswith("topo_"):
    die(f"{path} has hardware.profile.topo_tag='{topo_tag}', expected topo_<cores>")
profile_cores = topo_tag[len("topo_"):].replace("_", ",")
if not profile_cores:
    die(f"{path} has hardware.profile.topo_tag='{topo_tag}' with no core list")

nets = [(k, v) for k, v in (spec.get("networks") or {}).items()
        if not k.startswith("_") and isinstance(v, dict)]
if not nets:
    die(f'{path} has no entries under "networks"')

# Models, per-model quant and per-model dispatch-graph basename, in
# first-appearance order. Periodic workloads repeat the same identifier
# across instances/windows, so dedupe.
models: list[str] = []
quants: dict[str, str] = {}
basenames: dict[str, str] = {}
iree_targets: dict[str, str] = {}
dispatch_hws: dict[str, str] = {}
for key, net in nets:
    model = str(net.get("identifier") or key).strip()
    if not model:
        die(f"network {key!r} in {path} has no identifier")

    deps = str(net.get("dispatch_deps_path") or "").strip()
    # <...>/gen/vmfb/<model>/<iree_target>/<hw>/<basename>/<basename>_dispatch_graph.json
    parts = deps.split("/")
    basename = parts[-2] if len(parts) >= 2 else ""
    dispatch_hw = parts[-3] if len(parts) >= 3 else ""
    # Per-network, not global: a spec may legitimately schedule one model
    # against a spike-emitted graph and another against a firesim one
    # (networks_mlp10_dronet20_yolov8_firesim_static_q31profile.json does).
    iree_target = parts[-4] if len(parts) >= 4 else "generic_riscv64"

    # An explicit quant wins. Otherwise infer it from the basename the
    # spec schedules against -- correct for the firesim specs, which carry
    # the real precision there. The spike specs must declare it, because
    # step 2 renames their graphs to .fp32 to match the profile CSVs.
    quant = str(net.get("quant") or "").strip()
    if not quant:
        quant = basename.split(".", 1)[1] if basename.startswith(f"{model}.") else "fp32"
    if not basename:
        basename = f"{model}.{quant}"

    if model in quants:
        if quants[model] != quant:
            die(f"{path} gives model {model!r} two different quants "
                f"({quants[model]!r} and {quant!r}); one binary can only "
                f"build one quant tree per model")
        if basenames[model] != basename:
            die(f"{path} gives model {model!r} two different dispatch-graph "
                f"basenames ({basenames[model]!r} and {basename!r})")
        continue
    quants[model] = quant
    basenames[model] = basename
    iree_targets[model] = iree_target
    dispatch_hws[model] = dispatch_hw
    models.append(model)

# HW backend names, as spelled by hardware.profile_hw. Each is both the
# <hw> path component of dispatch_deps_path / gen/profile/<hw>/... and,
# once mapped below, a modelblaster TARGET=.
profile_hw = {str(k).strip().lower(): str(v).strip()
              for k, v in (hardware.get("profile_hw") or {}).items()
              if str(v).strip()}
if not profile_hw:
    die(f"{path} has no hardware.profile_hw entries")

hws: list[str] = []
for name in profile_hw.values():
    if name not in hws:
        hws.append(name)


def hw_to_backend(hw: str) -> str:
    """profile_hw name -> modelblaster TARGET= / registry core kind."""
    h = hw.lower()
    if h == "scalar":
        return "scalar"
    if h == "rvv" or h.endswith("_rvv"):      # RVV, V256D128_rvv, ...
        return "rvv"
    if h == "gemmini":
        return "gemmini"
    if h == "gemmini_q31" or h.endswith("_gemmini_q31"):
        return "gemmini_q31"
    die(f"{path} names profile_hw '{hw}', which maps to no modelblaster "
        f"TARGET (expected scalar, RVV/<cfg>_rvv, gemmini or gemmini_q31)")


backends = [hw_to_backend(h) for h in hws]

# spike has no RTL, so only the two software backends are reachable there.
if runner == "spike":
    bad = [h for h, b in zip(hws, backends) if b not in ("scalar", "rvv")]
    if bad:
        die(f"{path} profiles on spike but names profile_hw {bad}, which "
            f"needs RTL; use a firesim target instead")

# BACKENDS= order picks xpurt_demo's build-dir tag, so canonicalise it
# rather than inheriting whatever order profile_hw happens to list: builds
# from this script and from the doc's manual commands then land in the
# same tree. flow.build.backends overrides it outright.
canonical = ["scalar", "rvv", "gemmini", "gemmini_q31"]
build_backends = [b for b in canonical if b in backends]
override = flow_build.get("backends")
if override:
    if isinstance(override, str):
        override = [x.strip() for x in override.split(",") if x.strip()]
    build_backends = [str(x).strip() for x in override]
    unknown = [b for b in build_backends if b not in canonical]
    if unknown:
        die(f"{path} flow.build.backends names {unknown}; expected a subset "
            f"of {canonical}")

# The build links exactly two core kinds (xpurt_demo takes CPU_P_KIND /
# CPU_E_KIND), so the spec's cpu_p/cpu_e entries pick them.
extra = [k for k in profile_hw if k not in ("cpu_p", "cpu_e")]
if extra:
    die(f"{path} has hardware.profile_hw keys {sorted(extra)}; the "
        f"xpurt_demo build understands cpu_p and cpu_e only")
cpu_p_kind = hw_to_backend(profile_hw["cpu_p"]) if "cpu_p" in profile_hw else "rvv"
cpu_e_kind = hw_to_backend(profile_hw["cpu_e"]) if "cpu_e" in profile_hw else "scalar"

# Registry: the core list the build maps hardware_target onto. It encodes
# the bitstream, which the workload spec can't otherwise express, so the
# spec gets to name it outright; failing that we pick the one registry
# whose kinds are exactly the pair this spec asks for.
cores_dir = os.path.join(mb_root, "cores")
default_registry = os.path.join(cores_dir, "chipyard_hetero_example.json")
want_kinds = {cpu_p_kind, cpu_e_kind}


def registry_kinds(reg_path):
    try:
        with open(reg_path) as fh:
            return {str(c.get("kind") or "").strip()
                    for c in (json.load(fh).get("cores") or [])}
    except Exception:
        return None


registry = str(flow_build.get("registry") or "").strip()
if registry:
    registry = registry if os.path.isabs(registry) else os.path.join(mb_root, registry)
    registry_source = "flow.build.registry"
    kinds = registry_kinds(registry)
    if kinds is None:
        die(f"cannot read core registry {registry} named by flow.build.registry")
    missing = sorted(want_kinds - kinds)
    if missing:
        die(f"flow.build.registry ({os.path.relpath(registry, mb_root)}) defines "
            f"no core of kind {missing} (needed for cpu_p={cpu_p_kind}, "
            f"cpu_e={cpu_e_kind})")
elif runner == "spike" and want_kinds <= (registry_kinds(default_registry) or set()):
    # The spike flow is validated against xpurt_demo's own default; there is
    # no bitstream to get wrong, so don't second-guess it.
    registry = default_registry
    registry_source = "xpurt_demo default"
else:
    exact = sorted(f for f in os.listdir(cores_dir)
                   if f.endswith(".json") and registry_kinds(os.path.join(cores_dir, f)) == want_kinds)
    if len(exact) == 1:
        registry = os.path.join(cores_dir, exact[0])
        registry_source = f"only exact match for cpu_p={cpu_p_kind}/cpu_e={cpu_e_kind}"
    else:
        covering = sorted(f for f in os.listdir(cores_dir)
                          if f.endswith(".json")
                          and want_kinds <= (registry_kinds(os.path.join(cores_dir, f)) or set()))
        why = (f"{len(exact)} registries match cpu_p={cpu_p_kind}/cpu_e={cpu_e_kind} "
               f"exactly ({exact})" if exact else
               f"no registry has exactly cpu_p={cpu_p_kind} and cpu_e={cpu_e_kind}")
        die(f"{why}; the right one depends on the bitstream, which {os.path.basename(path)} "
            f"doesn't say. Set flow.build.registry to one of: "
            f"{covering or sorted(os.listdir(cores_dir))}")

# Where profile CSVs land. The spike flow writes into modelblaster's own
# gen/ (relative paths resolve there); the firesim flow writes into the
# zephyr-chipyard-sw sweep tree that scripts/profile_firesim.sh uses.
out_root = str(flow_profile.get("out_root") or "").strip()
if out_root:
    link_root = out_root if os.path.isabs(out_root) else os.path.join(mb_root, out_root)
elif runner == "spike":
    out_root, link_root = "gen/profile", os.path.join(mb_root, "gen", "profile")
else:
    out_root = link_root = os.path.join(zcs_root, "gen", "profile", "sweep_v8")

clock_mhz = flow_profile.get("clock_mhz")
if clock_mhz is None and runner == "spike":
    clock_mhz = 1000.0

out = [
    f"RUNNER_KIND={shlex.quote(runner)}",
    f"PROFILE_TARGET={shlex.quote(profile_target)}",
    f"PROFILE_CORES={shlex.quote(profile_cores)}",
    f"PROFILE_OUT_ROOT_ARG={shlex.quote(out_root)}",
    f"PROFILE_LINK_ROOT={shlex.quote(link_root)}",
    f"PROFILE_CLOCK_MHZ={shlex.quote('' if clock_mhz is None else str(clock_mhz))}",
    "MODELS_ARR=({})".format(" ".join(shlex.quote(m) for m in models)),
    "QUANTS_ARR=({})".format(" ".join(shlex.quote(quants[m]) for m in models)),
    "BASENAMES_ARR=({})".format(" ".join(shlex.quote(basenames[m]) for m in models)),
    "IREE_TARGETS_ARR=({})".format(" ".join(shlex.quote(iree_targets[m]) for m in models)),
    "DISPATCH_HWS_ARR=({})".format(" ".join(shlex.quote(dispatch_hws[m] or "-") for m in models)),
    "HW_DIRS=({})".format(" ".join(shlex.quote(h) for h in hws)),
    "HW_BACKENDS=({})".format(" ".join(shlex.quote(b) for b in backends)),
    f"BACKENDS_CSV={shlex.quote(','.join(build_backends))}",
    f"CPU_P_KIND={shlex.quote(cpu_p_kind)}",
    f"CPU_E_KIND={shlex.quote(cpu_e_kind)}",
    f"REGISTRY={shlex.quote(registry)}",
    f"REGISTRY_SOURCE={shlex.quote(registry_source)}",
    f"SPEC_SOLVER={shlex.quote(str(flow.get('solver') or ''))}",
    "SPEC_TRACE={}".format("1" if flow.get("trace") else ""),
    f"FIRESIM_CONF_SPEC={shlex.quote(str(flow_build.get('firesim_conf') or ''))}",
    f"FIRESIM_TIMEOUT_SPEC={shlex.quote(str(flow_build.get('firesim_timeout') or ''))}",
    f"FORCE_REGEN_SPEC={shlex.quote(str(flow_build.get('force_regen', 1)))}",
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

# Flags beat the spec; the spec beats the built-in default.
SOLVER="${SOLVER_FLAG:-${SPEC_SOLVER:-greedy_periodic}}"
if [[ "${TRACE_FLAG}" == "1" ]]; then
    TRACE=1
elif [[ "${TRACE_FLAG}" == "0" ]]; then
    TRACE=""
else
    TRACE="${SPEC_TRACE}"
fi

MODELS_CSV="$(IFS=,; echo "${MODELS_ARR[*]}")"
QUANTS_CSV="$(IFS=,; echo "${QUANTS_ARR[*]}")"

for model in "${MODELS_ARR[@]}"; do
    if [[ ! -f "${MB_ROOT}/examples/${model}/run.sh" ]]; then
        echo "ERROR: no modelblaster example for model '${model}'" \
             "(expected ${MB_ROOT}/examples/${model}/run.sh)" >&2
        exit 1
    fi
done

case "${SOLVER}" in
    greedy)          solver_tag="_greedy" ;;
    greedy_periodic) solver_tag="_greedy_periodic" ;;
    decomposed)      solver_tag="_decomposed" ;;
    milp)            solver_tag="" ;;          # MILP keeps no infix
    *) echo "ERROR: unknown solver '${SOLVER}' (expected milp, greedy," \
            "greedy_periodic or decomposed)" >&2; exit 1 ;;
esac
networks_base="$(basename "${NETWORKS_JSON}" .json)"
SCHEDULE_JSON="${TOP_ROOT}/schedules/scheduled_${networks_base}${solver_tag}_profiled.json"

log "spec:       ${NETWORKS_JSON}"
iree_targets_seen="$(printf '%s\n' "${IREE_TARGETS_ARR[@]}" | sort -u | paste -sd, -)"
log "target:     ${PROFILE_TARGET} (RUNNER=${RUNNER_KIND}, cores ${PROFILE_CORES}, IREE ${iree_targets_seen})"
log "models:     ${MODELS_CSV}"
log "quants:     ${QUANTS_CSV}"
log "profile hw: ${HW_DIRS[*]} -> TARGET=${HW_BACKENDS[*]}"
log "build:      BACKENDS=${BACKENDS_CSV}, cpu_p=${CPU_P_KIND}, cpu_e=${CPU_E_KIND}"
log "registry:   ${REGISTRY}"
log "            (${REGISTRY_SOURCE})"
log "solver:     ${SOLVER}  (schedule: ${SCHEDULE_JSON})"
log "trace:      ${TRACE:-0}"

if [[ -n "${DRY_RUN}" ]]; then
    log "--dry-run: nothing executed"
    exit 0
fi

if [[ -z "${SKIP_DEPS}" ]]; then
    log "installing xpu-rt/modelblaster deps into this env"
    bash "${TOP_ROOT}/scripts/install_xpurt_deps.sh"
else
    log "skipping dependency install (--skip-deps)"
fi

# --- Step 1: profile each model on the target (doc section 1) --------------
if [[ -z "${SKIP_PROFILE}" ]]; then
    log "step 1/5: profiling each model on ${PROFILE_TARGET} (${HW_DIRS[*]})"
    cd "${ZCS_ROOT}"
    for i in "${!MODELS_ARR[@]}"; do
        model="${MODELS_ARR[$i]}"; quant="${QUANTS_ARR[$i]}"
        for j in "${!HW_DIRS[@]}"; do
            hw="${HW_DIRS[$j]}"; target="${HW_BACKENDS[$j]}"
            extra_env=()
            # Without a curated dir, generate_kernels.py's kernel swap never
            # fires and int8 ops fall back to the slow, non-bit-exact
            # reference implementation (see run.sh comments). The firesim
            # flow always passes one, so its accelerated targets don't
            # silently measure scalar code (scripts/profile_firesim.sh);
            # spike keeps the fp32-untouched rule this flow was validated
            # with.
            if [[ "${RUNNER_KIND}" == "firesim" || "${quant}" != "fp32" ]]; then
                extra_env+=(GLOBAL_CURATED_DIR="${MB_ROOT}/kernels")
            fi
            # scalar has no curated kernels and several int8 specs only
            # register accelerator-affinity algorithms, so BACKEND=llm would
            # empty the queue there; keep scalar (and the whole spike flow,
            # which is validated on it) on the reference emitter.
            backend_kind=reference
            if [[ "${RUNNER_KIND}" == "firesim" && "${target}" != "scalar" ]]; then
                backend_kind=llm
            fi
            if [[ -n "${PROFILE_CLOCK_MHZ}" ]]; then
                extra_env+=(PROFILE_CLOCK_MHZ="${PROFILE_CLOCK_MHZ}")
            fi
            if [[ -n "${FIRESIM_CONF_SPEC}" ]]; then
                extra_env+=(FIRESIM_CONF="${FIRESIM_CONF_SPEC}")
            fi
            if [[ -n "${FIRESIM_TIMEOUT_SPEC}" ]]; then
                extra_env+=(FIRESIM_TIMEOUT="${FIRESIM_TIMEOUT_SPEC}")
            fi
            log "  profiling ${model} (${quant}, TARGET=${target}, hw dir ${hw})"
            env PROFILE_OUT_ROOT="${PROFILE_OUT_ROOT_ARG}" \
                PROFILE_CPU="${PROFILE_TARGET}" \
                PROFILE_CORES="${PROFILE_CORES}" \
                PROFILE_SOURCE="${RUNNER_KIND}" \
                PROFILE_BACKEND="${hw}" \
                QUANT="${quant}" TARGET="${target}" \
                BACKEND="${backend_kind}" RUNNER="${RUNNER_KIND}" \
                "${extra_env[@]}" \
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
        want="${BASENAMES_ARR[$i]}"; tgt="${IREE_TARGETS_ARR[$i]}"
        # Every profile_hw backend the scheduler may assign this model to,
        # plus the hw its own dispatch_deps_path points at (the two differ
        # when a spec schedules one model off a graph emitted for another
        # target's tree).
        hwlist=("${HW_DIRS[@]}")
        dhw="${DISPATCH_HWS_ARR[$i]}"
        if [[ "${dhw}" != "-" && -n "${dhw}" ]]; then
            seen=""
            for hw in "${hwlist[@]}"; do
                if [[ "${hw}" == "${dhw}" ]]; then seen=1; fi
            done
            if [[ -z "${seen}" ]]; then hwlist+=("${dhw}"); fi
        fi
        for hw in "${hwlist[@]}"; do
            python3 pipeline/emit_dispatch_graph.py \
                --ir "examples/${model}/${quant}/generated/graph.json" \
                --out-root "${VMFB_ROOT}" --target "${tgt}" --hw "${hw}"
        done
        # emit_dispatch_graph.py tags its output with the real quant, but
        # the spec schedules against whatever basename dispatch_deps_path
        # names -- .fp32 for the spike specs, because modelblaster's profile
        # CSVs are always tagged .fp32 regardless of QUANT and the scheduler
        # matches profile to graph by basename (doc section 2). Rename to
        # what the spec asked for whenever the two differ.
        got="${model}.${quant}"
        if [[ "${want}" != "${got}" ]]; then
            for hw in "${hwlist[@]}"; do
                d="${VMFB_ROOT}/${model}/${tgt}/${hw}"
                if [[ -d "${d}/${got}" && ! -d "${d}/${want}" ]]; then
                    mv "${d}/${got}" "${d}/${want}"
                    mv "${d}/${want}/${got}_dispatch_graph.json" \
                       "${d}/${want}/${want}_dispatch_graph.json"
                fi
            done
        fi
    done
else
    log "step 2/5: skipped (--skip-dispatch)"
fi

# --- Step 3: bridge profile data into the top-level repo (doc section 3) --
# profile_loader always reads <repo_root>/gen/profile/<hw>/<target>/<model>,
# whatever hardware.profile.gen_root says, so the captures have to be
# reachable from there.
log "step 3/5: bridging profile data (gen_root symlink workaround)"
cd "${TOP_ROOT}"
for hwdir in "${HW_DIRS[@]}"; do
    mkdir -p "gen/profile/${hwdir}/${PROFILE_TARGET}"
    for model in "${MODELS_ARR[@]}"; do
        link="gen/profile/${hwdir}/${PROFILE_TARGET}/${model}"
        src="${PROFILE_LINK_ROOT}/${hwdir}/${PROFILE_TARGET}/${model}"
        if [[ -L "${link}" ]]; then
            continue
        elif [[ -d "${link}" ]]; then
            # A real directory here means someone (e.g.
            # scripts/profile_firesim.sh) already wired per-basename links
            # inside it -- leave their data alone.
            log "  ${link} is a real directory, leaving it as-is"
        else
            ln -s "${src}" "${link}"
        fi
    done
done

# --- Step 4: generate the schedule (doc section 5) --------------------------
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
    echo "  (re-run without --skip-schedule, or check --solver/the spec path)" >&2
    exit 1
fi

# --- Step 5: build + run the combined binary (doc section 6) ---------------
mkdir -p "${TOP_ROOT}/logs"
LOG_PATH="${TOP_ROOT}/logs/xpurt_demo_$(date +%Y%m%d_%H%M%S).log"

if [[ -z "${SKIP_BUILD}" ]]; then
    log "step 5/5: building + running xpurt_demo on ${RUNNER_KIND} (log: ${LOG_PATH})"
    cd "${ZCS_ROOT}"
    build_env=(
        SCHEDULE_JSON="${SCHEDULE_JSON}"
        MODELS="${MODELS_CSV}"
        QUANTS="${QUANTS_CSV}"
        BACKENDS="${BACKENDS_CSV}"
        REGISTRY="${REGISTRY}"
        CPU_P_KIND="${CPU_P_KIND}"
        CPU_E_KIND="${CPU_E_KIND}"
        FORCE_REGEN="${FORCE_REGEN_SPEC}"
        RUNNER="${RUNNER_KIND}"
        XPURT_TRACE="${TRACE:-0}"
    )
    if [[ -n "${FIRESIM_CONF_SPEC}" ]]; then
        build_env+=(FIRESIM_CONF="${FIRESIM_CONF_SPEC}")
    fi
    if [[ -n "${FIRESIM_TIMEOUT_SPEC}" ]]; then
        build_env+=(FIRESIM_TIMEOUT="${FIRESIM_TIMEOUT_SPEC}")
    fi
    set +e
    env "${build_env[@]}" \
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
    log "rendering execution timeline"
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
