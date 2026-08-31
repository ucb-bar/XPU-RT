#!/usr/bin/env bash
# End-to-end reproduction of an XPU-RT scheduled-workload sweep on FPGA
# (AWS F2 / FireSim, driven through the `fq` queue). This is the FPGA
# counterpart of scripts/repro_mlp_dronet_yolo_spike.sh, and it automates
# every step of
#   runs/sweeps/fpga_20260829-195805/RUNBOOK.md
# in order. Read the RUNBOOK for the "why" behind each step and the bugs
# each workaround here stands in for; this script is the "just run it"
# path, not a replacement for understanding the flow. SETUP.md next to it
# is the experiment design.
#
# Usage:
#   bash scripts/repro_fpga_sweep.sh [flags]
#
# Step flags (each step is independently skippable):
#   --skip-generate     skip step 1 (generate + validate + schedule the
#                        workloads via scripts/sweep_unbounded_nonperiodic.py).
#                        Requires <out-dir>/results.json to already exist --
#                        that file is what tells later steps which points
#                        passed validation.
#   --skip-flatten      skip step 2 (rate-group alias flattening). Requires
#                        <out-dir>/schedules_flat/ to already be populated.
#   --skip-build        skip step 3 (per-point build + fq submit). Requires
#                        <out-dir>/jobs.tsv to already exist.
#   --skip-wait         skip step 4 (wait for the queue to drain + collect
#                        uartlogs into <out-dir>/swres_<point>/uartlog).
#   --skip-verify       skip step 5 (provenance: uartlog `entries=N` must
#                        equal the submitted schedule's dispatch count).
#                        Skipping this is how false results happened before.
#   --skip-analyse      skip step 6 (predicted-vs-actual from the
#                        XPURT_TRACE block -> <out-dir>/fpga_results.json).
#   --dry-run           run steps 1 + 2 only, then print exactly what would
#                        be built and submitted, and exit. Touches no FPGA
#                        and no remote host. Use this before every real run.
#
# Sweep-shape flags:
#   --out-dir DIR       sweep directory (default runs/sweeps/fpga_<UTC ts>;
#                        relative paths are under the xpu-rt root).
#   --seeds SPEC        seed list/range, e.g. 0-7 or 0,1,6 (default 0-7)
#   --arms LIST         comma list of arms from sweep_unbounded_nonperiodic.py
#                        (baseline, fused, baseline_vint, fused_vint,
#                        vint_only, all)               (default baseline,fused)
#   --max-ops N         generator op budget (default 2000; vint arms want
#                        4000-12000, see RUNBOOK step 1)
#   --hardware NAME     generator hardware key (default f2_gemmini_q31_opt)
#   --solver NAME       scheduler algorithm (default greedy)
#   --points LIST       restrict every later step to these point names
#                        (comma list, e.g. baseline_seed0,fused_seed3)
#   --serialize         after flattening, also remove same-network instance
#                        overlap. NOT needed for the vstate fault (that
#                        hypothesis was disproven) but buffers.c really is
#                        one scratch set per model, so concurrent instances
#                        of one network do corrupt each other.
#
# Build/submit flags:
#   --example-dir NAME  modelblaster example tree to build (default
#                        xpurt_demo_armB)
#   --registry PATH     core registry JSON, relative to modelblaster
#                        (default cores/chipyard_dual_rocket_gemmini_q31_f16.json)
#   --backends LIST     harness backends (default gemmini_q31,rvv_f16)
#   --mgr HOST          fq manager (default ubuntu@3.88.218.39)
#   --key PATH          ssh key (default ~/.ssh/firesim.pem)
#   --tree PATH         chipyard tree on the manager that built the bitstream
#                        (default /home/ubuntu/chipyard-rose)
#   --hw-config KEY     hwdb key (default f2_dual_small_norose_tacit_q31_60mhz)
#   --timeout N         per-job wall-clock backstop in seconds (default 3000;
#                        vint points want ~5400)
#   --prefix P          remote ELF/results basename prefix (default sw)
#   --stagger N         seconds between submits (default 12; a same-second
#                        dispatch onto lanes still tearing down has raced)
#
# Collect/analyse flags:
#   --wait-timeout N    give up waiting for the queue after N seconds
#                        (default 10800)
#   --poll N            queue poll interval in seconds (default 30)
#   --trace-clock-mhz F cycles-per-ms divisor for the XPURT_TRACE cycle
#                        columns is F*1000. Default 1.0 -- see "Cycle units"
#                        below. NEVER pass 60.
#   --results-json NAME analysis output basename inside <out-dir>
#                        (default fpga_results.json; results.json is already
#                        taken by the generator's own validation report)
#   -h, --help          show this help and exit
#
# ---------------------------------------------------------------------------
# Things that cost real time to learn (do not "simplify" these away):
#
# * ELF freshness. Every zephyr.elf under the example tree is deleted before
#   each build, the build must exit rc=0, AND the ELF must be newer than the
#   build start. A failed build otherwise leaves the previous point's ELF in
#   place and picks it up -- that produced false results twice in this
#   project, once reporting a prior sweep point's ratio as if it were ViNT's.
#
# * Cycle units. The XPURT_TRACE actual_start/end_cycles columns come from
#   Zephyr's k_cycle_get_64() on the guest, i.e. mtime ticks at
#   CONFIG_SYS_CLOCK_HW_CYCLES_PER_SEC, which is 1000000 for this board --
#   1 tick == 1 us, so 1000 ticks == 1 ms (--trace-clock-mhz 1.0). Verified
#   against the trace itself: mlp_control instance 20 has
#   predicted_start_ms=320.000000 and actual_start_cycles=320000. The 60 MHz
#   in the bitstream name is the HOST FPGA emulation frequency and has
#   nothing to do with this; using 60 scales every result by 16.7x.
#   (RUNBOOK.md used to say "1 GHz target clock, 1 cycle == 1 ns", which is
#   wrong in detail -- the divisor its own recorded fpga_results.json used
#   was 1000, not 1e6. Corrected in the RUNBOOK alongside this script.)
#
# * Sentinel rows. Trace rows with dispatch_id < 0 and a zero actual
#   timestamp never executed (fused/chunk pseudo-ops). They are dropped
#   before any span or error statistic.
#
# * vint arms force --no-horizon-covers-nonperiodic. Otherwise the horizon is
#   extended to cover ViNT's ~9 s at a ~16 ms control period => ~566 mlp
#   instances => ~12k dispatches, which the greedy solver does not finish
#   (measured: >1h39m of CPU on ONE point, killed).
#
# * Alias flattening (step 2) is REQUIRED, not optional. The generator names
#   rate groups dronet_a0, fused_full_b1, ...; ingest_xpurt_schedule only
#   knows base model names and rejects the rest with
#   "references unknown network".
#
# * Filenames are parsed with python/glob, never `ls | grep` -- `ls` output
#   carries ANSI colour in this environment.
#
# * Builds are SERIAL (they share one build dir); FPGA runs are PARALLEL
#   across lanes, so build(N+1) overlaps run(N).
#
# * The per-dispatch IRQ guard (XPURT_DISPATCH_IRQ_GUARD, default 1 in
#   pipeline/generate_xpurt_main.py) is load-bearing for vint arms and costs
#   ~1.5x makespan. It is a compile-time default, not a flag here; see
#   RUNBOOK "Known workarounds".
#
# Example (rehearse a sweep without touching an FPGA):
#   bash scripts/repro_fpga_sweep.sh --seeds 0-7 --arms baseline,fused \
#       --max-ops 2000 --out-dir runs/sweeps/mysweep --dry-run
#
# Example (the real sweep A from the RUNBOOK):
#   bash scripts/repro_fpga_sweep.sh --seeds 0-7 --arms baseline,fused \
#       --max-ops 2000 --out-dir runs/sweeps/mysweep
#
# Example (sweep B, heavy non-periodic vision model):
#   bash scripts/repro_fpga_sweep.sh --seeds 0-3 --arms fused_vint \
#       --max-ops 4000 --timeout 5400 --out-dir runs/sweeps/mysweep_vint
#
# Example (re-verify + re-analyse an already-collected sweep, no FPGA):
#   bash scripts/repro_fpga_sweep.sh --out-dir runs/sweeps/fpga_20260829-195805 \
#       --skip-generate --skip-flatten --skip-build --skip-wait
# ---8<--- end of help ---8<---

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
[[ -d "${ZCS_ROOT}" ]] || {
    echo "ERROR: zephyr-chipyard-sw submodule not checked out at ${ZCS_ROOT}" >&2
    echo "  run: git -C ${TOP_ROOT} submodule update --init ${_zcs_path}" >&2
    exit 1
}
_mb_path="$(git -C "${ZCS_ROOT}" config -f .gitmodules --get submodule.modelblaster.path)" || {
    echo "ERROR: no 'modelblaster' submodule entry in ${ZCS_ROOT}/.gitmodules" >&2
    exit 1
}
MB_ROOT="${ZCS_ROOT}/${_mb_path}"
[[ -d "${MB_ROOT}" ]] || {
    echo "ERROR: modelblaster submodule not checked out at ${MB_ROOT}" >&2
    echo "  run: git -C ${ZCS_ROOT} submodule update --init ${_mb_path}" >&2
    exit 1
}

SKIP_GENERATE=""; SKIP_FLATTEN=""; SKIP_BUILD=""
SKIP_WAIT=""; SKIP_VERIFY=""; SKIP_ANALYSE=""
DRY_RUN=""; SERIALIZE=""
OUT_DIR="runs/sweeps/fpga_$(date -u +%Y%m%d-%H%M%S)"
SEEDS="0-7"
ARMS="baseline,fused"
MAX_OPS="2000"
HARDWARE="f2_gemmini_q31_opt"
SOLVER="greedy"
POINTS_FILTER=""
EXAMPLE_DIR="xpurt_demo_armB"
REGISTRY="cores/chipyard_dual_rocket_gemmini_q31_f16.json"
BACKENDS="gemmini_q31,rvv_f16"
MGR="ubuntu@3.88.218.39"
KEY="${HOME}/.ssh/firesim.pem"
TREE="/home/ubuntu/chipyard-rose"
HW_CONFIG="f2_dual_small_norose_tacit_q31_60mhz"
JOB_TIMEOUT="3000"
PREFIX="sw"
STAGGER="12"
WAIT_TIMEOUT="10800"
POLL="30"
TRACE_CLOCK_MHZ="1.0"
RESULTS_JSON="fpga_results.json"

while (( $# )); do
    case "$1" in
        --skip-generate)  SKIP_GENERATE=1 ;;
        --skip-flatten)   SKIP_FLATTEN=1 ;;
        --skip-build)     SKIP_BUILD=1 ;;
        --skip-wait)      SKIP_WAIT=1 ;;
        --skip-verify)    SKIP_VERIFY=1 ;;
        --skip-analyse)   SKIP_ANALYSE=1 ;;
        --dry-run)        DRY_RUN=1 ;;
        --serialize)      SERIALIZE=1 ;;
        --out-dir)        shift; OUT_DIR="$1" ;;
        --seeds)          shift; SEEDS="$1" ;;
        --arms)           shift; ARMS="$1" ;;
        --max-ops)        shift; MAX_OPS="$1" ;;
        --hardware)       shift; HARDWARE="$1" ;;
        --solver)         shift; SOLVER="$1" ;;
        --points)         shift; POINTS_FILTER="$1" ;;
        --example-dir)    shift; EXAMPLE_DIR="$1" ;;
        --registry)       shift; REGISTRY="$1" ;;
        --backends)       shift; BACKENDS="$1" ;;
        --mgr)            shift; MGR="$1" ;;
        --key)            shift; KEY="$1" ;;
        --tree)           shift; TREE="$1" ;;
        --hw-config)      shift; HW_CONFIG="$1" ;;
        --timeout)        shift; JOB_TIMEOUT="$1" ;;
        --prefix)         shift; PREFIX="$1" ;;
        --stagger)        shift; STAGGER="$1" ;;
        --wait-timeout)   shift; WAIT_TIMEOUT="$1" ;;
        --poll)           shift; POLL="$1" ;;
        --trace-clock-mhz) shift; TRACE_CLOCK_MHZ="$1" ;;
        --results-json)   shift; RESULTS_JSON="$1" ;;
        -h|--help)
            awk 'NR>1 && /^# ---8<---/{exit} NR>1 && /^#/{sub(/^# ?/,"");print}' "${BASH_SOURCE[0]}"
            exit 0
            ;;
        *) echo "unknown flag: $1" >&2; exit 1 ;;
    esac
    shift
done

case "${OUT_DIR}" in /*) : ;; *) OUT_DIR="${TOP_ROOT}/${OUT_DIR}" ;; esac
case "${REGISTRY}" in /*) : ;; *) REGISTRY="${MB_ROOT}/${REGISTRY}" ;; esac

# Solver -> schedule filename infix, mirroring run_xpurt_schedule.py's own
# naming (`schedules/scheduled_<base><solver_tag>{_profiled}.json`).
case "${SOLVER}" in
    greedy)          SOLVER_TAG="_greedy" ;;
    greedy_periodic) SOLVER_TAG="_greedy_periodic" ;;
    decomposed)      SOLVER_TAG="_decomposed" ;;
    *)               SOLVER_TAG="" ;;
esac
SCHED_SUFFIX="${SOLVER_TAG}_profiled.json"

EX_ROOT="${MB_ROOT}/examples/${EXAMPLE_DIR}"
[[ -d "${EX_ROOT}" ]] || { echo "ERROR: no example tree at ${EX_ROOT}" >&2; exit 1; }

JOBS_TSV="${OUT_DIR}/jobs.tsv"
FLAT_DIR="${OUT_DIR}/schedules_flat"
mkdir -p "${OUT_DIR}" "${FLAT_DIR}" "${OUT_DIR}/buildlogs" "${OUT_DIR}/logs"

log() { echo "[repro_fpga_sweep] $*"; }

# --- Step 0: environment (RUNBOOK section 0) -------------------------------
log "activating zephyr toolchain env"
# shellcheck disable=SC1091
source "${ZCS_ROOT}/scripts/activate_conda.sh"
# NOTE: set_envvars_sdk.sh reassigns REPO_ROOT to the zephyr-chipyard-sw
# root. Nothing below reads REPO_ROOT (run.sh recomputes its own), so we let
# it stand rather than save/restore.
# shellcheck disable=SC1091
source "${ZCS_ROOT}/scripts/set_envvars_sdk.sh"
export PATH="/usr/bin:${PATH}"
export PATH="${CONDA_PREFIX}/bin:${PATH}"
export PYTHONPATH="${ZCS_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

# vint arms MUST turn horizon extension off, or the greedy solver never
# finishes (RUNBOOK step 1). Force it rather than trusting the caller.
HORIZON_FLAG=()
case ",${ARMS}," in
    *vint*) HORIZON_FLAG=(--no-horizon-covers-nonperiodic)
            log "arms contain vint -> forcing --no-horizon-covers-nonperiodic" ;;
esac

# --- Step 1: generate + validate + schedule (RUNBOOK section 1) ------------
if [[ -z "${SKIP_GENERATE}" ]]; then
    log "step 1/6: generating + validating + scheduling (arms=${ARMS} seeds=${SEEDS} max-ops=${MAX_OPS})"
    cd "${TOP_ROOT}"
    python3 scripts/sweep_unbounded_nonperiodic.py \
        --seeds "${SEEDS}" --arms "${ARMS}" --max-ops "${MAX_OPS}" \
        --hardware "${HARDWARE}" --solver "${SOLVER}" --schedule \
        "${HORIZON_FLAG[@]}" \
        --out-dir "${OUT_DIR}" 2>&1 | tee "${OUT_DIR}/generate.log"
else
    log "step 1/6: skipped (--skip-generate)"
fi

[[ -f "${OUT_DIR}/results.json" ]] || {
    echo "ERROR: ${OUT_DIR}/results.json missing -- that is the generator's" >&2
    echo "  validation report and the only record of which points passed." >&2
    echo "  Re-run without --skip-generate." >&2
    exit 1
}

# Points = arm/seed pairs that PASSED validation. Rejected points are never
# built (RUNBOOK step 1). Read from the generator's own report rather than
# globbing schedules/, so a leftover schedule from an earlier sweep cannot
# smuggle a rejected point back in.
mapfile -t POINTS < <(python3 - "${OUT_DIR}/results.json" "${POINTS_FILTER}" <<'PY'
import json, sys
rows = json.load(open(sys.argv[1]))
keep = {p for p in sys.argv[2].split(",") if p.strip()}
for r in rows:
    if r.get("status") != "ok":
        continue
    name = f"{r['arm']}_seed{r['seed']}"
    if keep and name not in keep:
        continue
    print(name)
PY
)
(( ${#POINTS[@]} )) || { echo "ERROR: no validated points in ${OUT_DIR}/results.json" >&2; exit 1; }
log "  ${#POINTS[@]} validated point(s): ${POINTS[*]}"

# --- Step 2: flatten rate-group aliases (RUNBOOK section 2) ----------------
# REQUIRED. The generator names rate groups dronet_a0/fused_full_b1/...;
# ingest_xpurt_schedule only knows base model names. Base names come from
# the model bank, never a hardcoded list -- a hardcoded triple is exactly
# what silently left fused_full/vint aliases unflattened before.
if [[ -z "${SKIP_FLATTEN}" ]]; then
    log "step 2/6: flattening rate-group aliases -> ${FLAT_DIR}"
    for point in "${POINTS[@]}"; do
        src="${TOP_ROOT}/schedules/scheduled_${point}${SCHED_SUFFIX}"
        [[ -f "${src}" ]] || { echo "  ${point}: NO SCHEDULE at ${src}"; continue; }
        echo "  ${point}:"
        python3 - "${src}" "${FLAT_DIR}/$(basename "${src}")" \
                 "${TOP_ROOT}/data/banks/model_bank.json" <<'PY'
import collections, json, re, sys
src, dst, bank_path = sys.argv[1], sys.argv[2], sys.argv[3]
bases = set()
for plat in (json.load(open(bank_path)).get("platforms") or {}).values():
    bases.update((plat.get("models") or {}).keys())
BASES = tuple(sorted(bases)) or ("mlp_control", "dronet", "yolov8_nano")
d = json.load(open(src))
jobs = []
for k in d["dispatches"]:
    j = k.split("_dispatch")[0]
    if j not in jobs:
        jobs.append(j)
def parse(j):
    for b in sorted(BASES, key=len, reverse=True):
        m = re.match(rf"^{b}(?:_([a-z]))?(\d*)$", j)
        if m:
            return b, (m.group(1) or ""), (int(m.group(2)) if m.group(2) else 0)
    return None, None, None
groups = collections.defaultdict(list)
for j in jobs:
    b, a, i = parse(j)
    if b:
        groups[b].append((a, i, j))
jm = {}
for b, lst in groups.items():
    for n, (a, i, j) in enumerate(sorted(lst)):   # alias then index -> contiguous
        jm[j] = f"{b}{n}"
for j in jobs:
    jm.setdefault(j, j)
km = {k: jm[k.split("_dispatch")[0]] + k[len(k.split("_dispatch")[0]):]
      for k in d["dispatches"]}
def remap(x):
    if isinstance(x, str):  return km.get(x, jm.get(x, x))
    if isinstance(x, list): return [remap(i) for i in x]
    if isinstance(x, dict): return {k: remap(v) for k, v in x.items()}
    return x
out = {}
for k, v in d["dispatches"].items():
    v = remap(dict(v)); v["job_name"] = jm[k.split("_dispatch")[0]]
    out[km[k]] = v
d["dispatches"] = out
json.dump(d, open(dst, "w"), indent=1)
keys = set(out)
bad = [x for v in out.values() for f in ("dependencies", "time_dependency")
       for x in (v.get(f) if isinstance(v.get(f), list) else [v.get(f)])
       if isinstance(x, str) and x not in keys]
print(f"    {len(jobs)} jobs -> {len(set(jm.values()))} instances; "
      f"renamed {sum(1 for k, v in jm.items() if k != v)}; dangling {len(bad)}")
if bad:
    sys.exit("    ERROR: dangling references after flatten -- refusing to build")
PY
    done
else
    log "step 2/6: skipped (--skip-flatten)"
fi

# Optional: remove same-network instance overlap. buffers.c is ONE scratch
# set per model ("must be linked EXACTLY ONCE per model"), so two concurrent
# instances of one network clobber each other. Off by default: this was NOT
# the cause of the vstate fault, but the hazard is real.
if [[ -n "${SERIALIZE}" && -z "${SKIP_FLATTEN}" ]]; then
    log "  serialising same-network instances (--serialize)"
    for point in "${POINTS[@]}"; do
        f="${FLAT_DIR}/scheduled_${point}${SCHED_SUFFIX}"
        [[ -f "${f}" ]] || continue
        printf '    %s: ' "${point}"
        python3 - "${f}" "${f}" <<'PY'
import collections, json, re, sys
src, dst = sys.argv[1], sys.argv[2]
d = json.load(open(src)); D = d["dispatches"]
KEY = re.compile(r"^([a-z0-9_]+?)(\d*)_dispatch_")
inst = collections.defaultdict(list)
for k in D:
    m = KEY.match(k)
    if m:
        inst[(m.group(1), int(m.group(2) or 0))].append(k)
span = {k: (min(float(D[x]["start_time"]) for x in v),
            max(float(D[x]["start_time"]) + float(D[x]["duration"]) for x in v))
        for k, v in inst.items()}
bynet = collections.defaultdict(list)
for (net, idx) in inst:
    bynet[net].append(idx)
shifted = 0
for net, idxs in bynet.items():
    if len(idxs) < 2:
        continue
    free = None
    for i in sorted(idxs, key=lambda i: span[(net, i)][0]):
        s, e = span[(net, i)]
        if free is not None and s < free:
            off = free - s
            for k in inst[(net, i)]:
                D[k]["start_time"] = float(D[k]["start_time"]) + off
            s, e = s + off, e + off
            span[(net, i)] = (s, e)
            shifted += 1
        free = e
mk = max(float(v["start_time"]) + float(v["duration"]) for v in D.values())
d.setdefault("metadata", {})["makespan"] = mk
d["metadata"]["serialized_same_network_instances"] = True
json.dump(d, open(dst, "w"), indent=1)
print(f"shifted {shifted} instance(s); makespan -> {mk:.2f} ms")
PY
    done
fi

# --- Build plan: MODELS / MODEL_EXDIRS / QUANTS per arm --------------------
# The model list per arm is read straight out of sweep_unbounded_nonperiodic.py
# so the two cannot drift. Example dir = <model>_armB except for the models
# that only exist under their own name; quant = fp32 for mlp_control, int8
# for everything else.
plan_for_point() {   # $1=point -> prints "MODELS<TAB>EXDIRS<TAB>QUANTS"
    python3 - "$1" "${TOP_ROOT}" "${MB_ROOT}" <<'PY'
import os, sys
point, top, mb = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, os.path.join(top, "scripts"))
from sweep_unbounded_nonperiodic import ARMS
arm = point.rsplit("_seed", 1)[0]
models = ARMS[arm]
exdirs, quants = [], []
for m in models:
    cand = f"{m}_armB"
    exdirs.append(cand if os.path.isdir(os.path.join(mb, "examples", cand)) else m)
    quants.append("fp32" if m == "mlp_control" else "int8")
print("\t".join([",".join(models), ",".join(exdirs), ",".join(quants)]))
PY
}

dispatch_count() {   # $1=flattened schedule json
    python3 -c "import json,sys;print(len(json.load(open(sys.argv[1]))['dispatches']))" "$1"
}

# Locate the freshest zephyr.elf under the example tree, newer than a stamp
# file. python/glob, not `ls -t` -- and the stamp is the freshness gate.
find_fresh_elf() {   # $1=stamp file -> prints path or nothing
    python3 - "${EX_ROOT}" "$1" <<'PY'
import glob, os, sys
root, stamp = sys.argv[1], sys.argv[2]
t0 = os.path.getmtime(stamp)
cands = [p for p in glob.glob(os.path.join(root, "**", "zephyr.elf"), recursive=True)
         if os.path.getmtime(p) > t0]
if cands:
    print(max(cands, key=os.path.getmtime))
PY
}

SSH=(ssh -n -o BatchMode=yes -o ConnectTimeout=30 -i "${KEY}")
FQ='cd ~/fpga_queue && export FQ_SOCKET=/var/lib/fq/fq.sock &&'

# --- Dry run: print the plan and stop --------------------------------------
if [[ -n "${DRY_RUN}" ]]; then
    echo
    log "=== DRY RUN: steps 1+2 done, nothing will be built or submitted ==="
    log "manager      ${MGR}   (untouched)"
    log "hw-config    ${HW_CONFIG}   tree ${TREE}"
    log "example tree ${EX_ROOT}"
    log "registry     ${REGISTRY}"
    log "backends     ${BACKENDS}   timeout ${JOB_TIMEOUT}s   stagger ${STAGGER}s"
    for point in "${POINTS[@]}"; do
        sj="${FLAT_DIR}/scheduled_${point}${SCHED_SUFFIX}"
        if [[ ! -f "${sj}" ]]; then
            echo "  ${point}: NO FLATTENED SCHEDULE (would be skipped)"
            continue
        fi
        IFS=$'\t' read -r m x q < <(plan_for_point "${point}")
        want="$(dispatch_count "${sj}")"
        echo
        echo "  --- ${point}  (${want} dispatches)"
        echo "      schedule   ${sj}"
        echo "      build      SCHEDULE_JSON=<above> MODELS=${m} MODEL_EXDIRS=${x} QUANTS=${q} \\"
        echo "                 BACKENDS=${BACKENDS} REGISTRY=${REGISTRY} \\"
        echo "                 CPU_P_KIND=gemmini_q31 CPU_E_KIND=rvv_f16 SCHED_NAME=${point} \\"
        echo "                 STOP_AFTER=build RUNNER=firesim FORCE_REGEN=0 XPURT_TRACE=1 \\"
        echo "                 bash examples/${EXAMPLE_DIR}/run.sh"
        echo "      ship       scp <fresh zephyr.elf> ${MGR}:/home/ubuntu/${PREFIX}_${point}.elf"
        echo "      submit     fq submit --tree ${TREE} --hw-config ${HW_CONFIG} \\"
        echo "                 --elf /home/ubuntu/${PREFIX}_${point}.elf --timeout ${JOB_TIMEOUT} \\"
        echo "                 --results /home/ubuntu/${PREFIX}res_${point}"
        echo "      expect     uartlog 'schedule=${point} entries=${want}'"
    done
    echo
    log "dry run complete -- drop --dry-run to execute."
    exit 0
fi

# --- Step 3: build + submit each point (RUNBOOK section 3) -----------------
if [[ -z "${SKIP_BUILD}" ]]; then
    log "step 3/6: building + submitting ${#POINTS[@]} point(s) (builds are SERIAL)"
    [[ -f "${KEY}" ]] || { echo "ERROR: ssh key ${KEY} not found" >&2; exit 1; }
    cd "${MB_ROOT}"
    : > "${JOBS_TSV}"
    for point in "${POINTS[@]}"; do
        sj="${FLAT_DIR}/scheduled_${point}${SCHED_SUFFIX}"
        [[ -f "${sj}" ]] || { echo "  ${point}: NO FLATTENED SCHEDULE"; continue; }
        IFS=$'\t' read -r models exdirs quants < <(plan_for_point "${point}")
        want="$(dispatch_count "${sj}")"
        echo "=== ${point}  (${want} dispatches)  $(date +%H:%M:%S) ==="

        # Freshness gate, part 1: nothing stale may survive into this build.
        find "${EX_ROOT}" -name zephyr.elf -delete 2>/dev/null || true
        stamp="${OUT_DIR}/.buildstamp"; : > "${stamp}"

        set +e
        SCHEDULE_JSON="${sj}" MODELS="${models}" MODEL_EXDIRS="${exdirs}" \
        QUANTS="${quants}" BACKENDS="${BACKENDS}" REGISTRY="${REGISTRY}" \
        CPU_P_KIND=gemmini_q31 CPU_E_KIND=rvv_f16 SCHED_NAME="${point}" \
        STOP_AFTER=build RUNNER=firesim FORCE_REGEN=0 XPURT_TRACE=1 \
            bash "examples/${EXAMPLE_DIR}/run.sh" \
            > "${OUT_DIR}/buildlogs/${point}.log" 2>&1
        rc=$?
        set -e
        # Freshness gate, part 2: rc must be 0 AND an ELF must postdate the
        # build start. Without both, a failed build silently reuses the
        # previous point's binary.
        if (( rc != 0 )); then
            echo "  BUILD FAILED rc=${rc}"
            grep -nE 'error:|ValueError' "${OUT_DIR}/buildlogs/${point}.log" | tail -3 || true
            continue
        fi
        elf="$(find_fresh_elf "${stamp}")"
        if [[ -z "${elf}" || ! -f "${elf}" ]]; then
            echo "  NO FRESH ELF (build claimed rc=0 -- discarding point)"
            continue
        fi
        echo "  elf ${elf} ($(stat -c %s "${elf}") bytes)"

        remote_elf="/home/ubuntu/${PREFIX}_${point}.elf"
        remote_res="/home/ubuntu/${PREFIX}res_${point}"
        scp -q -o BatchMode=yes -i "${KEY}" "${elf}" "${MGR}:${remote_elf}" || {
            echo "  scp failed"; continue; }
        "${SSH[@]}" "${MGR}" "rm -rf ${remote_res} && mkdir -p ${remote_res}" </dev/null
        out="$("${SSH[@]}" "${MGR}" "${FQ} timeout 60 ./bin/fq submit \
            --tree ${TREE} --hw-config ${HW_CONFIG} --elf ${remote_elf} \
            --timeout ${JOB_TIMEOUT} --results ${remote_res} 2>&1 | head -1" </dev/null)"
        jid="$(grep -oE 'job [0-9]+' <<<"${out}" | grep -oE '[0-9]+' | head -1)"
        echo "  -> job ${jid:-?}  (expect entries=${want})"
        printf '%s\t%s\t%s\n' "${point}" "${jid:-NONE}" "${want}" >> "${JOBS_TSV}"
        # Stagger: a same-second dispatch onto lanes still tearing down has raced.
        sleep "${STAGGER}"
    done
    log "  all submitted; job map in ${JOBS_TSV}"
else
    log "step 3/6: skipped (--skip-build)"
fi

[[ -f "${JOBS_TSV}" ]] || {
    echo "ERROR: ${JOBS_TSV} missing -- nothing was submitted and nothing to" >&2
    echo "  wait for. Re-run without --skip-build." >&2
    exit 1
}

# --- Step 4: wait for drain + collect uartlogs (RUNBOOK section 4) ---------
if [[ -z "${SKIP_WAIT}" ]]; then
    log "step 4/6: waiting for the queue to drain (timeout ${WAIT_TIMEOUT}s)"
    deadline=$(( $(date +%s) + WAIT_TIMEOUT ))
    while :; do
        pending=0
        while IFS=$'\t' read -r point jid want; do
            [[ "${jid}" == "NONE" || -z "${jid}" ]] && continue
            st="$("${SSH[@]}" "${MGR}" \
                "grep -E 'job ${jid} finished' /var/lib/fq/daemon.out 2>/dev/null | tail -1" </dev/null)"
            [[ -z "${st}" ]] && pending=$(( pending + 1 ))
        done < "${JOBS_TSV}"
        (( pending == 0 )) && break
        if (( $(date +%s) > deadline )); then
            log "  WAIT TIMEOUT with ${pending} job(s) still pending -- collecting anyway"
            break
        fi
        log "  ${pending} job(s) pending; sleeping ${POLL}s"
        sleep "${POLL}"
    done

    log "  collecting uartlogs"
    while IFS=$'\t' read -r point jid want; do
        dest="${OUT_DIR}/swres_${point}"
        mkdir -p "${dest}"
        verdict="$("${SSH[@]}" "${MGR}" \
            "grep -E 'job ${jid} finished' /var/lib/fq/daemon.out 2>/dev/null | tail -1" </dev/null)"
        echo "  ${point}: ${verdict:-NO TERMINAL STATE}"
        "${SSH[@]}" "${MGR}" \
            "find /home/ubuntu/${PREFIX}res_${point} -name uartlog | head -1 | xargs -r cat" \
            </dev/null > "${dest}/uartlog" 2>/dev/null || true
        [[ -s "${dest}/uartlog" ]] || echo "    no uartlog collected"
    done < "${JOBS_TSV}"
else
    log "step 4/6: skipped (--skip-wait)"
fi

# --- Step 5: provenance (RUNBOOK section 4, "Always verify provenance") ----
# The uartlog's embedded `schedule=<name> entries=N` must match what we
# submitted. A mismatch means a stale binary ran; the point is discarded.
if [[ -z "${SKIP_VERIFY}" ]]; then
    log "step 5/6: verifying provenance"
    python3 - "${OUT_DIR}" "${JOBS_TSV}" <<'PY'
import os, re, sys
out, tsv = sys.argv[1], sys.argv[2]
bad = 0
for line in open(tsv):
    parts = line.rstrip("\n").split("\t")
    if len(parts) != 3:
        continue
    point, jid, want = parts[0], parts[1], int(parts[2])
    p = os.path.join(out, f"swres_{point}", "uartlog")
    if not os.path.exists(p) or os.path.getsize(p) == 0:
        print(f"  {point:24s} MISSING UARTLOG"); bad += 1; continue
    text = open(p, errors="ignore").read()
    m = re.search(r"schedule=(\S+)\s+entries=(\d+)", text)
    if not m:
        print(f"  {point:24s} NO xpurt-runner BANNER"); bad += 1; continue
    got_name, got_n = m.group(1), int(m.group(2))
    ok = (got_name == point and got_n == want)
    print(f"  {point:24s} job={jid:<6s} schedule={got_name:24s} "
          f"entries={got_n:<6d} expect={want:<6d} {'OK' if ok else 'MISMATCH -> DISCARD'}")
    bad += 0 if ok else 1
print(f"  {bad} point(s) failed provenance" if bad else "  all points provenance-verified")
sys.exit(0)
PY
else
    log "step 5/6: skipped (--skip-verify)"
fi

# --- Step 6: predicted vs actual (RUNBOOK sections 4/5) --------------------
if [[ -z "${SKIP_ANALYSE}" ]]; then
    log "step 6/6: analysing XPURT_TRACE -> ${OUT_DIR}/${RESULTS_JSON}"
    python3 - "${OUT_DIR}" "${JOBS_TSV}" "${TRACE_CLOCK_MHZ}" "${OUT_DIR}/${RESULTS_JSON}" <<'PY'
import csv, io, json, os, re, statistics, sys
out, tsv, clock_mhz, dst = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4]
CPMS = clock_mhz * 1000.0          # cycles (mtime ticks) per millisecond
BEGIN = "=== MODELBLASTER_XPURT_TRACE_BEGIN ==="
END = "=== MODELBLASTER_XPURT_TRACE_END ==="

def rows_of(text):
    if BEGIN not in text or END not in text:
        return None, 0
    body = text[text.index(BEGIN) + len(BEGIN):text.index(END)].strip()
    keep, sentinels = [], 0
    for r in csv.DictReader(io.StringIO(body)):
        try:
            did = int(r["dispatch_id"])
            s, e = int(r["actual_start_cycles"]), int(r["actual_end_cycles"])
            ps, pd = float(r["predicted_start_ms"]), float(r["predicted_duration_ms"])
        except (TypeError, ValueError):
            continue
        # Sentinels: pseudo-ops that never executed (dispatch_id < 0 with a
        # zero timestamp). They would drag the span start to 0.
        if did < 0 and s == 0 and e == 0:
            sentinels += 1
            continue
        keep.append((ps, pd, s, e))
    return keep, sentinels

results = []
for line in open(tsv):
    parts = line.rstrip("\n").split("\t")
    if len(parts) != 3:
        continue
    point, jid, want = parts[0], parts[1], int(parts[2])
    p = os.path.join(out, f"swres_{point}", "uartlog")
    if not os.path.exists(p) or os.path.getsize(p) == 0:
        print(f"  {point:24s} no uartlog -- skipped")
        continue
    text = open(p, errors="ignore").read()
    m = re.search(r"schedule=(\S+)\s+entries=(\d+)", text)
    rows, sentinels = rows_of(text)
    if not rows:
        print(f"  {point:24s} no XPURT_TRACE block -- skipped "
              f"(was the build XPURT_TRACE=1?)")
        continue
    pred_ms = max(ps + pd for ps, pd, _, _ in rows)
    act_ms = (max(e for *_, e in rows) - min(s for *_, s, _ in rows)) / CPMS
    errs = sorted(abs(((e - s) / CPMS) - pd) / pd for _, pd, s, e in rows if pd > 0)
    rec = dict(
        point=point, job=jid,
        provenance_ok=bool(m and m.group(1) == point and int(m.group(2)) == want),
        expected_entries=want,
        uartlog_entries=int(m.group(2)) if m else None,
        trace_rows=len(rows), sentinels=sentinels,
        pred_ms=pred_ms, act_ms=act_ms, ratio=act_ms / pred_ms,
        per_op_p50_err_pct=100 * statistics.median(errs) if errs else None,
        per_op_p90_err_pct=100 * errs[int(0.9 * (len(errs) - 1))] if errs else None,
    )
    results.append(rec)
    flag = "" if rec["provenance_ok"] else "  [PROVENANCE MISMATCH]"
    print(f"  {point:24s} entries={want:<6d} pred={pred_ms:10.3f} ms  "
          f"act={act_ms:10.3f} ms  ratio={rec['ratio']:.4f}  "
          f"p50err={rec['per_op_p50_err_pct']:.2f}%{flag}")
json.dump(results, open(dst, "w"), indent=1)
print(f"  wrote {dst} ({len(results)} point(s))")
PY
else
    log "step 6/6: skipped (--skip-analyse)"
fi

echo
log "=== summary ==="
log "out-dir      ${OUT_DIR}"
log "points       ${#POINTS[@]} validated"
log "job map      ${JOBS_TSV}"
[[ -z "${SKIP_ANALYSE}" ]] && log "results      ${OUT_DIR}/${RESULTS_JSON}"
log "done."
