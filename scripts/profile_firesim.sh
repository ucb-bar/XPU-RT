#!/usr/bin/env bash
# profile_firesim.sh — capture per-op profile data for one (network, backend)
# on FireSim and wire it into the schedule profile-loader path.
#
# What it does:
#   1. Builds the multi_demo harness with the right per-bitstream overlay
#      (firesim_chipyard_dual_gemmini.conf when BACKEND=gemmini OR when
#      MODEL routes to the dual-rocket-saturn-gemmini bitstream — that's
#      both kinds for this generation of work). For pure-RVV builds on
#      the quad-rocket-saturn bitstream, FIRESIM_CONF=firesim_chipyard.conf
#      can be passed explicitly.
#   2. Runs the multi_demo harness on FireSim with profile-CSV emission
#      enabled, single-hart pool (matches the xpurt runtime's per-kind
#      NULL-pool path), int8 quant.
#   3. Output CSV lands at
#        zephyr-chipyard-sw/gen/profile/sweep_v8/<HW>/<TARGET>/<network>/<basename>/<input_tag>/<topo>/results.csv
#      where HW="RVV" for backend=rvv, "gemmini" for backend=gemmini.
#   4. Auto-creates the repo-root profile-tree symlink at
#        gen/profile/<HW>/<TARGET>/<network>/<basename>
#      so profile_loader (which reads under <repo_root>/gen/profile)
#      finds it without manual ln -s.
#
# Usage:
#   bash scripts/profile_firesim.sh <network> <backend> [topo]
#
#   <network>  e.g. dronet, yolov8_nano, mobilenet_v2
#   <backend>  rvv | gemmini   (scalar variant exists but isn't routed
#                              through this script — use multi_demo
#                              directly with TARGET=scalar if needed)
#   [topo]     optional: 0 (single hart, default) or 0,1 (dual hart).
#              Selects PROFILE_CORES which drives the topo_<list>
#              subdir naming consumed by profile_loader's topo_tag.
#
# Env overrides:
#   FIRESIM_CONF      override the per-bitstream overlay
#                     (default: firesim_chipyard_dual_gemmini.conf for
#                     backend=gemmini OR rvv on dual-rocket-saturn-gemmini;
#                     firesim_chipyard.conf otherwise).
#   PROFILE_OUT_ROOT  default: zephyr-chipyard-sw/gen/profile/sweep_v8
#   PROFILE_CPU       default: firesim_rocket_saturn (the profile-target
#                     name baked into the schedule input JSON's
#                     hardware.profile.target field).
#   FORCE_REGEN       default 0 — reuse existing model artifacts so the
#                     profile is captured against the EXACT kernel
#                     binary the xpurt run will link against.
#   FIRESIM_TIMEOUT   default 2400 — yolov8 RVV takes ~999M cycles
#                     (~150-300s wall under FireSim's FMR ≈ 0.01).
#
# Examples:
#   # Re-profile yolov8 on RVV (single hart) — the missing piece that
#   # let `rng.uniform(2,10)` synthetic times slip into schedules for
#   # ~5 months. Lands data + repo-root symlink in one go:
#   bash scripts/profile_firesim.sh yolov8_nano rvv
#
#   # Capture the dual-hart variant (topo_0_1) for future RVV-pool=2 work:
#   bash scripts/profile_firesim.sh dronet rvv 0,1
#
# Pre-reqs:
#   conda activate zephyr  (build env; firesim_runner uses xdma drivers)
#   source zephyr-chipyard-sw/scripts/set_envvars_sdk.sh

set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <network> <backend> [topo]" >&2
    echo "  <backend>  rvv | gemmini" >&2
    echo "  [topo]     0 (default) or 0,1" >&2
    exit 1
fi

NETWORK="$1"
BACKEND="$2"
TOPO="${3:-0}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SUBMODULE="${REPO_ROOT}/zephyr-chipyard-sw"

case "${BACKEND}" in
    rvv|RVV)
        TARGET=rvv
        HW=RVV
        ;;
    gemmini)
        TARGET=gemmini
        HW=gemmini
        ;;
    *)
        echo "ERROR: backend must be 'rvv' or 'gemmini', got '${BACKEND}'" >&2
        exit 1
        ;;
esac

# Default to the dual-rocket-saturn-gemmini overlay — both gemmini and
# RVV builds run on it, and that's where every recent profile was
# captured. Override with FIRESIM_CONF if profiling against a different
# bitstream.
DEFAULT_FS_CONF="firesim_chipyard_dual_gemmini.conf"
FIRESIM_CONF="${FIRESIM_CONF:-${DEFAULT_FS_CONF}}"
PROFILE_OUT_ROOT="${PROFILE_OUT_ROOT:-${SUBMODULE}/gen/profile/sweep_v8}"
PROFILE_CPU="${PROFILE_CPU:-firesim_rocket_saturn}"
FORCE_REGEN="${FORCE_REGEN:-0}"
FIRESIM_TIMEOUT="${FIRESIM_TIMEOUT:-2400}"
QUANT=int8
BASENAME="${NETWORK}.${QUANT}"

echo "[profile_firesim] network=${NETWORK} backend=${BACKEND} topo=${TOPO}"
echo "[profile_firesim] FIRESIM_CONF=${FIRESIM_CONF}"
echo "[profile_firesim] PROFILE_OUT_ROOT=${PROFILE_OUT_ROOT}"

# 1) Run multi_demo's profile capture.
cd "${SUBMODULE}"
TARGET="${TARGET}" \
MODELS="${NETWORK}" \
QUANT="${QUANT}" \
RUNNER=firesim \
AGENTS_POOL_THREADS=1 \
FIRESIM_CONF="${FIRESIM_CONF}" \
PROFILE_OUT_ROOT="${PROFILE_OUT_ROOT}" \
PROFILE_SOURCE=firesim \
PROFILE_BACKEND="${HW}" \
PROFILE_CPU="${PROFILE_CPU}" \
PROFILE_CORES="${TOPO}" \
FORCE_REGEN="${FORCE_REGEN}" \
FIRESIM_TIMEOUT="${FIRESIM_TIMEOUT}" \
    bash agents/examples/multi_demo/run.sh

# 2) Verify CSV landed.
TOPO_DIR="topo_${TOPO//,/_}"
INPUT_TAG="${NETWORK}_${PROFILE_CPU}_${HW}_${BASENAME}"
SUBMODULE_CSV="${PROFILE_OUT_ROOT}/${HW}/${PROFILE_CPU}/${NETWORK}/${BASENAME}/${INPUT_TAG}/${TOPO_DIR}/results.csv"
if [[ ! -f "${SUBMODULE_CSV}" ]]; then
    echo "ERROR: expected profile CSV not found at ${SUBMODULE_CSV}" >&2
    exit 1
fi
echo "[profile_firesim] CSV: ${SUBMODULE_CSV}"

# 3) Wire the repo-root symlink. profile_loader reads
#    <repo_root>/gen/profile/<HW>/<TARGET>/<network>/<basename>/...
# but the new CSV lives under zephyr-chipyard-sw/gen/profile/sweep_v8/...
# so the repo-root path must be a symlink into the submodule's tree.
REPO_PROFILE_DIR="${REPO_ROOT}/gen/profile/${HW}/${PROFILE_CPU}/${NETWORK}"
mkdir -p "${REPO_PROFILE_DIR}"
LINK_TARGET="../../../../../zephyr-chipyard-sw/gen/profile/sweep_v8/${HW}/${PROFILE_CPU}/${NETWORK}/${BASENAME}"
if [[ ! -L "${REPO_PROFILE_DIR}/${BASENAME}" || "$(readlink "${REPO_PROFILE_DIR}/${BASENAME}")" != "${LINK_TARGET}" ]]; then
    ln -sfn "${LINK_TARGET}" "${REPO_PROFILE_DIR}/${BASENAME}"
    echo "[profile_firesim] symlink: ${REPO_PROFILE_DIR}/${BASENAME} -> ${LINK_TARGET}"
else
    echo "[profile_firesim] symlink already in place"
fi

# 4) Sanity check: profile_loader-visible CSV resolves through the symlink.
RESOLVED_CSV="${REPO_PROFILE_DIR}/${BASENAME}/${INPUT_TAG}/${TOPO_DIR}/results.csv"
if [[ -f "${RESOLVED_CSV}" ]]; then
    n_rows=$(($(wc -l < "${RESOLVED_CSV}") - 1))   # minus header
    echo "[profile_firesim] OK — repo-root sees ${n_rows} dispatches at ${RESOLVED_CSV}"
else
    echo "WARN: symlink in place but resolved path doesn't exist? ${RESOLVED_CSV}" >&2
fi
