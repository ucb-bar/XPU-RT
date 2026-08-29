#!/usr/bin/env bash
# Installs everything the mlp/dronet/yolo XPU-RT reproduction flow needs
# ON TOP OF a standalone zephyr-chipyard-sw install -- nothing here touches
# zephyr-chipyard-sw's own install_conda.sh/install_submodules.sh/
# install_toolchain_sdk.sh, and zephyr-chipyard-sw stays fully usable on its
# own (e.g. for consumers that only want the Zephyr/RISC-V dev environment,
# with no ML/scheduling stack at all) without ever running this script.
#
# Run this AFTER activating the `zephyr` conda env produced by
# zephyr-chipyard-sw's own standalone setup (see
# zephyr-chipyard-sw/README.md's "Standalone Installation" section, or
# docs/mlp_dronet_yolo_spike_reproduction.md's Prerequisites section) --
# everything below installs into that SAME env, so one conda env covers
# both `west build` and the scheduling/reproduction flow.
#
# Usage:
#   source <path-to-zephyr-chipyard-sw>/tools/miniforge3/etc/profile.d/conda.sh
#   conda activate zephyr
#   bash scripts/install_xpurt_deps.sh [--milp]
#
# Flags:
#   --milp   also install cvxpy (needed only for --solver milp; the
#            greedy/greedy_periodic solvers used by the reproduction flow
#            don't need it). mosek itself is license-gated and NOT
#            installed by this flag -- see cvxpy's own docs for that.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOP_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ModelBlaster is reachable by two paths, and which ones are CHECKED OUT
# differs per clone: XPU-RT declares it as a top-level submodule, and
# zephyr-chipyard-sw declares it again for the spike/firesim flow. Both
# should name the same commit -- but an uninitialised submodule is an empty
# directory, not an error, so hardcoding either path makes this script fail
# with `pip install -e <empty dir>` on a perfectly good checkout.
#
# Resolve to whichever is actually present, preferring the top-level one
# because it is declared in THIS repo's .gitmodules and is the pointer this
# repo can keep current on its own.
_zcs_path="$(git -C "${TOP_ROOT}" config -f .gitmodules --get submodule.zephyr-chipyard-sw.path || true)"
ZCS_ROOT="${TOP_ROOT}/${_zcs_path:-zephyr-chipyard-sw}"

MB_ROOT=""
for _cand in \
    "${TOP_ROOT}/$(git -C "${TOP_ROOT}" config -f .gitmodules --get submodule.ModelBlaster.path 2>/dev/null || echo ModelBlaster)" \
    "${ZCS_ROOT}/$(git -C "${ZCS_ROOT}" config -f .gitmodules --get submodule.modelblaster.path 2>/dev/null || echo modelblaster)"
do
    if [[ -f "${_cand}/pyproject.toml" ]]; then MB_ROOT="${_cand}"; break; fi
done
if [[ -z "${MB_ROOT}" ]]; then
    echo "ERROR: no ModelBlaster checkout found. Initialise one:" >&2
    echo "  git submodule update --init ModelBlaster" >&2
    echo "  # or, for the spike/firesim flow:" >&2
    echo "  git submodule update --init --recursive zephyr-chipyard-sw" >&2
    exit 1
fi

MILP=""
while (( $# )); do
    case "$1" in
        --milp) MILP=1 ;;
        *) echo "unknown flag: $1" >&2; exit 1 ;;
    esac
    shift
done

log() { echo "[install_xpurt_deps] $*"; }

if [[ -z "${CONDA_PREFIX:-}" ]]; then
    echo "ERROR: no conda env active. Activate the zephyr env first (see" >&2
    echo "  zephyr-chipyard-sw/README.md's Standalone Installation section)" >&2
    exit 1
fi
log "installing into: ${CONDA_PREFIX}"

# 1) xpu-rt's own scheduler deps (numpy/scipy/matplotlib/pandas; cvxpy only
#    with --milp). Declared in pyproject.toml -- see that file's comments
#    for why xpu-rt/ itself isn't installed as an importable package.
log "installing xpu-rt's own deps"
if [[ -n "${MILP}" ]]; then
    pip install -e "${TOP_ROOT}[milp]"
else
    pip install -e "${TOP_ROOT}"
fi

# 2) modelblaster's own deps (torch, ultralytics, pillow, pyyaml, numpy,
#    requests -- see zephyr-chipyard-sw/modelblaster/pyproject.toml).
#    Unrelated to xpu-rt's own deps above; installed separately because
#    modelblaster is its own standalone, independently-versioned package.
log "installing modelblaster's own deps"
pip install -e "${MB_ROOT}"

# 3) The one pinned wheel with no other home -- spike itself. Despite the
#    custom-looking version string, this is a normal public PyPI package
#    (github.com/liuyu81/pyspike). Gemmini/Saturn RoCC support does NOT
#    come from this wheel -- see docs/mlp_dronet_yolo_spike_reproduction.md.
log "installing spike"
pip install spike==0.0.5.dev20

# 4) System package (not pip-installable): ultralytics pulls in
#    opencv-python, which dynamically links libGL.so.1 at import time --
#    missing on a bare/headless install. Best-effort; skip quietly if apt
#    isn't available or the caller has no sudo (already present on most
#    real dev workstations, which is why this went unnoticed for a while).
if command -v apt-get >/dev/null 2>&1; then
    log "installing libgl1 (system package, needed by ultralytics' opencv-python)"
    if [[ "$(id -u)" == "0" ]]; then
        apt-get install -y libgl1
    elif command -v sudo >/dev/null 2>&1; then
        sudo apt-get install -y libgl1
    else
        echo "WARNING: no sudo available -- install libgl1 manually if yolov8_nano's" >&2
        echo "  extraction step fails with 'ImportError: libGL.so.1: ...'" >&2
    fi
else
    log "apt-get not found -- skipping libgl1 (install the equivalent OpenGL runtime package for your distro if yolov8_nano's extraction step fails)"
fi

log "done."
