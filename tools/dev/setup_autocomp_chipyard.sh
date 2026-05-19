#!/usr/bin/env bash
# Set up the chipyard checkout autocomp's Gemmini backend expects.
#
# Autocomp's gemmini_setup.md pins a specific chipyard commit + a
# pair of branch heads (gemmini@auto-comp-v2 + libgemmini@auto-comp).
# Its evaluator depends on a modified Spike fork that emits the
# "Generated implementation latency: N cycles" stdout string the
# parser at autocomp/backend/gemmini/gemmini_eval.py:423 reads.
#
# This script is idempotent: skips clone / checkout / build steps
# that have already landed. Run it interactively (chipyard's
# submodule init takes 10+ minutes the first time):
#
#   bash scripts/dev/setup_autocomp_chipyard.sh
#
# After success it writes `.autocomp_env` at the repo root with the
# INT8_16PE_CHIPYARD_PATH export the cross-target driver picks up.

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration — override via env if your layout differs.
# ---------------------------------------------------------------------------

ROOT="${AUTOCOMP_CHIPYARD_ROOT:-/scratch2/agustin/chipyard-autocomp}"
PIN="${AUTOCOMP_CHIPYARD_PIN:-dbc082e2206f787c3aba12b9b171e1704e15b707}"
GEMMINI_BRANCH="${AUTOCOMP_GEMMINI_BRANCH:-auto-comp-v2}"
LIBGEMMINI_BRANCH="${AUTOCOMP_LIBGEMMINI_BRANCH:-auto-comp}"
XPU_RT_REPO="${XPU_RT_REPO:-/scratch2/agustin/xpu-rt-integration}"

log() { printf '[setup_autocomp] %s\n' "$*" >&2; }
die() { log "ERROR: $*"; exit 1; }

# ---------------------------------------------------------------------------
# Step 1 — clone chipyard if missing, checkout the pinned commit.
# ---------------------------------------------------------------------------

if [ ! -d "$ROOT/.git" ]; then
  log "cloning chipyard → $ROOT"
  git clone --filter=blob:none https://github.com/ucb-bar/chipyard "$ROOT"
fi

cd "$ROOT"
current_head="$(git rev-parse HEAD)"
if [ "$current_head" != "$PIN" ]; then
  log "fetching pinned commit $PIN"
  git fetch --depth=1 origin "$PIN" || die "git fetch failed for $PIN"
  git -c advice.detachedHead=false checkout "$PIN"
fi
log "chipyard at $(git rev-parse HEAD)"

# ---------------------------------------------------------------------------
# Step 2 — init the gemmini submodule + check out its auto-comp-v2 branch.
# ---------------------------------------------------------------------------

if [ ! -d "$ROOT/generators/gemmini/.git" ] && [ ! -f "$ROOT/generators/gemmini/.git" ]; then
  log "initialising generators/gemmini submodule"
  git submodule update --init --recursive generators/gemmini
fi

cd "$ROOT/generators/gemmini"
git fetch --depth=1 origin "$GEMMINI_BRANCH" || die "git fetch failed for gemmini $GEMMINI_BRANCH"
git -c advice.detachedHead=false checkout "$GEMMINI_BRANCH"
log "gemmini at branch $GEMMINI_BRANCH ($(git rev-parse --short HEAD))"

# Also init libgemmini submodule under gemmini's tree.
LIBGEMMINI="$ROOT/generators/gemmini/software/libgemmini"
if [ ! -d "$LIBGEMMINI/.git" ] && [ ! -f "$LIBGEMMINI/.git" ]; then
  cd "$ROOT/generators/gemmini"
  git submodule update --init --recursive software/libgemmini || true
fi

if [ ! -d "$LIBGEMMINI" ]; then
  die "libgemmini path not present after submodule init: $LIBGEMMINI"
fi

cd "$LIBGEMMINI"
git fetch --depth=1 origin "$LIBGEMMINI_BRANCH" || die "git fetch failed for libgemmini $LIBGEMMINI_BRANCH"
git -c advice.detachedHead=false checkout "$LIBGEMMINI_BRANCH"
log "libgemmini at branch $LIBGEMMINI_BRANCH ($(git rev-parse --short HEAD))"

# ---------------------------------------------------------------------------
# Step 3 — build the libgemmini Spike fork.
#
# This step requires the riscv-tools conda env on PATH (so Spike's
# build system + headers are visible). If autocomp's own Spike build
# fails, we surface a clean error pointing at the conda env activation
# rather than letting the calibration runner crash later.
# ---------------------------------------------------------------------------

LIBGEMMINI_SENTINEL="$LIBGEMMINI/.libgemmini_built_for_autocomp"
if [ ! -f "$LIBGEMMINI_SENTINEL" ]; then
  log "building libgemmini Spike fork (this requires Spike build deps + write access to its install prefix)"
  cd "$LIBGEMMINI"
  if ! make >/tmp/libgemmini_build.log 2>&1; then
    log "make failed — see /tmp/libgemmini_build.log"
    log "common cause: riscv-tools conda env not activated; try"
    log "  conda activate /scratch2/agustin/chipyard/.conda-env/riscv-tools"
    die "libgemmini build failed"
  fi
  if ! make install >>/tmp/libgemmini_build.log 2>&1; then
    die "libgemmini install failed — see /tmp/libgemmini_build.log"
  fi
  touch "$LIBGEMMINI_SENTINEL"
  log "libgemmini built + installed"
else
  log "libgemmini already built (sentinel $LIBGEMMINI_SENTINEL)"
fi

# ---------------------------------------------------------------------------
# Step 4 — install autocomp into the active python env.
# ---------------------------------------------------------------------------

cd "$XPU_RT_REPO"
if ! python -c "import autocomp" >/dev/null 2>&1; then
  log "installing autocomp (uv pip install -e third_party/autocomp)"
  uv pip install -e third_party/autocomp || die "autocomp install failed"
else
  log "autocomp already importable"
fi

# ---------------------------------------------------------------------------
# Step 5 — write the env-export file the matrix driver consumes.
# ---------------------------------------------------------------------------

cat > "$XPU_RT_REPO/.autocomp_env" <<EOF
# Auto-written by scripts/dev/setup_autocomp_chipyard.sh on $(date -u +%Y-%m-%dT%H:%M:%SZ).
# \`source .autocomp_env\` before running the cross-target matrix driver
# so autocomp's GemminiEvalBackend can find its chipyard checkout.
export INT8_16PE_CHIPYARD_PATH="$ROOT"
export AUTOCOMP_CHIPYARD_ROOT="$ROOT"
EOF
log ".autocomp_env written → source it before running the matrix driver"

# ---------------------------------------------------------------------------
# Step 6 — smoke-import autocomp's GemminiEvalBackend.
# ---------------------------------------------------------------------------

if python <<'PY'
import os, sys
os.environ.setdefault("INT8_16PE_CHIPYARD_PATH", os.environ.get("AUTOCOMP_CHIPYARD_ROOT", ""))
try:
    from autocomp.backend.gemmini.gemmini_eval import GemminiEvalBackend
    from autocomp.hw_config.gemmini_config import GemminiHardwareConfig
    eb = GemminiEvalBackend(GemminiHardwareConfig(pe_dim=16))
    if not eb.gemmini_path.is_dir():
        sys.exit(f"gemmini_path not found: {eb.gemmini_path}")
    print(f"smoke ok: gemmini_path={eb.gemmini_path}")
except Exception as exc:
    sys.exit(f"smoke failed: {type(exc).__name__}: {exc}")
PY
then
  log "smoke test passed"
else
  die "autocomp GemminiEvalBackend smoke failed — inspect the error above"
fi
