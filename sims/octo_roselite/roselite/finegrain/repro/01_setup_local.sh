#!/usr/bin/env bash
# usage: 01_setup_local.sh [--check-only]
#
# Build (or verify) the SIMPLER + Octo environment this study runs in, on a BARE
# machine.  Every step below is lifted verbatim from the two scripts that built
# the AWS workers -- roselite/g5wide/bootstrap.sh and roselite/g5/setup_simpler_g5.sh
# -- with the hardcoded /home/ubuntu paths turned into variables.  Nothing here
# is new; if this and those two ever disagree, THOSE are the reference.
#
#   ROOT    where SimplerEnv/ and octo/ are cloned      (default $HOME/simpler)
#   CONDA   miniforge prefix                            (default /scratch2/dima/miniforge3)
#   ENVNAME conda env name                              (default octo_sim)
#
# Time: ~40 min from bare (conda + wheels + two git clones + the asset packs).
# Re-running when everything is present: ~1 min.  Idempotent.
#
# Produces: conda env $ENVNAME with simpler_env, mani_skill2_real2sim and octo
# importable, and the ManiSkill2 real2sim asset packs downloaded.
#
# It worked if `01_setup_local.sh --check-only` ends with "SETUP OK".

set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/common.sh"

ROOT="${ROOT:-$HOME/simpler}"
CONDA="${CONDA_ROOT}"
CHECK_ONLY=0
[ "${1:-}" = "--check-only" ] && CHECK_ONLY=1
[ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ] && { sed -n '2,20p' "$0"; exit 0; }

# Pinned revisions -- identical to g5/setup_simpler_g5.sh.
SIMPLER_SHA=06accaca93535902d408da4855f21cece12bceb7
MANISKILL_SHA=ef7a4d4fdf4b69f2c2154db5b15b9ac8dfe10682
OCTO_SHA=241fb3514b7c40957a86d869fecb7c7fc353f540
MINIFORGE_VER=24.11.3-2

fail=0
say() { printf '%s\n' "$*"; }
chk() { if [ "$1" = ok ]; then say "  PASS  $2"; else say "  FAIL  $2"; fail=1; fi; }

# ---------------------------------------------------------------- 0. conda
# TRAP: the AWS Deep Learning AMI ships NO conda at all.  g5wide/bootstrap.sh
# installs Miniforge3 $MINIFORGE_VER before setup can run.  On a non-AWS box you
# may already have one; this only installs if $CONDA/bin/conda is missing.
if [ ! -x "$CONDA/bin/conda" ]; then
  if [ "$CHECK_ONLY" = 1 ]; then chk bad "conda at $CONDA (absent)"; else
    say "--- installing Miniforge3 $MINIFORGE_VER at $CONDA ---"
    curl -fsSL -o /tmp/miniforge.sh \
      "https://github.com/conda-forge/miniforge/releases/download/${MINIFORGE_VER}/Miniforge3-${MINIFORGE_VER}-Linux-x86_64.sh" || exit 1
    bash /tmp/miniforge.sh -b -p "$CONDA" || exit 1
  fi
else
  chk ok "conda at $CONDA ($("$CONDA/bin/conda" --version))"
fi

if [ "$CHECK_ONLY" = 0 ]; then
  # ------------------------------------------------------------ 1. system
  # TRAP: libvulkan1 (the Vulkan LOADER) is not in the DLAMI.  The driver and
  # /etc/vulkan/icd.d/nvidia_icd.json are, but SAPIEN needs the loader.
  if ! dpkg -s libvulkan1 >/dev/null 2>&1; then
    say "--- installing libvulkan1 (needs sudo) ---"
    sudo apt-get update -qq && \
    sudo apt-get install -y --no-install-recommends libvulkan1 vulkan-tools || {
      say "  apt failed.  On the AWS DLAMI the mirror us-east-1.ec2.archive.ubuntu.com"
      say "  has served HTTP 503 at ~5 kB/s; g5/setup_simpler_g5.sh substitutes"
      say "  mirror.math.princeton.edu.  See that script for the sed one-liner."; exit 1; }
  fi

  # ------------------------------------------------------------ 2. repos
  mkdir -p "$ROOT"
  [ -d "$ROOT/SimplerEnv" ] || git clone https://github.com/simpler-env/SimplerEnv.git "$ROOT/SimplerEnv"
  ( cd "$ROOT/SimplerEnv" && git fetch --quiet origin && git checkout --quiet "$SIMPLER_SHA" \
    && git submodule update --init --recursive \
    && cd ManiSkill2_real2sim && git checkout --quiet "$MANISKILL_SHA" ) || exit 1
  [ -d "$ROOT/octo" ] || git clone https://github.com/octo-models/octo.git "$ROOT/octo"
  ( cd "$ROOT/octo" && git fetch --quiet origin && git checkout --quiet "$OCTO_SHA" ) || exit 1

  # ------------------------------------------------------------ 3. env + wheels
  source "$CONDA/etc/profile.d/conda.sh"
  conda env list | grep -qE "^$ENVNAME[[:space:]]" || conda create -y -n "$ENVNAME" python=3.10
  conda activate "$ENVNAME"
  pip install --upgrade pip
  # requirements_g5.txt is a pip freeze of the validated env with the three
  # editables stripped; installing it first pins every transitive dependency.
  pip install -r "$ROSE/g5/requirements_g5.txt" \
      -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html || exit 1

  # THE TRAP THAT KILLS EVERY FRESH BOX: setuptools >= 81 REMOVES pkg_resources,
  # which sapien 2.2.2 imports at module scope (sapien/core/renderer_config.py).
  # requirements_g5.txt does not pin setuptools and `conda create python=3.10`
  # now seeds 84.0.0, so `import simpler_env` dies with
  #   ModuleNotFoundError: No module named 'pkg_resources'
  pip install "setuptools==80.10.2" || exit 1

  # ------------------------------------------------------------ 4. editables
  pip install --no-deps -e "$ROOT/SimplerEnv/ManiSkill2_real2sim"
  pip install --no-deps -e "$ROOT/SimplerEnv"
  pip install --no-deps -e "$ROOT/octo"

  # ------------------------------------------------------------ 5. assets
  ( cd "$ROOT/SimplerEnv/ManiSkill2_real2sim" && \
    python -m mani_skill2_real2sim.utils.download_asset bridge_v2_real2sim -y || true && \
    python -m mani_skill2_real2sim.utils.download_asset ycb -y || true )
fi

# ---------------------------------------------------------------- verify
octo_env 2>/dev/null || { chk bad "conda activate $ENVNAME"; say "SETUP FAILED"; exit 1; }
chk ok "conda env $ENVNAME active ($(python -V 2>&1))"

v=$(python -c 'import setuptools; print(setuptools.__version__)' 2>/dev/null)
case "$v" in
  80.10.2) chk ok "setuptools $v (pinned; >=81 removes pkg_resources and kills sapien)" ;;
  "")      chk bad "setuptools not importable" ;;
  *)       chk bad "setuptools $v -- MUST be 80.10.2, run: pip install setuptools==80.10.2" ;;
esac

python - <<'PY' || fail=1
import sys
try:
    import pkg_resources                     # noqa: F401  the actual failure point
    import simpler_env, mani_skill2_real2sim, octo, jax
    print(f"  PASS  imports ok (jax {jax.__version__}, devices {jax.devices()})")
except Exception as e:
    print(f"  FAIL  import: {type(e).__name__}: {e}"); sys.exit(1)
PY

test -f /etc/vulkan/icd.d/nvidia_icd.json \
  && chk ok "NVIDIA Vulkan ICD present" || chk bad "no /etc/vulkan/icd.d/nvidia_icd.json"

[ "$fail" -eq 0 ] && say "SETUP OK" || say "SETUP FAILED"
exit "$fail"
