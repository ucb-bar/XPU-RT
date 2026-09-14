#!/usr/bin/env bash
# ============================================================================
# SIMPLER-env setup on the AWS g5.xlarge (A10G, Ubuntu 22.04 DLAMI).
#
# Replicates the local reference env at /scratch2/dima/misc_sw/octo_work/
# (conda env `octo_sim`) exactly: same repo commits, same pinned wheels.
#
# Idempotent: safe to re-run. Installs into conda env `octo_sim` under
# /home/ubuntu/miniforge3/envs. Does NOT touch any other env on the box.
#
# Traps this script works around (all previously discovered on this AMI):
#   * libvulkan1 (the Vulkan *loader*) is not in the DLAMI. The driver and
#     /etc/vulkan/icd.d/nvidia_icd.json are, but SAPIEN needs the loader.
#   * The AMI apt mirror (us-east-1.ec2.archive.ubuntu.com) has served HTTP
#     503 at ~5 kB/s. mirror.math.princeton.edu is a working substitute.
#   * SAPIEN picks the llvmpipe CPU device unless VK_ICD_FILENAMES forces
#     the NVIDIA ICD -> ErrorExtensionNotPresent.
#   * JAX preallocates ~75% of the GPU unless XLA_PYTHON_CLIENT_PREALLOCATE=false.
#
# Pins that must NOT be "upgraded":
#   cuDNN 8.9.7.29 (jaxlib 0.4.20 is built against cuDNN 8.x, not 9.x),
#   scipy 1.11.4, tensorflow-metadata 1.14.0, tensorflow 2.15.0, jax 0.4.20,
#   and the dlimp git dependency.
# ============================================================================
set -euo pipefail

ROOT=/home/ubuntu/simpler
CONDA=/home/ubuntu/miniforge3
ENVNAME=octo_sim
SIMPLER_SHA=06accaca93535902d408da4855f21cece12bceb7
MANISKILL_SHA=ef7a4d4fdf4b69f2c2154db5b15b9ac8dfe10682
OCTO_SHA=241fb3514b7c40957a86d869fecb7c7fc353f540

mkdir -p "$ROOT/logs"
exec > >(tee -a "$ROOT/logs/setup.log") 2>&1
echo "=== SIMPLER g5 setup start $(date -Is) ==="

# ---------------------------------------------------------------- 1. system
if ! dpkg -s libvulkan1 >/dev/null 2>&1; then
  echo "--- installing libvulkan1 (missing from the DLAMI) ---"
  sudo sed -i 's|http://us-east-1.ec2.archive.ubuntu.com/ubuntu|http://mirror.math.princeton.edu/pub/ubuntu|g' \
      /etc/apt/sources.list
  sudo apt-get update -qq
  sudo apt-get install -y --no-install-recommends libvulkan1 vulkan-tools
else
  echo "--- libvulkan1 already present: $(dpkg-query -W -f='${Version}' libvulkan1) ---"
fi
test -f /etc/vulkan/icd.d/nvidia_icd.json || { echo "FATAL: no NVIDIA Vulkan ICD"; exit 1; }

# ---------------------------------------------------------------- 2. repos
cd "$ROOT"
[ -d SimplerEnv ] || git clone https://github.com/simpler-env/SimplerEnv.git
cd SimplerEnv
git fetch --quiet origin && git checkout --quiet "$SIMPLER_SHA"
git submodule update --init --recursive
cd ManiSkill2_real2sim && git checkout --quiet "$MANISKILL_SHA" && cd ..

cd "$ROOT"
[ -d octo ] || git clone https://github.com/octo-models/octo.git
cd octo && git fetch --quiet origin && git checkout --quiet "$OCTO_SHA"

# ---------------------------------------------------------------- 3. conda env
source "$CONDA/etc/profile.d/conda.sh"
if ! conda env list | grep -qE "^$ENVNAME\s"; then
  conda create -y -n "$ENVNAME" python=3.10
fi
conda activate "$ENVNAME"
python -V

# ---------------------------------------------------------------- 4. wheels
# requirements_g5.txt is `pip freeze` of the validated local octo_sim env with
# the three editable installs stripped out. Installing it first means every
# transitive pin lands at the reference version; the editables then go in with
# --no-deps so nothing gets renegotiated.
pip install --upgrade pip
pip install -r "$ROOT/requirements_g5.txt" \
    -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# setuptools >= 81 REMOVES pkg_resources, which sapien 2.2.2 imports at module
# scope (sapien/core/renderer_config.py). requirements_g5.txt does not pin
# setuptools and `conda create python=3.10` now seeds 84.0.0, so a fresh box dies
# at `import simpler_env` with "ModuleNotFoundError: No module named
# 'pkg_resources'". The original g5 worker predates setuptools 84 and carries
# 80.10.2, which is why this only bites on newly built boxes.
pip install "setuptools==80.10.2"

# ---------------------------------------------------------------- 5. editables
pip install --no-deps -e "$ROOT/SimplerEnv/ManiSkill2_real2sim"
pip install --no-deps -e "$ROOT/SimplerEnv"
pip install --no-deps -e "$ROOT/octo"

# ---------------------------------------------------------------- 6. assets
# ManiSkill2_real2sim ships its real2sim scene/asset pack separately.
cd "$ROOT/SimplerEnv/ManiSkill2_real2sim"
python -m mani_skill2_real2sim.utils.download_asset bridge_v2_real2sim -y || true
python -m mani_skill2_real2sim.utils.download_asset ycb            -y || true

echo "=== SIMPLER g5 setup done $(date -Is) ==="
