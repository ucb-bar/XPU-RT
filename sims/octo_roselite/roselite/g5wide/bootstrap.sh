#!/usr/bin/env bash
# Bring a bare Deep Learning OSS DLAMI (ami-012ba162b9cd2729c) up to the point
# where g5/setup_simpler_g5.sh can run: that script assumes a conda at
# /home/ubuntu/miniforge3, which the AMI does NOT ship (the original g5 worker
# had it installed by hand). Idempotent.
set -euo pipefail
ROOT=/home/ubuntu/simpler
CONDA=/home/ubuntu/miniforge3
mkdir -p "$ROOT/logs"
exec > >(tee -a "$ROOT/logs/bootstrap.log") 2>&1
echo "=== bootstrap start $(date -Is) on $(hostname) ==="

if [ ! -x "$CONDA/bin/conda" ]; then
  echo "--- installing Miniforge3 (absent from this AMI) ---"
  curl -fsSL -o /tmp/miniforge.sh \
    https://github.com/conda-forge/miniforge/releases/download/24.11.3-2/Miniforge3-24.11.3-2-Linux-x86_64.sh
  bash /tmp/miniforge.sh -b -p "$CONDA"
else
  echo "--- miniforge3 already present ---"
fi
"$CONDA/bin/conda" --version

echo "=== bootstrap done, handing off to setup_simpler_g5.sh $(date -Is) ==="
bash "$ROOT/setup_simpler_g5.sh"
echo "=== ALL SETUP DONE $(date -Is) ==="
