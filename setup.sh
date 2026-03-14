#!/bin/bash
set -e

# 1) Build host compiler tools
export WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export TOOLCHAIN_DIR="${WORKSPACE_DIR}./merlin/build_tools/riscv-tools-spacemit"

python ${WORKSPACE_DIR}/merlin/tools/build.py --target host --config release

# 2) Set up the SpacemiT RISC-V toolchain:


mkdir -p "${TOOLCHAIN_DIR}"

echo "Downloading SpacemiT Toolchain v1.1.2..."
wget https://archive.spacemit.com/merlin/toolchain/spacemit-toolchain-linux-glibc-x86_64-v1.1.2.tar.xz -P "${WORKSPACE_DIR}"

echo "Extracting..."
tar -xvf "${WORKSPACE_DIR}/spacemit-toolchain-linux-glibc-x86_64-v1.1.2.tar.xz" -C "${TOOLCHAIN_DIR}"

echo "Cleaning up archive..."
rm "${WORKSPACE_DIR}/spacemit-toolchain-linux-glibc-x86_64-v1.1.2.tar.xz"

echo "Done. Toolchain installed at ${TOOLCHAIN_DIR}/spacemit-toolchain-linux-glibc-x86_64-v1.1.2"


# 3) Build target runtime for spacemit_x60
python3 ${WORKSPACE_DIR}/merlin/tools/build.py --target spacemit --config perf --with-plugin --clean