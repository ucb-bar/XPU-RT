#!/usr/bin/env bash
# Reproduce the Saturn OPU → Zephyr → Spike end-to-end run with REAL
# ConvNet compute.
#
# The Python side (compgen.runtime.embedded.e2e_spike) produces every
# source file deterministically: the CompGen lowered forward, the
# foundational runtime copy, the model_blob, and the Zephyr main.c
# with the golden input baked in. This shell only handles toolchain
# env vars (Zephyr SDK, west, spike) and the numerical diff against
# golden_outputs.pt at the end.
set -euo pipefail

COMPGEN_ROOT="${COMPGEN_ROOT:-/scratch2/agustin/CompGen}"
ZEPHYR_CHIPYARD_SW="${ZEPHYR_CHIPYARD_SW:-/scratch2/agustin/zephyr-chipyard-sw}"
DIMA_TREE="${DIMA_TREE:-/scratch2/dima/testing/zephyr-chipyard-sw-torch-dryrun-2}"
MERLIN_CONDA="${MERLIN_CONDA:-/scratch2/agustin/miniforge3}"
BUNDLE_DIR="${BUNDLE_DIR:-/tmp/compgen_bundle}"
BUILD_DIR="${BUILD_DIR:-/tmp/compgen_zephyr_build}"

SDK="$DIMA_TREE/tools-manual/zephyr-sdk-1.0.0-beta1"
ZEPHYR_GCC="$SDK/gnu/riscv64-zephyr-elf/bin/riscv64-zephyr-elf-gcc"
ZEPHYR_AR="$SDK/gnu/riscv64-zephyr-elf/bin/riscv64-zephyr-elf-ar"
ZEPHYR_CONDA="$DIMA_TREE/tools/miniforge3/envs/zephyr/bin"
SPIKE="$ZEPHYR_CONDA/spike"
WEST="$ZEPHYR_CONDA/west"

echo "==> [1/5] lower ConvNet + emit bundle + Zephyr overlay (Python)"
cd "$COMPGEN_ROOT"
PYTHONPATH=python uv run python -c "
from pathlib import Path
from compgen.runtime.embedded.e2e_spike import build_e2e_artifacts
a = build_e2e_artifacts(
    model_fixture_module='tests.fixtures.saturn_opu_convnet.model',
    bundle_dir=Path('$BUNDLE_DIR'),
    zephyr_root=Path('$ZEPHYR_CHIPYARD_SW'),
    sample_name='compgen_convnet',
    golden_input_path=Path('tests/fixtures/saturn_opu_convnet/golden_inputs.pt'),
)
print(f'  params={a.lowered.num_params} arena={a.lowered.arena_bytes} ops={a.lowered.op_counts}')
print(f'  bundle={a.bundle_dir}  overlay={a.zephyr_sample_dir}')
"

echo "==> [2/5] cross-compile libcompgen_model.a (rv64gc, lp64d)"
CFLAGS_CROSS="-O2 -ffreestanding -fno-builtin -mabi=lp64d -march=rv64imafdc_zicsr_zifencei -mcmodel=medany -I$BUNDLE_DIR"
rm -f "$BUNDLE_DIR"/*.o "$BUNDLE_DIR/libcompgen_model.a"
OBJS=()
for src in \
    "$BUNDLE_DIR/arena.c" \
    "$BUNDLE_DIR/ops.c" \
    "$BUNDLE_DIR/compgen_model_forward.c" \
    "$BUNDLE_DIR/compgen_model.c" \
    "$BUNDLE_DIR/model_blob.c"; do
    obj="${src%.c}.o"
    "$ZEPHYR_GCC" $CFLAGS_CROSS -c "$src" -o "$obj"
    OBJS+=("$obj")
done
"$ZEPHYR_AR" rcs "$BUNDLE_DIR/libcompgen_model.a" "${OBJS[@]}"
cp "$BUNDLE_DIR/libcompgen_model.a" "$ZEPHYR_CHIPYARD_SW/samples/compgen_convnet/libcompgen_model.a"
ls -la "$BUNDLE_DIR/libcompgen_model.a"

echo "==> [3/5] west build -b spike_riscv64"
source "$MERLIN_CONDA/etc/profile.d/conda.sh" && conda activate merlin-dev
unset CFLAGS CXXFLAGS CPPFLAGS LDFLAGS DEBUG_CFLAGS DEBUG_CXXFLAGS DEBUG_CPPFLAGS \
      CONDA_BUILD_SYSROOT CC_FOR_BUILD CXX_FOR_BUILD
export ZEPHYR_BASE="$DIMA_TREE/zephyr_ws/zephyr"
export ZEPHYR_SDK_INSTALL_DIR="$SDK"
export ZEPHYR_TOOLCHAIN_VARIANT=zephyr
CLEAN_PATH=""
for p in $(echo "$PATH" | tr ":" "\n"); do
    if [[ "$p" != *Vitis* && "$p" != *Vivado* ]]; then CLEAN_PATH="$CLEAN_PATH:$p"; fi
done
export PATH="/usr/bin:$ZEPHYR_CONDA${CLEAN_PATH}"
hash -r
cd "$DIMA_TREE/zephyr_ws"
rm -rf "$BUILD_DIR"
EXTRA_LD="-mabi=lp64d -march=rv64imafdc_zicsr_zifencei -mcmodel=medany \
-fno-tree-vectorize -fno-tree-loop-vectorize -fno-tree-slp-vectorize \
-nostdlib -static -fuse-ld=bfd \
-Wl,--gc-sections -Wl,--build-id=none \
-Wl,--sort-common=descending -Wl,--sort-section=alignment \
-Wl,-X -Wl,-N"
"$WEST" build -p -b spike_riscv64 -d "$BUILD_DIR" \
    "$ZEPHYR_CHIPYARD_SW/samples/compgen_convnet" \
    -- -DEXTRA_LDFLAGS="$EXTRA_LD" 2>&1 | tail -5
ls -la "$BUILD_DIR/zephyr/zephyr.elf"

echo "==> [4/5] run ConvNet under Spike (rv64gc)"
SPIKE_LOG=/tmp/spike_output.log
timeout 1800 "$SPIKE" --isa=rv64gc "$BUILD_DIR/zephyr/zephyr.elf" > "$SPIKE_LOG" 2>&1 || true
tail -20 "$SPIKE_LOG"

echo "==> [5/5] numerical diff vs golden_outputs.pt"
cd "$COMPGEN_ROOT"
PYTHONPATH=python uv run python -c "
import re, sys
import numpy as np
import torch

log = open('$SPIKE_LOG').read()
m = re.search(r'compgen: out_hex=([0-9a-f]+)', log)
if not m:
    print('  FAIL: no out_hex in log')
    sys.exit(1)
raw = bytes.fromhex(m.group(1))
got = np.frombuffer(raw, dtype='<f4')
expected = torch.load('tests/fixtures/saturn_opu_convnet/golden_outputs.pt').detach().cpu().numpy().astype('<f4').ravel()
max_abs = float(np.max(np.abs(got - expected)))
max_rel = float(np.max(np.abs(got - expected) / (np.abs(expected) + 1e-6)))
ok = np.allclose(got, expected, rtol=1e-3, atol=1e-3)
print(f'  torch : {expected}')
print(f'  spike : {got}')
print(f'  max_abs={max_abs:.3e} max_rel={max_rel:.3e} match={ok}')
sys.exit(0 if ok else 1)
"
echo "==> done (Spike ConvNet inference matched torch within float32 tolerance)"
