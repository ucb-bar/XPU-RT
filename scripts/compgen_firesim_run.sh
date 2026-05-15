#!/usr/bin/env bash
# CompGen FireSim e2e driver.
#
# Pipeline:
#   1. Source chipyard/env.sh (activates riscv-tools + verilator + fpga-tools conda)
#   2. cd sims/firesim && source sourceme-manager.sh --skip-ssh-setup
#   3. ssh-add /scratch2/agustin/firesim (for manager → FPGA-host ssh)
#   4. Python: lower ConvNet, emit bundle, cross-compile bare-metal ELF,
#      stage into workloads/, update config_runtime.yaml
#   5. firesim kill  (ensure no stale sim)
#   6. firesim infrasetup
#   7. firesim runworkload
#   8. firesim kill  (cleanup — required even on success)
#   9. Find latest uartlog in results-workload/ and diff the hex output
#      against tests/fixtures/saturn_opu_convnet/golden_outputs.pt
#
# Monitor a running simulation: ``screen -r fsim0``.
set -euo pipefail

COMPGEN_ROOT="${COMPGEN_ROOT:-/scratch2/agustin/CompGen}"
CHIPYARD_ROOT="${CHIPYARD_ROOT:-/scratch2/agustin/chipyard}"
FIRESIM_SSH_KEY="${FIRESIM_SSH_KEY:-/scratch2/agustin/firesim}"
WORKLOAD_NAME="${WORKLOAD_NAME:-compgen-convnet}"
DEPLOY_DIR="$CHIPYARD_ROOT/sims/firesim/deploy"

echo "==> [1/9] source chipyard env"
# shellcheck disable=SC1091
source "$CHIPYARD_ROOT/env.sh"

echo "==> [2/9] source firesim sourceme-manager.sh --skip-ssh-setup"
cd "$CHIPYARD_ROOT/sims/firesim"
# shellcheck disable=SC1091
source ./sourceme-manager.sh --skip-ssh-setup

echo "==> [3/9] ssh-add $FIRESIM_SSH_KEY"
# ssh-agent is set up by sourceme-manager.sh; add the manager→host key.
# If the agent is already warm and the key is loaded, this is a no-op.
ssh-add "$FIRESIM_SSH_KEY" 2>&1 || echo "  (ssh-add skipped; continuing)"

echo "==> [4/9] stage workload (python: lower + cross-compile + config_runtime update)"
cd "$COMPGEN_ROOT"
PYTHONPATH=python uv run python -c "
from pathlib import Path
from compgen.extensions.firesim import build_firesim_workload
w = build_firesim_workload(
    model_fixture_module='tests.fixtures.saturn_opu_convnet.model',
    chipyard_root=Path('$CHIPYARD_ROOT'),
    workload_name='$WORKLOAD_NAME',
    golden_input_path=Path('tests/fixtures/saturn_opu_convnet/golden_inputs.pt'),
)
print(f'  bundle   : {w.bundle_dir}')
print(f'  elf      : {w.elf_path}')
print(f'  workload : {w.workload_dir}')
print(f'  json     : {w.workload_json}')
"
ls -la "$DEPLOY_DIR/workloads/$WORKLOAD_NAME"

echo "==> [5/9] firesim kill (cleanup any stale sim)"
cd "$DEPLOY_DIR"
firesim kill || true

echo "==> [6/9] firesim infrasetup"
firesim infrasetup

echo "==> [7/9] firesim runworkload  (monitor with: screen -r fsim0)"
firesim runworkload || true

echo "==> [8/9] firesim kill (cleanup)"
firesim kill || true

echo "==> [9/9] collect uartlog + diff vs torch golden"
LATEST_RESULT=$(ls -dt "$DEPLOY_DIR/results-workload/"*"${WORKLOAD_NAME}"* 2>/dev/null | head -1)
if [ -z "$LATEST_RESULT" ]; then
    echo "  FAIL: no results dir under $DEPLOY_DIR/results-workload/ matching $WORKLOAD_NAME"
    exit 1
fi
echo "  results: $LATEST_RESULT"
UARTLOG=""
for cand in "$LATEST_RESULT/${WORKLOAD_NAME}0/uartlog" "$LATEST_RESULT/uartlog"; do
    [ -f "$cand" ] && UARTLOG="$cand" && break
done
if [ -z "$UARTLOG" ]; then
    UARTLOG=$(find "$LATEST_RESULT" -name uartlog -print -quit 2>/dev/null)
fi
if [ -z "$UARTLOG" ] || [ ! -f "$UARTLOG" ]; then
    echo "  FAIL: uartlog not found under $LATEST_RESULT"
    exit 1
fi
echo "  uartlog: $UARTLOG"
tail -20 "$UARTLOG"

cd "$COMPGEN_ROOT"
PYTHONPATH=python uv run python -c "
import re, sys
import numpy as np
import torch
log = open('$UARTLOG').read()
m = re.search(r'compgen: out_hex=([0-9a-f]+)', log)
if not m:
    print('  FAIL: no out_hex in uartlog')
    sys.exit(1)
got = np.frombuffer(bytes.fromhex(m.group(1)), dtype='<f4')
expected = torch.load('tests/fixtures/saturn_opu_convnet/golden_outputs.pt').detach().cpu().numpy().astype('<f4').ravel()
max_abs = float(np.max(np.abs(got - expected)))
max_rel = float(np.max(np.abs(got - expected) / (np.abs(expected) + 1e-6)))
ok = np.allclose(got, expected, rtol=1e-3, atol=1e-3)
print(f'  torch   : {expected}')
print(f'  firesim : {got}')
print(f'  max_abs={max_abs:.3e} max_rel={max_rel:.3e} match={ok}')
sys.exit(0 if ok else 1)
"
echo "==> done (FireSim ConvNet run matched torch within float32 tolerance)"
