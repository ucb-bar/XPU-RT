#!/usr/bin/env bash
# Phases 2 and 3, host only: generate the 11 x 4 tasksets from the frozen cost
# model, emit the per-(network, backend) artifacts from it, solve every cell
# with all twelve entries, and pick the feasible-first winner per cell.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
REPO=$(cd ../../../.. && pwd)
export XPURT_CODE_ROOT="$REPO" XPURT_DATA_ROOT="$REPO"
export XPURT_CPSAT_PYTHON="${XPURT_CPSAT_PYTHON:-$REPO/.cpsat-venv/bin/python}"

echo "=== phase 2: tasksets ==="
python3 scripts/mk_workloads_qrb5165.py --emit
echo "=== phase 2b: flow c specs + artifacts from the frozen cost model ==="
python3 scripts/phase2_specs.py --write --artifacts
echo "=== phase 3: 12 solver entries over every cell ==="
mkdir -p raw/out
python3 -u scripts/sweep10_dispatch.py \
    --data-root "$REPO" --code-root "$REPO" \
    --cpsat-python "$XPURT_CPSAT_PYTHON" \
    --outdir raw/out --arms s10port \
    --cheap-workers 32 --cpsat-parallel 6 --cpsat-workers 8 --cpsat-time 60
echo "=== phase 3: tables ==="
python3 scripts/sweep10_analyze.py --outdir raw/out --dest results
cp results/all_results.json results/phase3_all_results.json
python3 scripts/sweep10_breakdown.py results
python3 fpga/pick_winners.py --results results/phase3_all_results.json \
                            --out results/winners.json
echo "=== PHASE 3 COMPLETE ==="
