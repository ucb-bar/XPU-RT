#!/usr/bin/env bash
# Every phase of this sweep, in order, as it was actually run.
#
# This is the record of the sequence, not a one-command reproduction --
# reproduce.py is that, and it checks its output against the committed record.
# Phase 1 needs the modelblaster zoo, the qnn-convert docker image, and ~40
# minutes of board time; Phase 4 needs the board for as long as the realised
# matrix takes.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
REPO=$(cd ../../../.. && pwd)

export QNN_BOARD_HOST="${QNN_BOARD_HOST:-root@qrb5165}"
export XPURT_CODE_ROOT="$REPO"
export XPURT_DATA_ROOT="$REPO"
export XPURT_CPSAT_PYTHON="${XPURT_CPSAT_PYTHON:-$REPO/.cpsat-venv/bin/python}"

# ---- Phase 1: build the 16 networks whole and measure them
python3 scripts/phase1_build.py                       # onnx -> dlc -> ctx, per backend
# ---- Phase 1R: remove each measured backend blocker with a graph rewrite and
#      build the variant as a candidate (needs an onnx-capable interpreter)
"${SLICE_PY:-/scratch2/dima/miniforge3/envs/xpurt/bin/python}"     scripts/phase1_rewrite.py
# ---- Phase 1B/1C: candidate manifests, then a cell for every candidate lane
python3 scripts/phase1_bindings.py --candidates --write
python3 scripts/phase1_measure.py --passes 5 --resume  # gap-phase cells, 5 passes
# ---- Phase 1A: adopt per (network, lane) on measured evidence; freeze
python3 scripts/phase1_adopt.py --write
python3 scripts/build_cost_model.py --write
# ---- Phase 1G: the granularity decision, against the break-even rule
"${SLICE_PY:-/scratch2/dima/miniforge3/envs/xpurt/bin/python}"     scripts/granularity.py
python3 scripts/make_ledger.py > results/REWRITE_LEDGER.md

# ---- Phase 2: the 11 families x 4 lane configs, periods from those cells
python3 scripts/mk_workloads_qrb5165.py --emit
python3 scripts/phase2_specs.py --write --artifacts

# ---- Phase 3: solve with all twelve entries
mkdir -p raw/out
python3 scripts/sweep10_dispatch.py \
    --data-root "$REPO" --code-root "$REPO" \
    --cpsat-python "$XPURT_CPSAT_PYTHON" \
    --outdir raw/out --arms s10port \
    --cheap-workers 32 --cpsat-parallel 6 --cpsat-workers 8
python3 scripts/sweep10_analyze.py --outdir raw/out --dest results
cp results/all_results.json results/phase3_all_results.json
python3 scripts/sweep10_breakdown.py results
python3 fpga/pick_winners.py --results results/phase3_all_results.json \
                             --out results/winners.json

# ---- Phase 4: run the schedules on the board
python3 scripts/drive.py plan --tier-a "${TIER_A:?set TIER_A to the comma list of cells to run with ALL twelve solvers}"
python3 scripts/drive.py emit
python3 scripts/drive.py runtime
python3 scripts/drive.py stage
python3 scripts/drive.py run --reps 3
python3 scripts/drive.py results

# ---- Phase 5
python3 scripts/analyse_phase4.py --plots
python3 verify_provenance.py
