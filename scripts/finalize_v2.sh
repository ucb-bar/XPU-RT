#!/usr/bin/env bash
# Run AFTER the strengthening campaign finishes: promote the fresh 240-flight grid to the figure
# source, regenerate the figures + reproduce numbers + master CSV. Does NOT touch the paper repo or
# push (caption-number update + push are done interactively after reviewing the printed numbers).
set -eu
WT="${WT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"   # this repo, wherever it is checked out
. "$WT/scripts/env.sh"                                                        # SIM_TREE, ISAAC_PY, RES
PY="$ISAAC_PY"
V2=$RES/hil_ablation_v2.csv
cd "$WT"

n=$(($(wc -l < "$V2")-1))
echo "=== v2 has $n flights (expect 240 = 5x4x12) ==="
[ "$n" -ge 200 ] || { echo "ABORT: v2 only $n rows — campaign not complete?"; exit 1; }

echo "=== promote v2 -> hil_ablation.csv (v1 already archived as hil_ablation_v1_120flights_fixedgain.csv) ==="
cp -v "$V2" "$RES/hil_ablation.csv"

echo "=== regenerate envelope panel + print the NEW plotted numbers ==="
$PY scripts/reproduce_hil_figure.py --render

echo "=== regenerate the showdown composite (reads the new hil_ablation.csv) ==="
CV=$RES/crash_verify
$PY sims/scripts/showdown_gatecourse.py --xpu-dir "$CV/new_xpu_figdata" --ros-dir "$CV/new_ros50_figdata" \
    --with-envelope --dpi 300 --out "$RES/refined/warehouse_showdown_envelope"

echo "=== refresh master CSV (envelope regime now 240; course-B added if present) ==="
$PY scripts/build_master_csv.py

echo "=== DONE. Review the numbers above, then: update caption figures, copy plots to paper, push. ==="
