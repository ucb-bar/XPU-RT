#!/usr/bin/env bash
# Wait for phase1_build.py to finish, then close out Phase 1:
# manifests from the compose verdicts, five measurement passes over every
# (tile, lane) pair, and the frozen cost model.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
while pgrep -f "[p]hase1_build[.]py" >/dev/null; do sleep 60; done
echo "=== build done ==="
tail -20 logs/phase1_build_run.log
echo "=== bindings ==="
python3 scripts/phase1_bindings.py --write
echo "=== measure (5 passes, resuming) ==="
python3 -u scripts/phase1_measure.py --iters 40 --gap-us 3000 --passes 5 --resume
echo "=== cost model ==="
python3 scripts/build_cost_model.py --write
echo "=== cell table ==="
python3 scripts/cell_table.py --audit --compose
echo "=== PHASE 1 COMPLETE ==="
