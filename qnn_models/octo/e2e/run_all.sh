#!/usr/bin/env bash
# Regenerate every figure that reproduces from committed data.
#
# TIER 1 only: these read the small JSON/TSV in data/ and need no run tree, no GPU and
# no network. Everything else in figures/ needs the 972 MB of raw runs that are NOT in
# this repo -- see docs/REPRODUCE.md for the tiers and how to regenerate them.
set -u
cd "$(dirname "$0")/figures"
FIGS=(fig_e2e_measured fig_e2e_heatmap fig_schmoo fig_energy
      make_backend_row make_backend_icons make_rtos_icon make_rtos_diagram)
fail=0
for f in "${FIGS[@]}"; do
  if [ ! -f "$f.py" ]; then echo "  SKIP  $f (absent)"; continue; fi
  if out=$(python3 "$f.py" 2>&1); then
    echo "  ok    $f"
  else
    echo "  FAIL  $f :: $(echo "$out" | grep -E 'Error|error' | tail -1 | cut -c1-100)"
    fail=$((fail+1))
  fi
done
echo
[ "$fail" -eq 0 ] && echo "all tier-1 figures regenerated" || echo "$fail figure(s) failed"
exit $fail
