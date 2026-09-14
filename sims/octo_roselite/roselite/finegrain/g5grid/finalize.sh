#!/usr/bin/env bash
# Fetch -> verify completeness -> analyse. Run ONLY after all workers report ALLDONE.
# Shutdown is deliberately NOT here: results must be verified locally first.
set -u; cd "$(dirname "$0")"
echo "=== fetch $(date +%H:%M:%S) ==="
bash fetch_grid.sh
echo
echo "=== verify: per-cell completeness ==="
python3 - <<'PY'
import glob, collections
from pathlib import Path
arms={l.split('\t')[0] for l in open('arms.tsv')}
c=collections.defaultdict(set)
for f in glob.glob('runs/*/*/summary.json'):
    n=Path(f).parent.name
    if '_rng' not in n: continue
    h,s=n.rsplit('_rng',1); t,a=h.split('_',1)
    if a in arms: c[(t,a)].add(int(s))
tasks=['egg','spoon','coke']
exp=len(tasks)*len(arms)
short=[(t,a,len(c.get((t,a),()))) for t in tasks for a in sorted(arms) if len(c.get((t,a),()))!=10]
print(f"cells: {len(c)}/{exp}   summaries: {sum(len(v) for v in c.values())}/1170")
if short:
    print(f"INCOMPLETE cells ({len(short)}):")
    for t,a,n in short: print(f"   {t:6s} {a:10s} {n}/10 seeds")
else:
    print("ALL CELLS COMPLETE: 117 cells x 10 seeds x 24 episodes")
PY
echo
echo "=== analysis: per-arm, contrasts, fits ==="
python3 analyze_grid.py > RESULTS_grid.txt 2>&1; echo "wrote RESULTS_grid.txt ($(wc -l < RESULTS_grid.txt) lines)"
echo "=== analysis: merged operating points, noise floor, trends ==="
python3 analyze_merged.py > RESULTS_merged.txt 2>&1; echo "wrote RESULTS_merged.txt ($(wc -l < RESULTS_merged.txt) lines)"
