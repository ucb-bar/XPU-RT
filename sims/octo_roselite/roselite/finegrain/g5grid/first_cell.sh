#!/usr/bin/env bash
# Fetch periodically; report the first (task,arm) cell that reaches all 10 seeds.
set -u; cd "$(dirname "$0")"
while :; do
  bash fetch_grid.sh >/dev/null 2>&1
  out=$(python3 - <<'PY'
import glob, json, collections
from pathlib import Path
c=collections.defaultdict(set); ok=collections.defaultdict(int); tot=collections.defaultdict(int)
for f in glob.glob('runs/*/*/summary.json'):
    n=Path(f).parent.name
    if '_rng' not in n: continue
    h,s=n.rsplit('_rng',1); t,a=h.split('_',1)
    if not a.startswith('g'): continue
    c[(t,a)].add(int(s))
    try: d=json.load(open(f))
    except Exception: continue
    ok[(t,a)]+=d.get('n_success',0); tot[(t,a)]+=d.get('n_episodes',0)
full=[(k,v) for k,v in c.items() if len(v)>=10]
if full:
    for (t,a),v in sorted(full):
        print(f"FIRSTCELL {t} {a} seeds={len(v)} success={ok[(t,a)]}/{tot[(t,a)]} "
              f"({100*ok[(t,a)]/max(tot[(t,a)],1):.1f}%)")
PY
)
  if [ -n "$out" ]; then echo "$out"; echo "at $(date +%H:%M:%S)"; break; fi
  sleep 420
done
