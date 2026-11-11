#!/usr/bin/env bash
# usage: 08_aws_fetch.sh [--logs]
#          --logs  also pull the per-job stdout logs (they exist ONLY on the boxes)
#
# Pull every summary.json off the workers into g5fine/runs/<worker>/, then check
# the harvest is complete and that the harness that produced it was the same
# everywhere.  Wraps g5fine/fetch.sh, which does the rsync.
#
# WHY --logs MATTERS BEFORE ANY PRUNE.  fetch.sh copies only summary.json.  The
# per-job stdout (per-episode progress, the SUCCESS RATE lines, warnings) lives
# only in /home/ubuntu/simpler/sim_eval/roselite/finegrain/logs/ on each box and
# is destroyed with the root volume when an instance is terminated.
#
# Time: ~30 s (summaries) / a few minutes with --logs.
# Produces: g5fine/runs/w*/<task>_<arm>_rng<seed>/summary.json
#           (and with --logs, g5fine/logs/<worker>/*.log)
# It worked if TOTAL equals (#tasks x #seeds x #arms) and the completeness table
# shows no missing cells.

set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/common.sh"
[ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ] && { sed -n '2,20p' "$0"; exit 0; }
[ -f "$G5FINE/hosts.sh" ] || { echo "no hosts.sh -- 05_aws_instances.sh hosts --write"; exit 1; }
# shellcheck disable=SC1091
source "$G5FINE/hosts.sh"

bash "$G5FINE/fetch.sh"

if [ "${1:-}" = "--logs" ]; then
  for w in $(printf '%s\n' "${!HOSTS[@]}" | sort); do
    h=${HOSTS[$w]}; mkdir -p "$G5FINE/logs/$w"
    rsync -a -e "$SSH" \
      ubuntu@"$h":/home/ubuntu/simpler/sim_eval/roselite/finegrain/logs/ "$G5FINE/logs/$w/" 2>/dev/null \
      && echo "$w logs: $(ls "$G5FINE/logs/$w" | wc -l) files" || echo "$w logs: UNREACHABLE"
  done
fi

octo_env
python - "$G5FINE/runs" <<'PY'
import glob, json, collections, sys
from pathlib import Path
runs = Path(sys.argv[1])
ARMS = ["lat0","pipe110","pipe200","serial283","fp32_555","cpu685"]
cells = collections.defaultdict(dict); md5s = collections.Counter(); bad = []
for f in glob.glob(str(runs / "*" / "*" / "summary.json")):
    name = Path(f).parent.name
    if "_rng" not in name: continue
    head, seed = name.rsplit("_rng", 1); task, arm = head.split("_", 1)
    try: d = json.load(open(f))
    except Exception: bad.append(f); continue
    cells[task].setdefault(int(seed), {})[arm] = d["n_success"]
    # quarantine rule: a google_robot run that did not use --actuation native
    # actuated on the fine grid under a planner-interpolated controller and is
    # rejected by analyze_fine.py / compare_tasks.py.  Flag it here too.
    if d.get("policy_setup") == "google_robot" and d.get("actuation") != "native":
        bad.append(f + "   [google_robot without --actuation native -- QUARANTINE]")
print(f"\ncompleteness ({sum(len(a) for s in cells.values() for a in s.values())} runs):")
for task in sorted(cells):
    seeds = sorted(cells[task]); missing = []
    for s in seeds:
        for a in ARMS:
            if a not in cells[task][s]: missing.append(f"{s}:{a}")
    print(f"  {task:7s} seeds {seeds[0]}-{seeds[-1]} ({len(seeds)})  "
          f"{'COMPLETE' if not missing else 'MISSING ' + ','.join(missing)}")
if bad:
    print("\nPROBLEM FILES:"); [print("  " + b) for b in bad]
PY
echo
echo "Harness md5 must be identical on every worker AND equal to the local file."
echo "  local: $(md5sum "$FINE/finegrain_eval.py" | cut -c1-12)"
for w in $(printf '%s\n' "${!HOSTS[@]}" | sort); do
  echo "  $w:    $(timeout 40 $SSH -n ubuntu@"${HOSTS[$w]}" \
     'md5sum /home/ubuntu/simpler/sim_eval/roselite/finegrain/finegrain_eval.py 2>/dev/null | cut -c1-12' 2>/dev/null)"
done
