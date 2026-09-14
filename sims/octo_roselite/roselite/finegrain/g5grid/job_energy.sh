#!/usr/bin/env bash
# ENERGY + SUCCESS from the SAME rollouts, for the 44-arm plane.
#
# Why a new run rather than pairing the existing plane's success with the existing
# 9-arm energy: the harness is not run-to-run deterministic (~20% of byte-identical
# invocations diverge, see docs/NONDETERMINISM.md), so success from run A and energy
# from run B describe DIFFERENT rollouts. A Pareto chart co-optimising the two is only
# valid if both come from the same episodes. trace_eval.py emits both.
#
# The per-tick arrays are reduced to per-episode SCALARS on the worker and then deleted:
# 13 MB/cell x 3520 runs would be 45 GB raw, and energy needs only two numbers per
# episode. summary.json (success) and energy.json (scalars) are all that survive.
#
# Argument: "TASK:ARM:SEED".
source /home/ubuntu/simpler/sim_eval/roselite/env.sh
cd /home/ubuntu/simpler/sim_eval/roselite/finegrain
IFS=: read -r TASK ARM S <<< "$1"
case "$TASK" in
  spoon)  T=widowx_spoon_on_towel ;;
  egg)    T=widowx_put_eggplant_in_basket ;;
  drawer) T=google_robot_close_drawer ;;
  coke)   T=google_robot_pick_coke_can ;;
  *) echo "bad task $TASK"; exit 2 ;;
esac
LAT=$(awk -v a="$ARM" '$1==a{print $2}' /home/ubuntu/arms_grid.tsv)
PER=$(awk -v a="$ARM" '$1==a{print $3}' /home/ubuntu/arms_grid.tsv)
[ -z "$LAT" ] && { echo "bad arm $ARM"; exit 2; }
OUT=/home/ubuntu/simpler/sim_eval/roselite/finegrain/runs_energy/${TASK}_${ARM}_rng${S}
[ -f "$OUT/energy.json" ] && { echo "skip ${TASK}_${ARM}_rng${S}"; exit 0; }
mkdir -p "$(dirname "$OUT")"
python trace_eval.py --task "$T" --latency-ms "$LAT" --issue-period-ms "$PER" \
       --init-rng "$S" --n 24 --out "$OUT" \
       > /home/ubuntu/simpler/sim_eval/roselite/finegrain/logs/en_${TASK}_${ARM}_rng${S}.log 2>&1
rc=$?
[ $rc -ne 0 ] && { echo "FAIL ${TASK}_${ARM}_rng${S} rc=$rc"; exit $rc; }
python - "$OUT" <<'PY'
import json, sys, numpy as np, pathlib, os
d = pathlib.Path(sys.argv[1])
s = json.load(open(d / "summary.json"))
dt = s["tick_ms"] / 1000.0
JG = {8: slice(0, 6), 11: slice(0, 7)}      # arm joints; fingers/head excluded
out = []
for e in s["episodes"]:
    ep = e["episode_id"]
    qf = np.load(d / f"ep{ep:02d}_qf.npy"); w = np.load(d / f"ep{ep:02d}_qvel.npy")
    a = JG.get(w.shape[1], slice(None))
    out.append(dict(ep=ep, success=bool(e["success"]), ticks=int(e["ticks"]),
                    t2=float((qf ** 2).sum(1).sum() * dt),
                    eff=float(((w ** 2).sum(0) * dt)[a].sum())))
json.dump(dict(task=s["task"], n_success=s["n_success"], n_episodes=s["n_episodes"],
               tick_ms=s["tick_ms"], act_ms=s.get("act_ms"), episodes=out),
          open(d / "energy.json", "w"))
# arrays served their purpose; keep only the two small JSONs
for f in os.listdir(d):
    if f.endswith((".npy", ".png")) or f.endswith("_trace.json"):
        os.remove(d / f)
PY
echo "done ${TASK}_${ARM}_rng${S} : $(python -c "import json;d=json.load(open('$OUT/energy.json'));print(f\"{d['n_success']}/{d['n_episodes']}\")" 2>/dev/null)"
