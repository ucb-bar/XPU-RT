#!/usr/bin/env python3
"""Reduce the 44-arm x 4-task x 20-seed plane to a per-SEED metric cache.

Why per-seed and not per-arm: every claim in PARETO_SEARCH.md rests on a noise estimate,
and the only honest source of noise here is the spread across the 20 initialisation seeds
(the harness is not run-to-run deterministic -- ~20% of byte-identical invocations
diverge, see NONDETERMINISM.md -- so a single seed proves nothing). Keeping the 20 seed
values lets the figures recompute both the standard error and a seed bootstrap of the
Pareto frontier itself.

TWO TRAPS IN THE RUN TREE, both of which corrupt the counts if ignored:

  DUPLICATE FETCH COPIES.  4,822 summary.json exist under g5grid/runs but only 3,520
  are distinct (task, arm, seed) keys -- 983 keys were fetched twice into different
  shard directories. Verified byte-identical (md5), so first-wins deduplication is
  lossless, but counting files instead of keys inflates every n by ~37%.

  STALE ARMS FROM AN OLDER SWEEP.  319 files sit in directories named for the earlier
  9-arm sweep (drawer_lat0, drawer_cpu685, ...), which are NOT of the g<period>_<window>
  form and are NOT part of this plane. The arm regex is anchored to exclude them.

After both, exactly 3,520 runs = 44 arms x 4 tasks x 20 seeds x 24 episodes = 84,480
episodes, with no cell missing and none over-full (asserted below).

SEED SETS ARE NOT UNIFORM on egg and coke: 39 arms ran seeds {100-109,200-209} (egg) /
{120-129,200-209} (coke) while the five g130_275..g130_350 arms ran {90-99,200-209}. All
44 share seeds 200-209. Checked and cleared: restricting those five arms to the shared
ten moves their egg success by -0.4/+4.6/-5.4/+2.9/-1.7 points, i.e. inside the seed
noise, and their neighbours move by as much in both directions. The cache records each
seed so the check can be repeated.
"""
from __future__ import annotations
import json, re, sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
RUNS = next((c for c in (HERE.parent / "g5grid" / "runs", HERE / "runs") if c.is_dir()), None)
OUT = HERE / "pareto_search_cache.json"
# anchored: rejects both the stale 9-arm dirs and any partial name match
KEY = re.compile(r"^(egg|spoon|coke|drawer)_(g\d+_\d+)_rng(\d+)$")


def cell(eps, task):
    """Per-episode fields of one (task, arm, seed) cell -> the scalars the search uses."""
    s = np.array([e["success"] for e in eps], bool)
    sim = np.array([e["sim_ms"] for e in eps], float)
    inf = np.array([e["n_inferences"] for e in eps], float)
    m = {"sr": 100.0 * s.mean(),
         # accelerator invocations per SECOND of mission time. Set by the release
         # cadence alone, so it carries no rollout noise -- all the uncertainty in the
         # compute figures lives on the success axis.
         "inf_rate_hz": float(np.mean(inf / (sim / 1000.0))),
         # median wall of SIM time to finish a WON episode. nan when the cell won none.
         "t_succ_ms": float(np.median(sim[s])) if s.any() else None,
         # accelerator work spent inside one winning episode (cadence x duration)
         "inf_succ": float(np.median(inf[s])) if s.any() else None,
         "n_succ": int(s.sum())}
    st = [e.get("episode_stats") or {} for e in eps]
    if task in ("egg", "spoon"):
        cg = np.array([bool(x.get("consecutive_grasp")) for x in st])
        ot = np.array([bool(x.get("src_on_target")) for x in st])
        m["cgrasp"] = 100.0 * cg.mean()
        m["ontgt"] = 100.0 * ot.mean()
        # conversion: of the episodes that reached a stable grasp, how many placed it
        m["conv_g2t"] = 100.0 * float(ot[cg].mean()) if cg.any() else None
    if task == "coke":
        cg = np.array([bool(x.get("consec_grasp")) for x in st])
        gr = np.array([bool(x.get("grasped")) for x in st])
        m["cgrasp"] = 100.0 * cg.mean()
        m["grasped"] = 100.0 * gr.mean()
        m["conv_g2t"] = 100.0 * float(s[cg].mean()) if cg.any() else None
    if task == "drawer":
        # residual drawer opening in metres. Success is qpos <= 0.05 exactly (checked:
        # every winning episode has qpos <= 0.05 and no losing one does), so qpos gives
        # this task the partial credit the grasp funnel gives the others, and "half
        # closed" at 0.10 is its precursor stage.
        q = np.array([float(x["qpos"]) for x in st])
        m["qpos"] = float(q.mean())
        m["stage"] = 100.0 * float((q <= 0.10).mean())
    # the funnel figure needs a precursor stage that STRICTLY CONTAINS success, so that
    # a point below the diagonal means "reached the stage and did not convert". On the
    # widowx tasks that is consecutive_grasp (a placement implies one). On coke it is
    # `grasped`, NOT consec_grasp: coke success (a significant lift) is reached in
    # 34.6-46.5% of episodes against a consec_grasp rate of only 20.2-30.4%, so
    # consec_grasp is the stricter event and would put the whole task above the diagonal.
    if task in ("egg", "spoon"):
        m["stage"] = m["cgrasp"]
    if task == "coke":
        m["stage"] = m["grasped"]
    return m


def main():
    if RUNS is None:
        sys.exit("run tree not found (expected ../g5grid/runs)")
    seen = {}
    for f in RUNS.rglob("summary.json"):
        k = KEY.match(f.parent.name)
        if k and f.parent.name not in seen:      # first wins; copies are byte-identical
            seen[f.parent.name] = (k.group(1), k.group(2), int(k.group(3)), f)
    out = defaultdict(dict)
    for task, arm, seed, f in seen.values():
        eps = json.load(open(f))["episodes"]
        assert len(eps) == 24, f"{task} {arm} rng{seed}: {len(eps)} episodes, expected 24"
        out[f"{task}|{arm}"][str(seed)] = cell(eps, task)
    assert len(seen) == 3520, f"{len(seen)} distinct runs, expected 3520"
    for k, v in out.items():
        assert len(v) == 20, f"{k}: {len(v)} seeds, expected 20"
    json.dump(out, open(OUT, "w"), separators=(",", ":"))
    print(f"[ok] {OUT}  {len(out)} cells x 20 seeds  ({len(seen)} runs, "
          f"{len(seen) * 24:,} episodes)")


if __name__ == "__main__":
    main()
