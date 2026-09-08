#!/usr/bin/env python3
"""Reduce traces_torque2/ (688 MB of per-tick arrays) to the scalars fig_energy needs.

fig_energy integrates two quantities per episode. Both collapse to ONE NUMBER per
episode, so the figure does not need the arrays -- only this cache, which is a few KB.
Run this once where the traces live; commit the cache; the figure then reproduces
anywhere. Regenerate with `python make_energy_cache.py` if the traces change.
"""
import json
from pathlib import Path
import numpy as np

SRC = Path(__file__).parent.parent / "traces_torque2"
OUT = Path(__file__).parent / "energy_cache.json"
JG = {8: slice(0, 6), 11: slice(0, 7)}      # arm joints only; fingers/head excluded

cache = {}
for d in sorted(p for p in SRC.iterdir() if p.is_dir()):
    f = d / "summary.json"
    if not f.exists():
        continue
    s = json.load(open(f))
    dt = s["tick_ms"] / 1000.0
    t2, eff, inf, act = [], [], [], []
    for e in s["episodes"]:
        ep = e["episode_id"]
        try:
            qf = np.load(d / f"ep{ep:02d}_qf.npy")
            w = np.load(d / f"ep{ep:02d}_qvel.npy")
        except Exception:
            continue
        a = JG.get(w.shape[1], slice(None))
        t2.append(float((qf ** 2).sum(1).sum() * dt))
        eff.append(float(((w ** 2).sum(0) * dt)[a].sum()))
        inf.append(e["n_inferences"]); act.append(e.get("n_actuations", e["ticks"]))
    if not t2:
        continue
    ages = [e.get("act_age_mean_ms") for e in s["episodes"]
            if e.get("act_age_mean_ms") is not None]
    cache[d.name] = dict(
        t2=t2, eff=eff, n=len(t2), dt=dt, nep=s["n_episodes"],
        sr=100.0 * s["n_success"] / s["n_episodes"],
        grid=s.get("act_ms", s["tick_ms"]),
        age=float(np.mean(ages)) if ages else 0.0,
        ipa=float(np.sum(inf)) / float(np.sum(act)))
json.dump(cache, open(OUT, "w"))
print(f"[ok] {OUT}  {len(cache)} cells, {OUT.stat().st_size/1024:.0f} KB "
      f"(from {sum(f.stat().st_size for f in SRC.rglob('*'))/1e6:.0f} MB of traces)")
