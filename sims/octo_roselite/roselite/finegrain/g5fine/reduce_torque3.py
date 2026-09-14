"""Reduce one traces_torque3 cell on the WORKER, then delete the bulky arrays.

Keeps
  summary.json    (unchanged; trace_eval2.py already puts every per-episode scalar
                   in it, so this is the authoritative record)
  drive_config.json
  energy2.json    per-episode scalars, flat, in the shape job_energy.sh's
                  energy.json used, plus the v2 columns
  series.npz      per-episode PER-ACTUATION series, arm joints summed:
                    tau2_<ep>  INT-able mean_substeps Sum_arm tau_drive^2   [N^2 m^2]
                    tau2s_<ep> the same over substeps >= sus_skip only
                    tot2_<ep>  same for (qf + tau_drive)^2
                    qf2_<ep>   same for qf^2      (substep-rate OLD metric)
                    qv2_<ep>   same for qvel^2
                    cF_<ep>    mean external contact force on the robot   [N]
                    cFmax_<ep> max  external contact force on the robot   [N]
                    cG_<ep>    mean contact force on the finger links     [N]
                    tick_<ep>  tick index of each actuation
                  ~100 kB/cell, so the stall/contact analysis can be redone off
                  the sweep without refetching 13 MB/cell of per-tick arrays.
Deletes every .npy, the .png plates and the per-episode _trace.json.
"""
import json, os, sys, pathlib
import numpy as np

d = pathlib.Path(sys.argv[1])
s = json.load(open(d / "summary.json"))
tdt = s["tick_ms"] / 1000.0
adt = s["act_ms"] / 1000.0
JG = {8: slice(0, 6), 11: slice(0, 7)}          # arm joints; fingers/head excluded

eps, ser = [], {}
for e in s["episodes"]:
    ep = e["episode_id"]
    qf = np.load(d / f"ep{ep:02d}_qf.npy")
    qv = np.load(d / f"ep{ep:02d}_qvel.npy")
    a = JG.get(qv.shape[1], slice(None))
    t2 = np.load(d / f"ep{ep:02d}_tau_drive_sq.npy")
    t2s = np.load(d / f"ep{ep:02d}_tau_drive_sq_sus.npy")
    tt = np.load(d / f"ep{ep:02d}_tau_total_sq.npy")
    q2 = np.load(d / f"ep{ep:02d}_qf_sub_sq.npy")
    v2 = np.load(d / f"ep{ep:02d}_qvel_sub_sq.npy")
    con = np.load(d / f"ep{ep:02d}_contact.npy")
    ser[f"tau2_{ep:02d}"] = t2[:, a].sum(1).astype(np.float32)
    ser[f"tau2s_{ep:02d}"] = t2s[:, a].sum(1).astype(np.float32)
    ser[f"tot2_{ep:02d}"] = tt[:, a].sum(1).astype(np.float32)
    ser[f"qf2_{ep:02d}"] = q2[:, a].sum(1).astype(np.float32)
    ser[f"qv2_{ep:02d}"] = v2[:, a].sum(1).astype(np.float32)
    ser[f"cF_{ep:02d}"] = con[:, 2].astype(np.float32)
    ser[f"cFmax_{ep:02d}"] = con[:, 3].astype(np.float32)
    ser[f"cG_{ep:02d}"] = con[:, 4].astype(np.float32)
    ser[f"tick_{ep:02d}"] = np.load(d / f"ep{ep:02d}_act_tick.npy")
    row = dict(ep=ep, success=bool(e["success"]), ticks=int(e["ticks"]),
               # the OLD columns, computed exactly as job_energy.sh did
               t2=float((qf ** 2).sum(1).sum() * tdt),
               eff=float(((qv ** 2).sum(0) * tdt)[a].sum()))
    for k, v in e.items():
        if k in ("episode_stats", "instruction", "mean_abs_rot", "mean_abs_trans"):
            continue
        if isinstance(v, (int, float, bool)) or v is None:
            row.setdefault(k, v)
    eps.append(row)

json.dump(dict(task=s["task"], latency_ms=s["latency_ms"], issue_period_ms=s["issue_period_ms"],
               init_rng=s["init_rng"], n_success=s["n_success"], n_episodes=s["n_episodes"],
               tick_ms=s["tick_ms"], act_ms=s["act_ms"], act_every_ticks=s["act_every_ticks"],
               substep_dt=s["substep_dt"], substeps_per_actuation=s["substeps_per_actuation"],
               sus_skip=s["sus_skip"],
               joint_names=s["joint_names"], contact_cols=s["contact_cols"],
               trace_eval_version=s["trace_eval_version"], episodes=eps),
          open(d / "energy2.json", "w"))
np.savez_compressed(d / "series.npz", **ser)
n = 0
for f in os.listdir(d):
    if f.endswith((".npy", ".png", ".mp4")) or f.endswith("_trace.json"):
        os.remove(d / f); n += 1
print(f"[reduce] {d.name}: {len(eps)} episodes, deleted {n} files, "
      f"kept {sorted(x.name for x in d.iterdir())}")
