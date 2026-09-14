import glob, json, os, numpy as np
rows=[]
for f in sorted(glob.glob("/scratch2/dima/misc_sw/octo_work/sim_eval/runs/*/summary.json")):
    s=json.load(open(f))
    eps=s["episodes"]
    def frac(k): return float(np.mean([e["episode_stats"].get(k,False) for e in eps]))
    rows.append(dict(run=os.path.basename(os.path.dirname(f)), task=s["task"],
        rng=s["init_rng"], legacy=s["legacy_unnorm"], n=s["n_episodes"],
        ok=s["n_success"], sr=s["success_rate"],
        grasped=frac("is_src_obj_grasped"), consec=frac("consecutive_grasp"),
        moved=frac("moved_correct_obj"), on_target=frac("src_on_target"),
        grip_closed=float(np.mean([e["gripper_frac_closed"] for e in eps])),
        rot=[round(x,4) for x in np.mean([e["mean_abs_rot"] for e in eps],axis=0)],
        trans=[round(x,4) for x in np.mean([e["mean_abs_trans"] for e in eps],axis=0)],
        wall=s["wall_s"]))
hdr=f"{'run':62s} {'n':>3s} {'ok':>3s} {'SR%':>6s} {'grasp':>6s} {'ontgt':>6s} {'gclose':>7s}"
print(hdr); print("-"*len(hdr))
for r in rows:
    print(f"{r['run'][:62]:62s} {r['n']:3d} {r['ok']:3d} {100*r['sr']:6.1f} "
          f"{100*r['grasped']:6.1f} {100*r['on_target']:6.1f} {r['grip_closed']:7.2f}")
print()
for r in rows:
    print(f"{r['run'][:62]:62s} mean|rot|={r['rot']}  mean|trans|={r['trans']}  wall={r['wall']}s")
json.dump(rows, open("/scratch2/dima/misc_sw/octo_work/sim_eval/aggregate.json","w"), indent=2)
