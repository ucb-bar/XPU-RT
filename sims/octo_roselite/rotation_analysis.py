"""Is the low open-loop rotation correlation (r=0.20/-0.10) genuine drift,
or a noise-dominated target that makes correlation uninformative?

Two tests:
  A) BridgeV2 ground truth: lag-1 autocorrelation per action dim. A 'repeat the
     last action' predictor achieves exactly r = lag1-autocorr, so it is a
     cheap reference for how much structure each dim even has.
  B) Closed-loop: integrate the commanded rotation deltas over each episode.
     Genuine drift => cumulative orientation walks away monotonically and the
     excursion grows ~linearly with episode length. Noise => bounded, and the
     cumulative sum stays small relative to sum(|delta|).
"""
import glob, os, pickle, sys
import numpy as np

LBL = ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "grip"]

print("=" * 78)
print("A) BridgeV2 GROUND-TRUTH structure per action dim (MEASURED)")
print("=" * 78)
eps = pickle.load(open("/scratch2/dima/misc_sw/octo_work/bridge_episodes.pkl", "rb"))
A = np.concatenate([np.asarray(e["actions"]) for e in eps], axis=0)
print(f"{'dim':7s} {'std':>9s} {'lag1 autocorr':>14s}   <- persistence-baseline r ceiling proxy")
for i, l in enumerate(LBL[:6]):
    x = A[:, i] - A[:, i].mean()
    print(f"{l:7s} {A[:,i].std():9.4f} {float((x[:-1]*x[1:]).sum()/(x*x).sum()):14.3f}")

print()
print("=" * 78)
print("B) CLOSED-LOOP rotation drift test (MEASURED, SIMPLER rollouts)")
print("=" * 78)
runs = sorted(glob.glob("/scratch2/dima/misc_sw/octo_work/sim_eval/runs/*/"))
print(f"{'run':52s} {'eps':>4s} {'steps':>6s} {'|cumrot|end':>12s} {'sum|rot|':>10s} {'ratio':>7s}")
print("-" * 96)
for r in runs:
    fs = sorted(glob.glob(os.path.join(r, "*_raw_actions.npy")))
    if not fs:
        continue
    ratios, cums, sums, steps = [], [], [], []
    for f in fs:
        a = np.load(f)[:, 3:6]
        cum = np.abs(a.sum(0)).max()          # net orientation excursion, worst axis
        tot = np.abs(a).sum(0).max()          # total path length, same axis
        cums.append(cum); sums.append(tot); steps.append(len(a))
        ratios.append(cum / tot if tot > 0 else 0.0)
    name = os.path.basename(r.rstrip("/"))
    print(f"{name[:52]:52s} {len(fs):4d} {np.mean(steps):6.1f} "
          f"{np.mean(cums):12.4f} {np.mean(sums):10.4f} {np.mean(ratios):7.3f}")
print()
print("ratio = |net rotation| / total rotation path length, averaged over episodes.")
print("  ~1.0  => every step pushes the same way: genuine monotonic drift.")
print("  <<1.0 => deltas cancel: bounded jitter, not drift.")
print("  A pure random walk of n steps gives ratio ~ sqrt(2/(pi*n)) (n=120 -> 0.073).")

print()
print("=" * 78)
print("C) Does rotation excursion DISCRIMINATE success from failure? (MEASURED)")
print("=" * 78)
import json
K = 20   # matched step budget: successes end early, so compare equal prefixes
print(f"{'run':46s} {'grp':>5s} {'n':>3s} {'|cumrot|@20':>12s} {'|cumtr|@20':>14s}")
print("-" * 86)
for r in runs:
    sf = os.path.join(r, "summary.json")
    if not os.path.exists(sf):
        continue
    S = {e["episode_id"]: e["success"] for e in json.load(open(sf))["episodes"]}
    grp = {True: [], False: []}
    grpt = {True: [], False: []}
    for f in sorted(glob.glob(os.path.join(r, "*_raw_actions.npy"))):
        ep = int(os.path.basename(f)[2:4])
        if ep not in S or len(np.load(f)) < K:
            continue
        a = np.load(f)[:K]
        grp[S[ep]].append(np.abs(a[:, 3:6].sum(0)).max())
        grpt[S[ep]].append(np.abs(a[:, :3].sum(0)).max())
    name = os.path.basename(r.rstrip("/"))
    for k in (True, False):
        if grp[k]:
            print(f"{name[:46]:46s} {str(k):>5s} {len(grp[k]):3d} "
                  f"{np.mean(grp[k]):12.4f} {np.mean(grpt[k]):14.4f}")
print()
print("If rotation drift caused failures, failed episodes would show a LARGER")
print("rotation excursion than successful ones over the same 20-step prefix.")
