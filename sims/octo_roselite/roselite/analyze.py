"""Aggregate the RoSE-lite latency sweep: success rate vs MODELLED latency,
plus the rotation-excursion failure-mode check."""
import glob, json, math, os, sys
import numpy as np

RL = "/scratch2/dima/misc_sw/octo_work/sim_eval/roselite/runs"
SE = "/scratch2/dima/misc_sw/octo_work/sim_eval/runs"


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * max(0, c - h), 100 * min(1, c + h))


def load(pat, root):
    out = []
    for f in sorted(glob.glob(os.path.join(root, pat, "summary.json"))):
        s = json.load(open(f))
        s["_dir"] = os.path.dirname(f)
        out.append(s)
    return out


# ---------------------------------------------------------------- harness noise
print("=" * 96)
print("0) HARNESS REPRODUCIBILITY (MEASURED) -- identical config, repeated")
print("=" * 96)
reps = []
for d, lbl in [(f"{SE}/widowx_put_eggplant_in_basket_octo-small_rng0", "stock run_eval.py (original)"),
               (f"{SE}/widowx_put_eggplant_in_basket_octo-small_rng0_REPRO", "stock run_eval.py (re-run)"),
               (f"{RL}/lat0ms_D0_from_arrival_ens-stock_rng0_VALIDATE", "wrapper latency=0 --ensemble stock")]:
    p = os.path.join(d, "summary.json")
    if os.path.exists(p):
        s = json.load(open(p))
        reps.append(s["n_success"])
        print(f"  {lbl:42s} {s['n_success']:2d}/{s['n_episodes']} = {100*s['success_rate']:.1f}%")
if len(reps) > 1:
    print(f"\n  -> spread over {len(reps)} replicates of the SAME configuration: "
          f"{min(reps)}-{max(reps)}/24 (mean {np.mean(reps):.1f}). The harness is"
          f" NOT bit-reproducible;\n     JAX/GPU nondeterminism + contact-rich SAPIEN"
          f" physics diverge chaotically. Treat +/-2 successes at n=24 as floor noise.")

# ---------------------------------------------------------------- latency sweep
print()
print("=" * 96)
print("1) SUCCESS RATE vs MODELLED LATENCY  (serial schedule, chunk executed from arrival)")
print("=" * 96)
groups = {}
for s in load("lat*_from_arrival_ens-none_rng*", RL):
    if s.get("pipeline"):
        continue
    groups.setdefault((s["latency_ms"], s["D_steps"]), []).append(s)

hdr = (f"{'latency (MEASURED board)':28s} {'D':>2s} {'seeds':>16s} {'n':>4s} {'ok':>4s} "
       f"{'SR%':>6s} {'95% CI':>14s} {'grasp%':>7s} {'ontgt%':>7s} {'inf/ep':>7s} {'fresh':>6s}")
print(hdr); print("-" * len(hdr))
curve = []
for (lat, D) in sorted(groups):
    runs = groups[(lat, D)]
    eps = [e for s in runs for e in s["episodes"]]
    n = len(eps); ok = sum(e["success"] for e in eps)
    seeds = ",".join(str(s["init_rng"]) for s in sorted(runs, key=lambda x: x["init_rng"]))
    g = 100 * np.mean([e["episode_stats"].get("is_src_obj_grasped", False) for e in eps])
    o = 100 * np.mean([e["episode_stats"].get("src_on_target", False) for e in eps])
    inf = np.mean([e["n_inferences"] for e in eps]); fr = np.mean([e["frac_fresh"] for e in eps])
    lo, hi = wilson(ok, n)
    curve.append((lat, D, n, ok, 100 * ok / n, lo, hi))
    print(f"{lat:>8.1f} ms {'(free-run)' if lat==0 else '':17s} {D:2d} {seeds:>16s} {n:4d} {ok:4d} "
          f"{100*ok/n:6.1f} {f'[{lo:.1f},{hi:.1f}]':>14s} {g:7.1f} {o:7.1f} {inf:7.1f} {fr:6.2f}")

print()
print("  Reference (MEASURED, stock free-running harness, ensembler ON, 3 seeds 0/2/4):")
st = load("widowx_put_eggplant_in_basket_octo-small_rng[024]", SE)
st = [s for s in st if not s["_dir"].endswith("_REPRO")]
if st:
    n = sum(s["n_episodes"] for s in st); ok = sum(s["n_success"] for s in st)
    lo, hi = wilson(ok, n)
    print(f"    stock baseline {ok}/{n} = {100*ok/n:.1f}%  95% CI [{lo:.1f},{hi:.1f}]")

# --------------------------------------------------------------- pipelined
pipe = {}
for s in load("lat*_pipe_rng*", RL):
    pipe.setdefault((s["latency_ms"], s["D_steps"]), []).append(s)
if pipe:
    print()
    print("=" * 96)
    print("2) PIPELINED upper bound (a fresh inference every 200 ms, each D steps late,")
    print("   timestep-aligned ensembling over however many predictions target this step).")
    print("   NOT deployable on one accelerator chain -- it needs 1/D of an inference per period.")
    print("=" * 96)
    print(f"{'latency':>10s} {'D':>2s} {'#avgd':>6s} {'seeds':>10s} {'n':>4s} {'ok':>4s} {'SR%':>6s} {'95% CI':>14s}")
    print("-" * 64)
    for (lat, D) in sorted(pipe):
        runs = pipe[(lat, D)]
        eps = [e for s in runs for e in s["episodes"]]
        n = len(eps); ok = sum(e["success"] for e in eps)
        seeds = ",".join(str(s["init_rng"]) for s in sorted(runs, key=lambda x: x["init_rng"]))
        lo, hi = wilson(ok, n)
        print(f"{lat:10.1f} {D:2d} {max(0, 4-D):6d} {seeds:>10s} {n:4d} {ok:4d} {100*ok/n:6.1f} "
              f"{f'[{lo:.1f},{hi:.1f}]':>14s}")

# --------------------------------------------------------- rotation excursion
print()
print("=" * 96)
print("3) FAILURE MODE: rotation vs translation excursion, success vs failure (MEASURED)")
print("   |net excursion| over a matched 20-step prefix of the APPLIED command stream.")
print("=" * 96)
K = 20
print(f"{'latency':>10s} {'grp':>5s} {'n':>4s} {'|cum rot|':>11s} {'|cum trans|':>12s}")
print("-" * 48)
rows = {}
for (lat, D) in sorted(groups):
    gr = {True: [], False: []}
    gt = {True: [], False: []}
    for s in groups[(lat, D)]:
        S = {e["episode_id"]: e["success"] for e in s["episodes"]}
        for f in sorted(glob.glob(os.path.join(s["_dir"], "*_applied_actions.npy"))):
            ep = int(os.path.basename(f)[2:4])
            A = np.load(f)
            if ep not in S or len(A) < K:
                continue
            A = A[:K]
            gr[S[ep]].append(np.abs(A[:, 3:6].sum(0)).max())
            gt[S[ep]].append(np.abs(A[:, :3].sum(0)).max())
    for k in (True, False):
        if gr[k]:
            print(f"{lat:10.1f} {str(k):>5s} {len(gr[k]):4d} {np.mean(gr[k]):11.4f} {np.mean(gt[k]):12.4f}")
    if gr[True] and gr[False]:
        rows[lat] = (np.mean(gr[False]) / max(np.mean(gr[True]), 1e-9),
                     np.mean(gt[False]) / max(np.mean(gt[True]), 1e-9),
                     np.mean(gr[True]), np.mean(gr[False]))
print()
print(f"{'latency':>10s} {'rot fail/ok':>12s} {'trans fail/ok':>14s}   (earlier free-run finding: rot 2.53x, trans ~1x)")
print("-" * 70)
for lat in sorted(rows):
    r = rows[lat]
    print(f"{lat:10.1f} {r[0]:12.2f} {r[1]:14.2f}")

print()
print("=" * 96)
print("4) ROTATION EXCURSION vs LATENCY, all episodes pooled (does latency inflate it?)")
print("=" * 96)
print(f"{'latency':>10s} {'n':>4s} {'mean |cum rot|@20':>19s} {'mean |cum trans|@20':>21s}")
print("-" * 58)
for (lat, D) in sorted(groups):
    R, T = [], []
    for s in groups[(lat, D)]:
        for f in sorted(glob.glob(os.path.join(s["_dir"], "*_applied_actions.npy"))):
            A = np.load(f)
            if len(A) < K:
                continue
            R.append(np.abs(A[:K, 3:6].sum(0)).max())
            T.append(np.abs(A[:K, :3].sum(0)).max())
    if R:
        print(f"{lat:10.1f} {len(R):4d} {np.mean(R):19.4f} {np.mean(T):21.4f}")

json.dump(curve, open(f"{RL}/../curve.json", "w"), indent=2)
