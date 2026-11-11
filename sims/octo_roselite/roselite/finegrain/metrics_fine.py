"""Beyond success rate: funnel, time-to-success, hold fraction, excursion.

All numbers MEASURED-in-sim under MODELLED QRB5165 latency, except the
latencies themselves, which are MEASURED on the board.
"""
import glob, json, os, sys
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(ROOT, "runs")
ARMS = ["ctrl_lat0", "pipe110", "pipe200", "serial283", "fp32_555", "cpu685", "noscale_lat0"]


class _Tee:
    def __init__(self, p):
        self.f = open(p, "w")

    def write(self, s):
        sys.__stdout__.write(s); self.f.write(s)

    def flush(self):
        sys.__stdout__.flush(); self.f.flush()


sys.stdout = _Tee(os.path.join(ROOT, "METRICS_FINEGRAIN.txt"))

rows = []
for arm in ARMS:
    ss = [json.load(open(f)) for f in
          sorted(glob.glob(os.path.join(RUNS, arm + "_rng*", "summary.json")))]
    if not ss:
        continue
    eps = [e for s in ss for e in s["episodes"]]
    n = len(eps)
    st = lambda k: [e["episode_stats"].get(k, False) for e in eps]
    ok = [e["success"] for e in eps]
    tick = ss[0]["tick_ms"]

    # time-to-success: episodes terminate the tick success is detected
    succ_ticks = [e["ticks"] for e in eps if e["success"]]
    # hold-tick fraction: ticks running on a NON-fresh (held) action
    hold_frac = 1.0 - float(np.mean([e["frac_fresh"] for e in eps]))

    # rotation / translation excursion over a matched 20-tick prefix of the
    # APPLIED command stream (the same statistic the coarse study used, but the
    # applied stream is now at 40 ms so 20 ticks = 800 ms, not 4 s)
    rot, tra = [], []
    for s in ss:
        d = os.path.join(RUNS, f"{arm}_rng{s['init_rng']}")
        for f in sorted(glob.glob(os.path.join(d, "*_applied_actions.npy"))):
            A = np.load(f)[:20]
            if len(A) < 20:
                continue
            rot.append(np.abs(A[:, 3:6].sum(0)).sum())
            tra.append(np.abs(A[:, :3].sum(0)).sum())
    rows.append(dict(
        arm=arm, lat=ss[0]["latency_ms"], cad=ss[0]["issue_period_ms"], n=n,
        sr=100 * np.mean(ok),
        moved=100 * np.mean(st("moved_correct_obj")),
        grasp=100 * np.mean(st("is_src_obj_grasped")),
        cgrasp=100 * np.mean(st("consecutive_grasp")),
        ontgt=100 * np.mean(st("src_on_target")),
        wrong=100 * np.mean(st("moved_wrong_obj")),
        t_succ=float(np.mean(succ_ticks)) * tick / 1000 if succ_ticks else float("nan"),
        t_succ_sd=float(np.std(succ_ticks)) * tick / 1000 if len(succ_ticks) > 1 else float("nan"),
        holdf=100 * hold_frac,
        inf=float(np.mean([e["n_inferences"] for e in eps])),
        ticks=float(np.mean([e["ticks"] for e in eps])),
        agem=float(np.mean([e["age_mean_ms"] for e in eps if e["age_mean_ms"] is not None])),
        agex=float(np.max([e["age_max_ms"] for e in eps if e["age_max_ms"] is not None])),
        rot=float(np.mean(rot)) if rot else float("nan"),
        tra=float(np.mean(tra)) if tra else float("nan"),
    ))

print("=" * 116)
print("FUNNEL (MEASURED-in-sim under MODELLED latency). Each column is the % of episodes")
print("reaching that stage. success == src_on_target, so the last two columns coincide.")
print("=" * 116)
h = (f"{'arm':<13}{'lat ms':>8}{'n':>5}{'moved obj':>11}{'grasped':>9}{'held 1s':>9}"
     f"{'on target':>11}{'wrong obj':>11}{'SUCCESS%':>10}")
print(h); print("-" * len(h))
for r in rows:
    print(f"{r['arm']:<13}{r['lat']:>8.1f}{r['n']:>5}{r['moved']:>11.1f}{r['grasp']:>9.1f}"
          f"{r['cgrasp']:>9.1f}{r['ontgt']:>11.1f}{r['wrong']:>11.1f}{r['sr']:>10.1f}")

print()
print("=" * 116)
print("TIMING / EFFORT (MEASURED-in-sim). 'hold ticks' = fraction of 40 ms ticks whose action")
print("was a zero-order hold on an already-applied result rather than a freshly landed one.")
print("=" * 116)
h = (f"{'arm':<13}{'lat ms':>8}{'cad ms':>8}{'t to success s':>15}{'hold ticks %':>14}"
     f"{'inf/ep':>8}{'ticks/ep':>10}{'age mean ms':>12}{'age max ms':>11}")
print(h); print("-" * len(h))
for r in rows:
    ts = f"{r['t_succ']:.1f} +/- {r['t_succ_sd']:.1f}" if r['t_succ'] == r['t_succ'] else "-- (no successes)"
    print(f"{r['arm']:<13}{r['lat']:>8.1f}{r['cad']:>8.1f}{ts:>15}{r['holdf']:>14.1f}"
          f"{r['inf']:>8.1f}{r['ticks']:>10.1f}{r['agem']:>12.1f}{r['agex']:>11.1f}")

print()
print("=" * 116)
print("COMMAND EXCURSION over a matched 20-tick (800 ms) prefix of the APPLIED action stream.")
print("Sum of |net rotation| and |net translation| the controller was commanded to integrate.")
print("=" * 116)
h = f"{'arm':<13}{'lat ms':>8}{'|cum rot|':>12}{'|cum trans|':>13}"
print(h); print("-" * len(h))
for r in rows:
    print(f"{r['arm']:<13}{r['lat']:>8.1f}{r['rot']:>12.4f}{r['tra']:>13.4f}")
print()
print("Note: the applied stream is scaled by tick/200ms = 1/5, so these are 1/5 the")
print("magnitude of the coarse study's per-step numbers by construction; compare across")
print("arms, not against the coarse table.")
json.dump(rows, open(os.path.join(ROOT, "metrics_fine.json"), "w"), indent=2)
print(f"\n-> {os.path.join(ROOT, 'metrics_fine.json')}")
