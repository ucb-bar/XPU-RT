#!/usr/bin/env python3
"""Cross-task and cross-EMBODIMENT contrast: does the latency penalty depend on
what you are evaluating on?

The wide (coarse) sweep's headline was "task choice moved the answer 3x". The
fine model retired that for the two widowx tasks (eggplant vs spoon: +1.3
[-11.8,+14.3] at 283.4 ms). But both are the same embodiment, same bridge scene,
5 Hz control. This script now also runs the harder contrast: widowx (5 Hz,
500 Hz sim, arm_pd_ee_target_delta_pose_align2) against google_robot (3 Hz,
513 Hz sim, arm_pd_ee_delta_pose_align_interpolate_by_planner).

Per arm and per pair of tasks:

    penalty(task)   = mean over seeds of [ SR(arm, seed) - SR(lat0, seed) ]
    difference      = penalty(task B) - penalty(task A)

Within a task the two arms share seeds and configs, so the per-seed difference is
PAIRED and config difficulty cancels. Across tasks the seeds are disjoint
(spoon 90-99, eggplant 100-109, drawer 110-119, coke 120-129), so the two
penalties are INDEPENDENT and their difference gets the pooled standard error
with Welch's df. t quantiles throughout -- with ten seeds t(9)=2.262, not 1.96.

A CI on the difference that contains 0 means no evidence the penalty depends on
which task (or which embodiment) you picked.

Usage:  compare_tasks.py [--pairs A:B,C:D] [--runs DIR]
        default pairs = every pair of tasks present, plus the embodiment contrast
"""
from __future__ import annotations
import json, glob, math, collections, argparse
from pathlib import Path

HERE = Path(__file__).parent
ARMS = ["pipe110", "pipe200", "serial283", "fp32_555", "cpu685"]
LAT = {"pipe110": 117.7, "pipe200": 231.8, "serial283": 283.4,
       "fp32_555": 555.0, "cpu685": 684.8}
# Which embodiment each short task name belongs to (job.sh's task map).
EMBODIMENT = {"egg": "widowx", "spoon": "widowx",
              "drawer": "google_robot", "coke": "google_robot"}

TCRIT = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
         8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160,
         14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093,
         20: 2.086, 22: 2.074, 25: 2.060, 30: 2.042}
def tcrit(df):
    if df < 1: return float("nan")
    if df > 30: return 1.96
    ks = sorted(TCRIT)
    return TCRIT[min(ks, key=lambda k: abs(k - df))]


def load(runs_dir):
    """(task, arm) -> seed -> success rate in points.

    Rejects any google_robot summary that was NOT produced by the fixed harness.
    The pre-fix runs actuated on the 27 Hz fine grid with 1/9-scaled deltas under
    a planner-interpolated controller and scored 0/24 at ZERO latency
    (GOOGLE_ROBOT_PORT.md); pooling them with the fixed ones would be meaningless.
    """
    by = collections.defaultdict(dict)
    rejected = 0
    for f in glob.glob(str(runs_dir / "*" / "*" / "summary.json")):
        try: d = json.load(open(f))
        except Exception: continue
        name = Path(f).parent.name
        if "_rng" not in name: continue
        head, seed = name.rsplit("_rng", 1)
        task, arm = head.split("_", 1)
        if d.get("policy_setup") == "google_robot" and d.get("actuation") != "native":
            rejected += 1
            continue
        eps = d.get("episodes", [])
        if eps:
            by[(task, arm)][int(seed)] = 100.0 * sum(e["success"] for e in eps) / len(eps)
    if rejected:
        print(f"[load] rejected {rejected} pre-fix google_robot run(s) (actuation != native)\n")
    return by


def penalty(by, task, arm):
    """Paired per-seed penalty vs that task's own lat0 arm -> (mean, sd, n)."""
    a, b = by.get((task, arm), {}), by.get((task, "lat0"), {})
    common = sorted(set(a) & set(b))
    if not common: return None
    d = [a[s] - b[s] for s in common]
    mu = sum(d) / len(d)
    sd = math.sqrt(sum((x - mu) ** 2 for x in d) / max(len(d) - 1, 1))
    return mu, sd, len(d)


def fmt(p, w=24):
    if not p: return f"{'(no data)':>{w}s}"
    mu, sd, n = p
    if n < 2: return f"{mu:+6.1f} [   n=1   ] n={n:2d}".rjust(w)
    se = sd / math.sqrt(n)
    tc = tcrit(n - 1)
    return f"{mu:+6.1f} [{mu-tc*se:+6.1f},{mu+tc*se:+6.1f}] n={n:2d}".rjust(w)


def welch(pa, pb):
    """pb - pa for two INDEPENDENT penalty estimates -> (diff, lo, hi, df)."""
    (m1, s1, n1), (m2, s2, n2) = pa, pb
    if n1 < 2 or n2 < 2: return None
    v1, v2 = s1 * s1 / n1, s2 * s2 / n2
    diff, se = m2 - m1, math.sqrt(v1 + v2)
    df = ((v1 + v2) ** 2 / (v1 * v1 / max(n1 - 1, 1) + v2 * v2 / max(n2 - 1, 1))
          if (v1 + v2) > 0 else 1)
    tc = tcrit(df)
    return diff, diff - tc * se, diff + tc * se, df


def combine(ps):
    """Unweighted mean of INDEPENDENT per-task penalties, returned as a pseudo
    (mean, sd, n) whose sd/sqrt(n) is the correct SE, so welch() can consume it."""
    ps = [p for p in ps if p and p[2] > 1]
    if not ps: return None
    mu = sum(p[0] for p in ps) / len(ps)
    var = sum((p[1] ** 2 / p[2]) for p in ps) / (len(ps) ** 2)   # SE^2 of the mean
    n = sum(p[2] for p in ps)
    return mu, math.sqrt(var * n), n


ap = argparse.ArgumentParser()
ap.add_argument("--pairs", default=None,
                help="comma-separated A:B task pairs; default = every pair present")
ap.add_argument("--runs", default=str(HERE / "runs"))
a = ap.parse_args()

by = load(Path(a.runs))
tasks = sorted({t for t, _ in by})
print(f"tasks present: {tasks}")
print("seeds: " + ", ".join(
    f"{t}={min(by[(t,'lat0')])}-{max(by[(t,'lat0')])}"
    for t in tasks if (t, "lat0") in by) + "\n")

if a.pairs:
    pairs = [tuple(p.split(":")) for p in a.pairs.split(",")]
else:
    pairs = [(x, y) for i, x in enumerate(tasks) for y in tasks[i + 1:]]

for A, B in pairs:
    if A not in tasks or B not in tasks:
        print(f"skipping {A} vs {B}: not both present\n"); continue
    ea, eb = EMBODIMENT.get(A, "?"), EMBODIMENT.get(B, "?")
    kind = f"SAME embodiment ({ea})" if ea == eb else f"CROSS-EMBODIMENT ({ea} vs {eb})"
    print("=" * 104)
    print(f"{A} vs {B}   [{kind}]")
    print("=" * 104)
    print(f"{'arm':11s} {'lat_ms':>7s} | {A + ' penalty':>24s} | {B + ' penalty':>24s} | "
          f"{B + ' - ' + A:>26s}")
    print("-" * 104)
    for arm in ARMS:
        pa, pb = penalty(by, A, arm), penalty(by, B, arm)
        line = f"{arm:11s} {LAT[arm]:7.1f} | {fmt(pa)} | {fmt(pb)} | "
        w = welch(pa, pb) if (pa and pb) else None
        if w:
            diff, lo, hi, df = w
            sig = "  SIGNIFICANT" if not (lo <= 0 <= hi) else ""
            line += f"{diff:+6.1f} [{lo:+6.1f},{hi:+6.1f}]{sig}"
        else:
            line += "(need both tasks)"
        print(line)
    print()

# ---- embodiment contrast: the mean penalty over each embodiment's tasks -------
wx = [t for t in tasks if EMBODIMENT.get(t) == "widowx"]
gr = [t for t in tasks if EMBODIMENT.get(t) == "google_robot"]
if wx and gr:
    print("=" * 104)
    print(f"EMBODIMENT CONTRAST   widowx {wx}  vs  google_robot {gr}")
    print("Per-task penalties are independent (disjoint seeds), so the embodiment")
    print("penalty is their unweighted mean and its SE is the pooled one.")
    print("=" * 104)
    print(f"{'arm':11s} {'lat_ms':>7s} | {'widowx penalty':>24s} | {'google_robot penalty':>24s} | "
          f"{'google - widowx':>26s}")
    print("-" * 104)
    for arm in ARMS:
        pw = combine([penalty(by, t, arm) for t in wx])
        pg = combine([penalty(by, t, arm) for t in gr])
        line = f"{arm:11s} {LAT[arm]:7.1f} | {fmt(pw)} | {fmt(pg)} | "
        w = welch(pw, pg) if (pw and pg) else None
        if w:
            diff, lo, hi, df = w
            sig = "  SIGNIFICANT" if not (lo <= 0 <= hi) else ""
            line += f"{diff:+6.1f} [{lo:+6.1f},{hi:+6.1f}]{sig}"
        else:
            line += "(need both embodiments)"
        print(line)
    print()

print("A CI containing 0 in the last column = no evidence the penalty differs.\n"
      "Coarse-model reference at 283.4 ms: eggplant -7.1, spoon -22.9 -> -15.8\n"
      "(the '3x' headline). Fine model, widowx only: +1.3 [-11.8,+14.3].\n"
      "NOTE the resolution limit of the google_robot arms: they actuate on the\n"
      "3 Hz native grid (--actuation native), so a latency change only alters the\n"
      "commanded sequence when an arrival crosses a 333.3 ms control boundary.")
