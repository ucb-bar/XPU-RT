"""Wide multi-instance Octo latency sweep: per-seed, pooled, and cluster-aware stats.

Arms (all four WidowX Bridge tasks share PutOnBridgeInSceneEnv -> control_freq=5
-> 200 ms per env step, so D = ceil(latency_ms / 200 ms) is the same map on each):

  A  0 ms baseline, stock ensembler ON        D=0
  B  283.4 ms MEASURED QRB5165, pipelined     D=2
  C  684.8 ms MEASURED QRB5165, serial        D=4

Everything reported is MEASURED except D, which is MODELLED from the MEASURED
board latency.

A design note that drives the statistics
----------------------------------------
`latency_eval.py` resets with options={"obj_init_options": {"episode_id": ep_id}}
and ep_id runs 0..23 in EVERY run. `--init-rng` seeds only the policy's JAX
sampling key. So a "seed" is a policy-noise replicate over the SAME 24 official
visual-matching episode configurations -- the 24 configs are not resampled.

Consequently the pooled Fisher/Newcombe treatment (which is what the earlier
n=312 analysis used, and is kept here for continuity and comparability) assumes
independent Bernoulli trials it does not strictly have: replicates of the same
config are correlated. The honest denominator is the number of SEEDS, not the
number of episodes. So this script also reports a seed-clustered interval, which
is the one that should be quoted when the two disagree.
"""
import glob, json, os, sys, math
from collections import defaultdict
import numpy as np
from scipy.stats import fisher_exact, beta, binomtest, wilcoxon, t as tdist, norm

HERE = os.path.dirname(os.path.abspath(__file__))
TASKS = {"widowx_put_eggplant_in_basket": "eggplant",
         "widowx_spoon_on_towel": "spoon",
         "widowx_carrot_on_plate": "carrot",
         "widowx_stack_cube": "stack_cube"}


def cp_ci(k, n, alpha=0.05):
    if n == 0:
        return float("nan"), float("nan")
    lo = 0.0 if k == 0 else beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - alpha / 2, k + 1, n - k)
    return 100 * lo, 100 * hi


def wilson(k, n, z=1.959963985):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - h) / d, (c + h) / d


def newcombe(k1, n1, k0, n0):
    """Newcombe hybrid-score CI for p1 - p0, in percentage points."""
    l1, u1 = wilson(k1, n1)
    l0, u0 = wilson(k0, n0)
    d = k1 / n1 - k0 / n0
    lo = d - math.sqrt((k1 / n1 - l1) ** 2 + (u0 - k0 / n0) ** 2)
    hi = d + math.sqrt((u1 - k1 / n1) ** 2 + (k0 / n0 - l0) ** 2)
    return 100 * lo, 100 * hi


def arm_of(s):
    # run_eval.py (the stock free-running harness) writes a different schema: no
    # latency_ms/pipeline/ensemble, just action_ensemble. Latency 0 + ensembler ON
    # IS arm A -- latency_eval.py's docstring notes that D=0 reproduces the stock
    # configuration bit-for-bit, and RESULTS.txt measured 13/15 (stock) vs 14
    # (wrapper) on the same config. So fold those in as arm A.
    if "latency_ms" not in s:
        return "A" if s.get("action_ensemble") else None
    lat, pipe, ens = s["latency_ms"], s["pipeline"], s["ensemble"]
    if abs(lat) < 1e-9 and not pipe and ens == "stock":
        return "A"
    if abs(lat - 283.4) < 1e-6 and pipe:
        return "B"
    if abs(lat - 684.8) < 1e-6 and not pipe and ens == "none":
        return "C"
    return None


def load(patterns):
    out, files = [], []
    for pat in patterns:
        for f in sorted(glob.glob(pat)):
            try:
                s = json.load(open(f))
            except Exception:
                continue
            if "n_success" not in s or s.get("n_episodes", 0) < 24:
                continue
            # octo-small-1.0 ONLY. The tree also holds octo-small-1.5 runs, which
            # are a different policy (and score 0% on carrot/cube/spoon).
            ck = s.get("ckpt", "")
            if "octo-small" not in ck or "1.5" in ck:
                continue
            s["_file"] = f
            out.append(s)
            files.append(f)
    return out


def collect(S):
    """-> data[task][arm][seed] = dict(ok, n, per_ep {ep_id: bool}, file)"""
    data = defaultdict(lambda: defaultdict(dict))
    for s in S:
        a = arm_of(s)
        if a is None:
            continue
        task = s.get("task")
        if task not in TASKS:
            continue
        seed = s["init_rng"]
        if seed in data[task][a]:            # de-duplicate: keep the larger n
            if s["n_episodes"] <= data[task][a][seed]["n"]:
                continue
        data[task][a][seed] = dict(
            ok=s["n_success"], n=s["n_episodes"], wall=s.get("wall_s", 0.0),
            per_ep={e["episode_id"]: bool(e["success"]) for e in s.get("episodes", [])},
            file=s["_file"], worker=worker_of(s["_file"]))
    return data


def worker_of(path):
    """fetch_wide.sh drops each box's runs under g5wide/runs/<w#>/, so the box a
    run came from is recoverable from the path. Everything else is prior work."""
    parts = path.split(os.sep)
    for i, p in enumerate(parts):
        if p == "runs" and i + 1 < len(parts) and parts[i + 1].startswith("w") \
                and parts[i + 1][1:].isdigit():
            return parts[i + 1]
    if os.sep + "g5" + os.sep in path:
        return "w0-prior"
    return "local-prior"


def worker_qc(arms, task):
    """QC only: does any box look different on the SAME arm? Seeds are assigned so
    that both arms of a seed run on one box, so a box effect cannot confound the
    A-vs-B contrast -- but a broken box would still show up here."""
    print(f"\n  PER-WORKER QC ({task}) -- same arm, split by box")
    for a in ["A", "B"]:
        if not arms.get(a):
            continue
        byw = defaultdict(lambda: [0, 0])
        for v in arms[a].values():
            byw[v["worker"]][0] += v["ok"]; byw[v["worker"]][1] += v["n"]
        cells = sorted(byw.items())
        txt = "  ".join(f"{w}:{k}/{n}={100*k/n:.0f}%" for w, (k, n) in cells)
        print(f"    arm {a}:  {txt}")
        if len(cells) > 1:
            K = sum(k for _, (k, _) in cells); N = sum(n for _, (_, n) in cells)
            worst = None
            for w, (k, n) in cells:
                p = fisher_exact([[k, n - k], [K - k, N - n - (K - k)]])[1]
                if worst is None or p < worst[1]:
                    worst = (w, p)
            print(f"      most-deviant box vs the rest: {worst[0]}  Fisher p = {worst[1]:.3f}"
                  f"{'  <-- CHECK' if worst[1] < 0.01 else ''}")


def per_seed_table(d, label):
    seeds = sorted(d)
    k = sum(d[s]["ok"] for s in seeds)
    n = sum(d[s]["n"] for s in seeds)
    print(f"\n  {label}   ({len(seeds)} seeds, n={n})")
    line = "    "
    for i, s in enumerate(seeds):
        line += f"{s}:{d[s]['ok']:>2}/{d[s]['n']:<2} "
        if (i + 1) % 8 == 0:
            print(line); line = "    "
    if line.strip():
        print(line)
    lo, hi = cp_ci(k, n)
    print(f"    POOLED {k}/{n} = {100*k/n:.1f}%   95% CI (Clopper-Pearson) [{lo:.1f}, {hi:.1f}]")
    return k, n, seeds


def compare(dx, dy, namex, namey):
    """dx = test arm, dy = reference arm. Prints pooled + seed-clustered stats."""
    kx = sum(v["ok"] for v in dx.values()); nx = sum(v["n"] for v in dx.values())
    ky = sum(v["ok"] for v in dy.values()); ny = sum(v["n"] for v in dy.values())
    if nx == 0 or ny == 0:
        print("    (insufficient data)"); return
    diff = 100 * (kx / nx - ky / ny)
    p = fisher_exact([[kx, nx - kx], [ky, ny - ky]])[1]
    lo, hi = newcombe(kx, nx, ky, ny)
    print(f"\n  {namex}  vs  {namey}")
    print(f"    {kx}/{nx} = {100*kx/nx:.1f}%   vs   {ky}/{ny} = {100*ky/ny:.1f}%")
    print(f"    difference = {diff:+.1f} points")
    print(f"    Fisher exact two-sided        p = {p:.4f}"
          f"{'   ***' if p < 0.05 else '   n.s.'}")
    print(f"    95% CI on the difference (Newcombe, episode-level): "
          f"[{lo:+.1f}, {hi:+.1f}] points   half-width +/-{(hi-lo)/2:.1f}")

    # ---- seed-clustered (the design-honest denominator) ----
    common = sorted(set(dx) & set(dy))
    if len(common) >= 3:
        dx_r = np.array([dx[s]["ok"] / dx[s]["n"] for s in common])
        dy_r = np.array([dy[s]["ok"] / dy[s]["n"] for s in common])
        delta = 100 * (dx_r - dy_r)
        m, sd, nn = delta.mean(), delta.std(ddof=1), len(delta)
        se = sd / math.sqrt(nn)
        tcrit = tdist.ppf(0.975, nn - 1)
        clo, chi = m - tcrit * se, m + tcrit * se
        pos = int((delta > 0).sum()); neg = int((delta < 0).sum()); tie = int((delta == 0).sum())
        sg = binomtest(pos, pos + neg, 0.5).pvalue if pos + neg else float("nan")
        try:
            wp = wilcoxon(dx_r - dy_r, zero_method="wilcox").pvalue
        except Exception:
            wp = float("nan")
        pt = tdist.sf(abs(m / se), nn - 1) * 2 if se > 0 else float("nan")
        print(f"    PAIRED BY SEED ({nn} seeds shared, this is the design-honest unit)")
        print(f"      mean per-seed delta = {m:+.1f} points  (sd {sd:.1f})")
        print(f"      95% CI seed-clustered         [{clo:+.1f}, {chi:+.1f}] points"
              f"   half-width +/-{(chi-clo)/2:.1f}")
        print(f"      paired t p = {pt:.4f} | sign test p = {sg:.4f} "
              f"({neg} worse, {pos} better, {tie} tied) | Wilcoxon p = {wp:.4f}")
    return kx, nx, ky, ny


def per_episode(dx, dy, namex, namey):
    """Same 24 episode configs recur in every run -> pair on episode_id too."""
    common = sorted(set(dx) & set(dy))
    if not common:
        return
    ids = sorted({e for s in common for e in dx[s]["per_ep"]})
    print(f"\n  PER-EPISODE-CONFIG rate over {len(common)} shared seeds "
          f"({namey} -> {namex}):")
    rows = []
    for e in ids:
        a = [dy[s]["per_ep"][e] for s in common if e in dy[s]["per_ep"]]
        b = [dx[s]["per_ep"][e] for s in common if e in dx[s]["per_ep"]]
        if a and b:
            rows.append((e, 100 * np.mean(a), 100 * np.mean(b), len(a)))
    for i in range(0, len(rows), 4):
        print("    " + "  ".join(
            f"ep{e:02d} {ra:5.1f}->{rb:5.1f}" for e, ra, rb, _ in rows[i:i + 4]))
    d = np.array([rb - ra for _, ra, rb, _ in rows])
    if len(d) >= 3:
        se = d.std(ddof=1) / math.sqrt(len(d))
        tc = tdist.ppf(0.975, len(d) - 1)
        print(f"    mean per-config delta = {d.mean():+.1f} points, "
              f"95% CI [{d.mean()-tc*se:+.1f}, {d.mean()+tc*se:+.1f}]  "
              f"({int((d<0).sum())} configs worse, {int((d>0).sum())} better, "
              f"{int((d==0).sum())} tied, of {len(d)})")


def power_n(p0, delta, alpha=0.05, power=0.80):
    p1 = p0 + delta
    pbar = (p0 + p1) / 2
    za, zb = norm.ppf(1 - alpha / 2), norm.ppf(power)
    return (za * math.sqrt(2 * pbar * (1 - pbar)) + zb *
            math.sqrt(p0 * (1 - p0) + p1 * (1 - p1))) ** 2 / delta ** 2


if __name__ == "__main__":
    # Order matters: de-duplication on (task, arm, seed) keeps the FIRST file
    # seen. The stock local run_eval.py runs come first so that local arm A
    # seed 0 stays the 13/24 the published n=312 analysis used (the same config
    # was measured three times locally: 13, 15, 14 -- see roselite/RESULTS.txt
    # section 0 on harness nondeterminism).
    pats = sys.argv[1:] or [
        os.path.join(HERE, "..", "..", "runs", "*", "summary.json"),
        os.path.join(HERE, "..", "runs", "*", "summary.json"),
        os.path.join(HERE, "..", "g5", "runs", "*", "summary.json"),
        # g5wide is one level deeper: runs/<worker>/<rundir>/summary.json
        os.path.join(HERE, "runs", "*", "*", "summary.json"),
        os.path.join(HERE, "validation", "*", "*", "summary.json"),
    ]
    S = load(pats)
    data = collect(S)
    print(f"loaded {len(S)} summary.json files")
    print("=" * 92)
    print("MEASURED success, octo-small-1.0, corrected masked unnormalization,")
    print("official SIMPLER visual-matching protocol, 24 episodes/seed.")
    print("D = ceil(latency_ms / 200 ms) is MODELLED from MEASURED board latency;")
    print("control_freq = 5 verified on PutOnBridgeInSceneEnv (all four tasks).")
    print("=" * 92)

    for task in ["widowx_put_eggplant_in_basket", "widowx_spoon_on_towel",
                 "widowx_carrot_on_plate", "widowx_stack_cube"]:
        if task not in data:
            continue
        print("\n" + "#" * 92)
        print(f"# TASK  {task}   ({TASKS[task]})")
        print("#" * 92)
        arms = data[task]
        labels = {"A": "ARM A  0 ms baseline, ensembler ON        D=0",
                  "B": "ARM B  283.4 ms MEASURED, pipelined       D=2",
                  "C": "ARM C  684.8 ms MEASURED, serial, no ens  D=4"}
        for a in ["A", "B", "C"]:
            if arms.get(a):
                per_seed_table(arms[a], labels[a])
        if arms.get("A") and arms.get("B"):
            compare(arms["B"], arms["A"], "B 283.4 ms pipelined", "A 0 ms baseline")
            worker_qc(arms, TASKS[task])
            per_episode(arms["B"], arms["A"], "B", "A")
        if arms.get("A") and arms.get("C"):
            compare(arms["C"], arms["A"], "C 684.8 ms serial", "A 0 ms baseline")

    print("\n" + "=" * 92)
    print("POWER REFERENCE (two-sided alpha=0.05, 80% power, independent-Bernoulli)")
    for d in (0.03, 0.05, 0.08, 0.10):
        print(f"  detecting {100*d:.0f} points around a 55% base rate needs "
              f"n ~= {power_n(0.55, -d):.0f} episodes per arm")
    print("=" * 92)
