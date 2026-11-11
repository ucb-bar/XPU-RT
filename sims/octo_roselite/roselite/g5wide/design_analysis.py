"""Design-appropriate analysis of the Octo latency arms.

WHY THIS FILE EXISTS
--------------------
Both `latency_eval.py` and `run_eval.py` reset with

    env.reset(options={"obj_init_options": {"episode_id": ep_id}})   ep_id = 0..23

in EVERY run, and `--init-rng` seeds ONLY the policy's JAX sampling key
(`Octo15Inference(..., init_rng=...)`). So a "seed" is a policy-noise replicate
over the SAME 24 official visual-matching episode configurations. The episode
set is never resampled.

That means an arm reported as "n = 312 episodes" is really 24 episode
configurations with ~13 replicates each. The 24 configs differ enormously in
intrinsic difficulty (measured: from ~15% to ~100% success), so episodes are
strongly correlated within a config and a binomial interval computed as if all
312 were independent is OVERCONFIDENT for that arm's marginal rate.

For the DIFFERENCE between two arms the picture is the opposite, and this is the
important part: both arms see the same 24 configs, so the config-difficulty
variance is COMMON to both arms and cancels when you pair on config. Fisher /
Newcombe treat the arms as two independent unpaired samples and therefore throw
that cancellation away, which makes their interval on the difference too WIDE.
The design-appropriate paired analysis is expected to be tighter.

WHAT IS REPORTED
  1. ICC / design effect per arm -> effective n, i.e. how overconfident the
     marginal per-arm intervals were.
  2. CLUSTER BOOTSTRAP over the 24 episode configs -> the headline CI on the
     difference. Resampling configs (not episodes) respects both the pairing and
     the within-config replication.
  3. PER-CONFIG PAIRED t / Wilcoxon on the 24 config-level rate differences.
  4. McNEMAR on (seed, config) matched pairs.
  5. The naive pooled Fisher / Newcombe, kept ONLY for continuity with the
     previously published n=312 number.

Caveat carried through the output: pairing on `init_rng` is weak. The arms issue
different numbers of inference calls, so their JAX RNG streams diverge after the
first call, and the harness is nondeterministic even for an identical
command+seed (MEASURED: 13/24 then 15/24 on the same configuration). The
pairing that carries real information is on episode_id, which fixes the initial
object poses. Seed-level pairing is reported but not leaned on.
"""
import glob, json, math, os, sys
from collections import defaultdict
import numpy as np
from scipy.stats import fisher_exact, beta, binomtest, wilcoxon, t as tdist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_wide import load, collect, newcombe, cp_ci, TASKS

HERE = os.path.dirname(os.path.abspath(__file__))
RNG = np.random.default_rng(20260906)


def matrix(arm, configs=None):
    """-> (seeds, configs, M) with M[s, c] in {0,1}, NaN where absent."""
    seeds = sorted(arm)
    if configs is None:
        configs = sorted({c for s in seeds for c in arm[s]["per_ep"]})
    M = np.full((len(seeds), len(configs)), np.nan)
    for i, s in enumerate(seeds):
        for j, c in enumerate(configs):
            v = arm[s]["per_ep"].get(c)
            if v is not None:
                M[i, j] = float(v)
    return seeds, configs, M


def icc_deff(M):
    """One-way random-effects ICC with episode CONFIG as the cluster."""
    k = M.shape[1]
    cols = [M[:, j][~np.isnan(M[:, j])] for j in range(k)]
    cols = [c for c in cols if len(c) > 1]
    if len(cols) < 2:
        return float("nan"), float("nan"), float("nan")
    N = sum(len(c) for c in cols)
    gm = np.concatenate(cols).mean()
    ssb = sum(len(c) * (c.mean() - gm) ** 2 for c in cols)
    ssw = sum(((c - c.mean()) ** 2).sum() for c in cols)
    msb = ssb / (len(cols) - 1)
    msw = ssw / (N - len(cols))
    ns = [len(c) for c in cols]
    m0 = (N - sum(n * n for n in ns) / N) / (len(cols) - 1)
    icc = (msb - msw) / (msb + (m0 - 1) * msw) if (msb + (m0 - 1) * msw) > 0 else 0.0
    icc = max(icc, 0.0)
    deff = 1 + (m0 - 1) * icc
    return icc, deff, N / deff


def cluster_bootstrap(Ma, Mb, B=20000):
    """Resample the 24 episode CONFIGS with replacement (columns), recompute the
    pooled rate difference. Respects pairing + within-config replication."""
    k = Ma.shape[1]
    a_ok = np.nansum(Ma, axis=0); a_n = np.sum(~np.isnan(Ma), axis=0)
    b_ok = np.nansum(Mb, axis=0); b_n = np.sum(~np.isnan(Mb), axis=0)
    idx = RNG.integers(0, k, size=(B, k))
    dio = (b_ok[idx].sum(1) / b_n[idx].sum(1)) - (a_ok[idx].sum(1) / a_n[idx].sum(1))
    point = b_ok.sum() / b_n.sum() - a_ok.sum() / a_n.sum()
    lo, hi = np.percentile(dio, [2.5, 97.5])
    # bootstrap p for H0: difference = 0 (centre the distribution)
    centred = dio - dio.mean()
    p = float((np.abs(centred) >= abs(point)).mean())
    return 100 * point, 100 * lo, 100 * hi, p


def mcnemar(Ma, Mb):
    both = (~np.isnan(Ma)) & (~np.isnan(Mb))
    a = Ma[both]; b = Mb[both]
    n01 = int(((a == 1) & (b == 0)).sum())   # baseline win, latency loss
    n10 = int(((a == 0) & (b == 1)).sum())   # latency win, baseline loss
    conc = int(both.sum()) - n01 - n10
    if n01 + n10 == 0:
        return n01, n10, conc, float("nan"), (float("nan"), float("nan"))
    p = binomtest(n10, n01 + n10, 0.5).pvalue
    n = int(both.sum())
    d = (n10 - n01) / n
    var = (n01 + n10 - (n10 - n01) ** 2 / n) / n ** 2
    se = math.sqrt(max(var, 0.0))
    return n01, n10, conc, p, (100 * (d - 1.96 * se), 100 * (d + 1.96 * se))


def analyse(task, arms, tag_x="B", tag_y="A",
            name_x="283.4 ms pipelined (D=2)", name_y="0 ms baseline (D=0)"):
    if not (arms.get(tag_x) and arms.get(tag_y)):
        return
    ax, ay = arms[tag_x], arms[tag_y]
    common_seeds = sorted(set(ax) & set(ay))
    cfgs = sorted(set.intersection(
        *[set(ay[s]["per_ep"]) for s in common_seeds],
        *[set(ax[s]["per_ep"]) for s in common_seeds])) if common_seeds else []
    if not cfgs:
        return
    _, _, Ma = matrix({s: ay[s] for s in common_seeds}, cfgs)
    _, _, Mb = matrix({s: ax[s] for s in common_seeds}, cfgs)

    ka = int(np.nansum(Ma)); na = int((~np.isnan(Ma)).sum())
    kb = int(np.nansum(Mb)); nb = int((~np.isnan(Mb)).sum())

    print("\n" + "=" * 94)
    print(f"{TASKS[task].upper()}:  {name_x}   vs   {name_y}")
    print(f"{len(common_seeds)} shared seeds x {len(cfgs)} episode configs")
    print("=" * 94)

    print("\n-- 0. THE DESIGN --------------------------------------------------------")
    print(f"   arm {tag_y}: {ka}/{na} = {100*ka/na:.1f}%    arm {tag_x}: {kb}/{nb} = {100*kb/nb:.1f}%")
    print(f"   These are NOT {na} independent samples. They are {len(cfgs)} episode")
    print(f"   configurations with {len(common_seeds)} policy-noise replicates each.")
    for nm, M, k, n in ((tag_y, Ma, ka, na), (tag_x, Mb, kb, nb)):
        icc, deff, neff = icc_deff(M)
        lo, hi = cp_ci(k, n)
        # interval recomputed at the effective sample size
        ke = k / n * neff
        elo, ehi = cp_ci(ke, neff)
        print(f"   arm {nm}: ICC(config) = {icc:.3f}   design effect = {deff:.2f}"
              f"   effective n = {neff:.0f}  (nominal {n})")
        print(f"            naive 95% CI [{lo:.1f}, {hi:.1f}]  ->  "
              f"design-corrected [{elo:.1f}, {ehi:.1f}]")
        # As replicates R -> inf with the config count K fixed,
        #   n_eff = K*R / (1 + (R-1)*ICC)  ->  K / ICC.
        # The marginal rate of an arm can never be measured better than this,
        # no matter how many seeds are run, because there are only K configs.
        if icc > 1e-6:
            cap = len(cfgs) / icc
            clo, chi = cp_ci(k / n * cap, cap)
            print(f"            CEILING: with only {len(cfgs)} configs, n_eff -> "
                  f"{cap:.0f} as seeds -> inf; best-possible 95% CI "
                  f"[{clo:.1f}, {chi:.1f}]")

    print("\n-- 1. HEADLINE: cluster bootstrap over the 24 episode configs -----------")
    pt, blo, bhi, bp = cluster_bootstrap(Ma, Mb)
    print(f"   difference ({tag_x} - {tag_y}) = {pt:+.1f} points")
    print(f"   95% CI  [{blo:+.1f}, {bhi:+.1f}] points   half-width +/-{(bhi-blo)/2:.1f}"
          f"   bootstrap p = {bp:.4f}")

    print("\n-- 2. PER-CONFIG PAIRED (24 pairs, config difficulty cancels) -----------")
    ra = np.nanmean(Ma, axis=0); rb = np.nanmean(Mb, axis=0)
    d = 100 * (rb - ra)
    m, sd, nn = d.mean(), d.std(ddof=1), len(d)
    se = sd / math.sqrt(nn); tc = tdist.ppf(0.975, nn - 1)
    pt_p = tdist.sf(abs(m / se), nn - 1) * 2 if se > 0 else float("nan")
    try:
        wp = wilcoxon(rb - ra, zero_method="wilcox").pvalue
    except Exception:
        wp = float("nan")
    worse = int((d < 0).sum()); better = int((d > 0).sum()); tied = int((d == 0).sum())
    print(f"   mean per-config delta = {m:+.1f} points (sd {sd:.1f})")
    print(f"   95% CI  [{m-tc*se:+.1f}, {m+tc*se:+.1f}] points   half-width +/-{tc*se:.1f}")
    print(f"   paired t p = {pt_p:.4f} | Wilcoxon signed-rank p = {wp:.4f}")
    print(f"   {worse} of {nn} configs worse under latency, {better} better, {tied} tied")

    # Variance decomposition of the per-config differences. The observed spread is
    #   Var(true per-config effect) + Var(sampling noise in the two rate estimates).
    # If the true between-config component is ~0 the latency penalty is UNIFORM
    # across configs, and then more SEEDS keep shrinking the difference interval.
    # If it is large, only more CONFIGS help.
    Ra = np.sum(~np.isnan(Ma), axis=0); Rb = np.sum(~np.isnan(Mb), axis=0)
    noise = (ra * (1 - ra) / np.maximum(Ra, 1) + rb * (1 - rb) / np.maximum(Rb, 1))
    obs_var = (sd / 100.0) ** 2
    true_var = max(obs_var - noise.mean(), 0.0)
    print(f"   variance split: observed sd {sd:.1f} pts; sampling-noise sd "
          f"{100*math.sqrt(noise.mean()):.1f} pts; implied TRUE between-config sd "
          f"{100*math.sqrt(true_var):.1f} pts")
    if true_var <= 0.15 * obs_var:
        print("   -> the per-config spread is essentially all sampling noise: the latency")
        print("      penalty looks UNIFORM across configs, so more seeds DO keep tightening")
        print("      the difference (though not the marginal per-arm rates).")
    else:
        print("   -> a real config-dependent component is present: more CONFIGS, not more")
        print("      seeds, are what would tighten the difference.")

    print("\n-- 3. McNEMAR on (seed, config) matched pairs ---------------------------")
    n01, n10, conc, mp, (mlo, mhi) = mcnemar(Ma, Mb)
    print(f"   concordant {conc};  discordant: baseline-only-success {n01}, "
          f"latency-only-success {n10}")
    print(f"   exact McNemar p = {mp:.5f}   95% CI on the paired difference "
          f"[{mlo:+.1f}, {mhi:+.1f}] points")
    print(f"   (pairing on init_rng is weak -- the arms' RNG streams diverge after the")
    print(f"    first inference call -- so treat this as supporting, not primary.)")

    print("\n-- 4. PER-SEED PAIRED --------------------------------------------------")
    sa = np.nanmean(Ma, axis=1); sb = np.nanmean(Mb, axis=1)
    ds = 100 * (sb - sa)
    if len(ds) >= 3:
        mm, ss, nnn = ds.mean(), ds.std(ddof=1), len(ds)
        sse = ss / math.sqrt(nnn); tcc = tdist.ppf(0.975, nnn - 1)
        print(f"   mean per-seed delta = {mm:+.1f} points (sd {ss:.1f}), {nnn} seeds")
        print(f"   95% CI  [{mm-tcc*sse:+.1f}, {mm+tcc*sse:+.1f}] points")
        w = int((ds < 0).sum()); b_ = int((ds > 0).sum())
        print(f"   {w} seeds worse, {b_} better, {int((ds==0).sum())} tied;"
              f" sign test p = {binomtest(b_, w+b_, 0.5).pvalue:.4f}"
              if w + b_ else "   all tied")

    print("\n-- 5. NAIVE POOLED (kept only for continuity with the published number) -")
    fp = fisher_exact([[kb, nb - kb], [ka, na - ka]])[1]
    nlo, nhi = newcombe(kb, nb, ka, na)
    print(f"   Fisher exact two-sided p = {fp:.4f};  Newcombe 95% CI "
          f"[{nlo:+.1f}, {nhi:+.1f}]  half-width +/-{(nhi-nlo)/2:.1f}")
    print(f"   This treats the two arms as independent unpaired samples and ignores")
    print(f"   that they share all {len(cfgs)} episode configurations.")
    return dict(task=task, kb=kb, nb=nb, ka=ka, na=na, boot=(pt, blo, bhi, bp),
                cfg=(m, m - tc * se, m + tc * se, pt_p), mcnemar=(n01, n10, mp))


if __name__ == "__main__":
    pats = sys.argv[1:] or [
        os.path.join(HERE, "..", "..", "runs", "*", "summary.json"),
        os.path.join(HERE, "..", "runs", "*", "summary.json"),
        os.path.join(HERE, "..", "g5", "runs", "*", "summary.json"),
        os.path.join(HERE, "runs", "*", "*", "summary.json"),
        os.path.join(HERE, "validation", "*", "*", "summary.json"),
    ]
    data = collect(load(pats))
    for task in ["widowx_put_eggplant_in_basket", "widowx_spoon_on_towel",
                 "widowx_carrot_on_plate", "widowx_stack_cube"]:
        if task in data:
            analyse(task, data[task], "B", "A")
            if data[task].get("C"):
                analyse(task, data[task], "C", "A",
                        "684.8 ms serial (D=4)", "0 ms baseline (D=0)")
