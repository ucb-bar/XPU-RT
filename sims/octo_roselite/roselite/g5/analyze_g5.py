"""Large-n replication of the RoSE-lite headline comparison.

Arm A  0 ms baseline, ensembler ON.
Arm B  283.4 ms MEASURED QRB5165 latency, pipelined schedule, D=2.

Reads latency_eval.py summary.json files, reports per-seed counts (so the
pooling is auditable), pooled rates with Clopper-Pearson intervals, and a
two-sided Fisher exact test of B vs A.
"""
import glob, json, os, sys
from scipy.stats import fisher_exact, beta, binomtest

HERE = os.path.dirname(os.path.abspath(__file__))


def ci(k, n, alpha=0.05):
    """Clopper-Pearson exact binomial interval."""
    lo = 0.0 if k == 0 else beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - alpha / 2, k + 1, n - k)
    return 100 * lo, 100 * hi


def load(pattern):
    out = []
    for f in sorted(glob.glob(pattern)):
        s = json.load(open(f))
        out.append(s)
    return out


def arm_rows(summaries, want_lat, want_pipe, want_ens):
    rows = []
    for s in summaries:
        if (abs(s["latency_ms"] - want_lat) < 1e-6 and s["pipeline"] == want_pipe
                and s["ensemble"] == want_ens and s["task"] == "widowx_put_eggplant_in_basket"):
            rows.append((s["init_rng"], s["n_success"], s["n_episodes"], s["wall_s"]))
    # de-duplicate on seed, keep first
    seen, ded = set(), []
    for r in sorted(rows):
        if r[0] in seen:
            continue
        seen.add(r[0]); ded.append(r)
    return ded


def report(name, rows):
    k = sum(r[1] for r in rows); n = sum(r[2] for r in rows)
    print(f"\n{name}")
    print("  seed   ok/n     SR%")
    for rng, ok, ne, _w in rows:
        print(f"  {rng:>4}  {ok:>2}/{ne:<3}  {100*ok/ne:5.1f}")
    lo, hi = ci(k, n)
    print(f"  POOLED {k:>3}/{n:<4} {100*k/n:5.1f}%   95% CI (Clopper-Pearson) [{lo:.1f}, {hi:.1f}]")
    return k, n


if __name__ == "__main__":
    pats = sys.argv[1:] or [os.path.join(HERE, "runs", "*", "summary.json")]
    S = []
    for p in pats:
        S += load(p)
    print(f"loaded {len(S)} summary.json files from {pats}")

    A = arm_rows(S, 0.0, False, "stock")     # baseline, ensembler ON
    B = arm_rows(S, 283.4, True, "none")     # 283.4 ms pipelined, D=2

    print("=" * 88)
    print("MEASURED success on widowx_put_eggplant_in_basket, octo-small-1.0,")
    print("corrected masked unnormalization, official visual-matching protocol (24 ep/seed)")
    print("=" * 88)
    ka, na = report("ARM A -- 0 ms baseline, ensembler ON", A)
    kb, nb = report("ARM B -- 283.4 ms MEASURED, pipelined, D=2", B)

    if na and nb:
        p = fisher_exact([[kb, nb - kb], [ka, na - ka]])[1]
        d = 100 * (kb / nb - ka / na)
        print("\n" + "=" * 88)
        print("Fisher exact, two-sided:  283.4 ms pipelined  vs  0 ms baseline")
        print(f"  {kb}/{nb} = {100*kb/nb:.1f}%   vs   {ka}/{na} = {100*ka/na:.1f}%")
        print(f"  difference = {d:+.1f} points     p = {p:.3f}"
              f"   {'n.s.' if p >= 0.05 else 'SIGNIFICANT'}")
        # Newcombe hybrid-score interval on the difference of proportions
        l1, u1 = [x / 100 for x in ci(kb, nb)]
        l2, u2 = [x / 100 for x in ci(ka, na)]
        p1, p2 = kb / nb, ka / na
        dlo = (p1 - p2) - ((p1 - l1) ** 2 + (u2 - p2) ** 2) ** 0.5
        dhi = (p1 - p2) + ((u1 - p1) ** 2 + (p2 - l2) ** 2) ** 0.5
        print(f"  95% CI on the difference (Newcombe): [{100*dlo:+.1f}, {100*dhi:+.1f}] points")
        print("=" * 88)

        # --- paired-by-seed view: the two arms share init_rng and episode ids,
        #     so seeds pair up and a sign test uses that pairing. ---
        da = dict((r[0], (r[1], r[2])) for r in A)
        db = dict((r[0], (r[1], r[2])) for r in B)
        both = sorted(set(da) & set(db))
        if both:
            print("\nPAIRED BY SEED (same init_rng and episode ids in both arms)")
            print("  seed   A ok/n   B ok/n   B-A")
            wins = losses = ties = 0
            for s_ in both:
                a_ok, a_n = da[s_]; b_ok, b_n = db[s_]
                d_ = b_ok - a_ok
                wins += d_ > 0; losses += d_ < 0; ties += d_ == 0
                print(f"  {s_:>4}   {a_ok:>2}/{a_n:<3}   {b_ok:>2}/{b_n:<3}  {d_:+3d}")
            nz = wins + losses
            sign_p = (binomtest(wins, nz, 0.5).pvalue if nz else 1.0)
            print(f"  B beat A in {wins}/{nz} non-tied seeds ({ties} ties); "
                  f"sign test two-sided p = {sign_p:.3f}")

        # --- pooled with the local reference run (seeds 0/2/4, n=72 per arm) ---
        # local per-seed counts (audited from the local summary.json files):
        #   ARM A stock free-running baseline, ensembler ON : rng0 13, rng2 15, rng4 10
        #   ARM B 283.4 ms pipelined D=2                    : rng0  9, rng2 16, rng4 16
        LOC_A_SEEDS = {0: 13, 2: 15, 4: 10}
        LOC_B_SEEDS = {0: 9, 2: 16, 4: 16}
        LOC_A, LOC_NA = sum(LOC_A_SEEDS.values()), 24 * len(LOC_A_SEEDS)
        LOC_B, LOC_NB = sum(LOC_B_SEEDS.values()), 24 * len(LOC_B_SEEDS)
        print("\n  local per-seed (audit): "
              f"A {{k: v for k, v in LOC_A_SEEDS.items()}} = {LOC_A}/{LOC_NA} ; "
              f"B {{k: v for k, v in LOC_B_SEEDS.items()}} = {LOC_B}/{LOC_NB}"
              .replace("{k: v for k, v in LOC_A_SEEDS.items()}", str(LOC_A_SEEDS))
              .replace("{k: v for k, v in LOC_B_SEEDS.items()}", str(LOC_B_SEEDS)))
        pa, pna = ka + LOC_A, na + LOC_NA
        pb, pnb = kb + LOC_B, nb + LOC_NB
        pp = fisher_exact([[pb, pnb - pb], [pa, pna - pa]])[1]
        la, ha = ci(pa, pna); lb, hb = ci(pb, pnb)
        print("\nPOOLED WITH THE LOCAL REFERENCE (local seeds 0/2/4, n=72/arm)")
        print(f"  ARM A baseline           {pa}/{pna} = {100*pa/pna:.1f}%   95% CI [{la:.1f}, {ha:.1f}]")
        print(f"  ARM B 283.4 ms pipelined {pb}/{pnb} = {100*pb/pnb:.1f}%   95% CI [{lb:.1f}, {hb:.1f}]")
        print(f"  difference = {100*(pb/pnb - pa/pna):+.1f} points     Fisher p = {pp:.3f}"
              f"   {'n.s.' if pp >= 0.05 else 'SIGNIFICANT'}")
        p1, p2 = pb / pnb, pa / pna
        l1, u1 = lb / 100, hb / 100
        l2, u2 = la / 100, ha / 100
        dlo2 = (p1 - p2) - ((p1 - l1) ** 2 + (u2 - p2) ** 2) ** 0.5
        dhi2 = (p1 - p2) + ((u1 - p1) ** 2 + (p2 - l2) ** 2) ** 0.5
        print(f"  95% CI on the difference (Newcombe): [{100*dlo2:+.1f}, {100*dhi2:+.1f}] points")
        print("=" * 88)
