#!/usr/bin/env python3
"""Recompute every success count and rate contrast of the rate-injected envelope from the CSVs.

Run this before quoting the envelope. It reproduces the per-rate success fractions and tests every pairwise
contrast with Fisher's exact test, on both gate courses, so each annotation can be checked against the
flights behind it. Two contrasts matter and they are not the same: the step from below the control-rate
floor to above it (25 -> 50 Hz), and the step above the floor (50 -> 100 Hz). The floor contrast is the claim
the figure makes; this script exits non-zero when that contrast is not positive and significant on course A,
and writes every contrast it computed to results/codesign_feedback/refined/audit_showdown_claims_metrics.json.

Scope: this grid holds moment_scale fixed at 0.0055, a gain calibrated for a 50 Hz loop, so control authority
varies with rate across it and the two are not separated here.
"""
import collections, itertools, json, os, sys
from scipy.stats import fisher_exact

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# the same reader every other consumer of a campaign CSV uses: a simulator-fault batch is not a
# flight, and a contrast computed over a population the figures do not draw is not their contrast
from flight_quarantine import flight_rows        # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = {}
ok = True

def load(p):
    rows = list(flight_rows(p))
    by = collections.defaultdict(lambda: [0, 0])
    for r in rows:
        hz = round(float(r["eff_cmd_hz"]))
        by[hz][1] += 1
        by[hz][0] += (r.get("outcome") == "success")
    return by, len(rows)

for name, label in (("hil_ablation.csv", "COURSE A"), ("hil_ablation_courseB.csv", "COURSE B")):
    by, n = load(os.path.join(REPO, f"results/codesign_feedback/{name}"))
    print(f"\n{'='*74}\n{label}  ({n} flights)\n{'='*74}")
    for hz in sorted(by):
        s, t = by[hz]
        print(f"  {hz:>4} Hz   {s:>3}/{t:<4} = {s/t:.3f}")
    print(f"\n  {'comparison':<22} {'delta pts':>10} {'Fisher p':>12}   verdict")
    for a, b in itertools.combinations(sorted(by), 2):
        sa, ta = by[a]; sb, tb = by[b]
        d = (sb/tb - sa/ta) * 100
        _, p = fisher_exact([[sb, tb-sb], [sa, ta-sa]])
        v = "SIGNIFICANT" if p < 0.05 else "not significant"
        print(f"  {a:>3} Hz -> {b:>3} Hz      {d:>+9.1f}  {p:>12.2e}   {v}")
        OUT[f"{label}:{a}->{b}"] = {"from": [sa, ta], "to": [sb, tb], "delta_pts": round(d, 2), "fisher_p": p}
        if label == "COURSE A" and (a, b) == (25, 50) and not (d > 0 and p < 0.05):
            ok = False; print("  FAIL: the floor contrast (25 -> 50 Hz) is not a positive, significant step on course A")

    # the claim the figure makes vs the claim the headline makes
    print()
    if 50 in by and 100 in by:
        s5, t5 = by[50]; s1, t1 = by[100]
        _, p = fisher_exact([[s1, t1-s1], [s5, t5-s5]])
        print(f"  HEADLINE TEST  100 Hz vs 50 Hz: {s1}/{t1}={s1/t1:.3f} vs {s5}/{t5}={s5/t5:.3f}  "
              f"delta {(s1/t1-s5/t5)*100:+.1f} pts, p={p:.3f}")
    # pooled: under-rate (25,33) vs feasible (50,100) -- what the shaded bands imply
    lo = [sum(by[h][i] for h in by if h < 42) for i in (0, 1)]
    hi = [sum(by[h][i] for h in by if h >= 42) for i in (0, 1)]
    _, p = fisher_exact([[hi[0], hi[1]-hi[0]], [lo[0], lo[1]-lo[0]]])
    print(f"  POOLED (what the bands imply)  under-rate {lo[0]}/{lo[1]}={lo[0]/lo[1]:.3f} vs "
          f"feasible {hi[0]}/{hi[1]}={hi[0]/hi[1]:.3f}  delta {(hi[0]/hi[1]-lo[0]/lo[1])*100:+.1f} pts, p={p:.2e}")

OUTDIR = os.path.join(REPO, "results/codesign_feedback/refined")
os.makedirs(OUTDIR, exist_ok=True)
json.dump(OUT, open(os.path.join(OUTDIR, "audit_showdown_claims_metrics.json"), "w"), indent=1)
print("\nFLOOR CONTRAST", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
