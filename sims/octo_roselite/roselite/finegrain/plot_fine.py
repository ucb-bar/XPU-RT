"""Success-rate-vs-latency at fine granularity, coarse 5 Hz overlay."""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
rows = [r for r in json.load(open(os.path.join(ROOT, "curve_fine.json")))
        if r["prefix"] != "noscale_lat0"]
rows = sorted(rows, key=lambda r: r["latency"])

x = [r["latency"] for r in rows]
y = [r["sr"] for r in rows]
lo = [r["sr"] - r["lo"] for r in rows]
hi = [r["hi"] - r["sr"] for r in rows]

# coarse 5 Hz results (RESULTS.txt, MEASURED-in-sim under the coarse model)
coarse_serial = [(0.0, 14.2), (283.4, 15.8), (555.0, 9.7), (684.8, 3.3)]
coarse_pipe = [(0.0, 62.5), (283.4, 56.9), (555.0, 20.8), (684.8, 0.0)]

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14, 5.6),
                             gridspec_kw={"width_ratios": [1.35, 1]})
ax.axhline(52.8, color="0.4", ls="--", lw=1,
           label="5 Hz stock baseline 38/72 = 52.8% (MEASURED)")
ax.axhspan(41.4, 63.9, color="0.85", zorder=0)
ax.errorbar(x, y, yerr=[lo, hi], fmt="o-", color="#1f77b4", lw=2, ms=7, capsize=4,
            label="fine-grain 40 ms tick + ZOH (this work)")
ax.plot(*zip(*coarse_pipe), "s--", color="#ff7f0e", ms=5, alpha=0.8,
        label="coarse 5 Hz, pipelined (previous)")
ax.plot(*zip(*coarse_serial), "^--", color="#d62728", ms=5, alpha=0.8,
        label="coarse 5 Hz, serial (previous)")
for r in rows:
    ax.annotate(f"{r['prefix']}\n{r['stale_max']} ticks stale",
                (r["latency"], r["sr"]), textcoords="offset points",
                xytext=(6, 8), fontsize=7.5, color="#1f77b4")
ax.set_xlabel("MEASURED QRB5165 per-inference latency (ms)")
ax.set_ylabel("success rate (%)  widowx_put_eggplant_in_basket, n=72")
ax.set_title("Octo in SIMPLER under MODELLED QRB5165 latency\n"
             "fine 40 ms control tick with zero-order hold vs the coarse 200 ms model")
ax.set_ylim(-3, 80)
ax.grid(alpha=0.3)
ax.legend(fontsize=8, loc="upper right")
# ---- right panel: observation age at actuation ----
xs = [r["latency"] for r in rows]
ax2.plot(xs, [r["age_mean"] for r in rows], "o-", color="#2b7", lw=2, ms=6,
         label="mean age at actuation (MEASURED-in-sim)")
ax2.plot(xs, [r["age_max"] for r in rows], "s--", color="#c62", lw=1.6, ms=5,
         label="max age at actuation")
ax2.plot(xs, xs, ":", color="0.5", lw=1.4, label="raw board latency (MEASURED)")
for r in rows:
    ax2.annotate(f"{r['age_mean']:.0f}", (r["latency"], r["age_mean"]),
                 textcoords="offset points", xytext=(5, -12), fontsize=7.5, color="#2b7")
ax2.set_xlabel("MEASURED QRB5165 per-inference latency (ms)")
ax2.set_ylabel("observation age when the action is applied (ms)")
ax2.set_title("Real sensor->actuation age\n(compute latency + zero-order hold)")
ax2.grid(alpha=0.3)
ax2.legend(fontsize=8, loc="upper left")

fig.tight_layout()
out = os.path.join(ROOT, "success_vs_latency_finegrain.png")
fig.savefig(out, dpi=150)
print("->", out)
