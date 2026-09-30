# Octo end-to-end schedule study

Closed-loop evaluation of how a VLA policy's **execution schedule** on a QRB5165 changes
what a robot can actually do. 44 distinct scheduler operating points x 4 SIMPLER tasks x
20 seeds x 24 episodes = **84,480 episodes**.

```bash
./run_all.sh          # regenerates every figure that reproduces from committed data
```

No GPU, no network, no run tree needed for that — the inputs are the small JSON/TSV in
`data/`. Everything heavier is documented in `docs/REPRODUCE.md`.

## The result

| task | embodiment | success range across the plane | in its own noise band |
|---|---|---|---|
| eggplant in basket | widowx, 40 ms actuation | **34.6 pts** (19.4 → 54.0%) | 10.4x |
| spoon on towel | widowx, 40 ms | **27.3 pts** (15.4 → 42.7%) | 11.9x |
| pick coke can | google, 333 ms | 11.9 pts (34.6 → 46.5%) | 5.7x |
| close drawer | google, 333 ms | 8.3 pts (34.4 → 42.7%) | 4.0x |

**Cadence dominates latency, and only where the actuator can consume it.** Four of six
pre-specified trend tests survive Benjamini-Hochberg at q=0.05: both cadence ladders and
both latency ladders on the two widowx tasks. Neither google task survives — their trend
intervals span zero.

That split is **actuator saturation, not task class**. On google_robot every schedule from
111 to 283 ms cadence outruns the 333 ms control grid, so a faster schedule produces
results the robot cannot consume. Refining the same drawer task to a 27 Hz grid makes the
effect appear: rho = -0.727 [-0.835, -0.618] with 10 of 10 seeds agreeing, against
rho = -0.241 [-0.602, +0.119] at the native rate. Same controller, same seeds; control
rate is the only variable.

## Layout

| dir | what |
|---|---|
| `harness/` | `finegrain_eval.py` (closed-loop sim under a modelled schedule), `trace_eval.py` (adds per-tick force/velocity/pose logging), `replay_frame.py` (deterministic action replay) |
| `sweep/` | per-cell job scripts and the fleet fetch/wait drivers |
| `analysis/` | trend tests + noise floors (`analyze_merged.py`), tick-grid audit, success-table and energy-cache builders |
| `figures/` | every `fig_*.py` / `make_*.py` |
| `data/` | the committed inputs: grid statuses, per-cell success, energy scalars, arm tables |
| `docs/` | mechanism write-ups and the reproduction tiers |

## Method notes that matter

**The plane cannot be fully filled by measurement, and that is a property of the schedule
space.** Of 108 (period, window) cells, 25 are proven INFEASIBLE (no schedule exists), 12
are UNKNOWN after a 3600 s CP-SAT pass (a budget outcome, *not* infeasibility), and 27 are
exact duplicates — once the deadline stops binding CP-SAT returns the same schedule, and
the simulator is given only latency and cadence, so those cells re-run an identical
command. Duplicates are marked `propagated` in `data/grid_e2e_success.json` and hatched in
the heatmap; they never enter a statistic. `figures/fig_e2e_measured.py` plots the 44
measured points in achieved coordinates, where no propagation is involved at all.

**Noise floors are empirical, per embodiment, and never transferred.** The harness is not
run-to-run deterministic (`docs/NONDETERMINISM.md`). Same-experiment run pairs give
~80% bit-identical and a rare-jump tail, so a sigma multiplier understates it; the band is
a sign-symmetrised bootstrap percentile of the paired mean difference at the seed count
the contrasts actually average. 20-seed bands: egg ±3.33, spoon ±2.29, coke ±2.08.
Drawer currently borrows coke's — same embodiment, same grid — and is flagged provisional.

**Trend before endpoint.** Because noise is a rare-jump process, one pairwise difference is
weak evidence while a monotone ordering across a ladder is strong. Every ladder reports a
per-seed Spearman with a paired-t interval and the -/+ seed counts, and multiplicity is
controlled across the pre-specified set.

**google_robot cannot use the widowx fine-tick model**, and `docs/GOOGLE_FINE_MECHANISMS.md`
derives why: its planner's path duration goes as `t = 2*sqrt(d/a)`, so subdividing the
control period by N shrinks the period by N but the plan only by sqrt(N) — executed
fraction falls from ~75% at 3 Hz to ~25% at 27 Hz. Four independent routes were tested and
each failed at its own native-rate control.

## Not in this repo

`traces_torque2/` (688 MB per-tick traces), `g5grid/runs/` (133 MB summaries) and
`runs_g40/` (151 MB). `docs/REPRODUCE.md` gives the tiers and the job scripts that
regenerate them. `analysis/make_energy_cache.py` is the pattern for the rest: it reduces
702 MB of traces to a 39 KB scalar cache, verified equal to the full computation at
rtol 1e-12, so `fig_energy.py` reproduces from the repo alone.
