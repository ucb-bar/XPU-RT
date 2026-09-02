# Reproducing the solver study plots

Everything under `scripts/solver_study/` regenerates the figures in `plots/`
that compare scheduling solvers. The measurement data is committed alongside
the scripts in `scripts/solver_study/data/`, so **the plots can be redrawn
without re-measuring** — which matters, because the measurements take hours
and two of the solvers are not deterministic.

## Environment

The study runs in the `xpurt` conda env (cvxpy + MOSEK; a MOSEK licence at
`~/mosek/mosek.lic` is needed for the `milp` rows):

```bash
XPURT=/path/to/XPU-RT      # this checkout
PY=/path/to/miniforge3/envs/xpurt/bin/python
```

CP-SAT needs OR-Tools, which cannot be installed into that env — it requires a
newer numpy and protobuf than `xpurt` pins (see
`docs/xpurt_pip_freeze_2026-07-08.txt`). Put it in its own venv and point
`XPURT_CPSAT_PYTHON` at it:

```bash
$PY -m venv /tmp/cpsat-venv
/tmp/cpsat-venv/bin/pip install ortools
export XPURT_CPSAT_PYTHON=/tmp/cpsat-venv/bin/python
```

## Redraw the figures from the committed data (seconds, exact)

```bash
cd "$XPURT"
$PY -c "import sys; sys.path.insert(0,'scripts/solver_study'); import make_plots as m; \
        [print('wrote', p) for p in (m.fig1(), m.fig2(), m.fig3(), m.fig4(), m.fig5(), m.fig6()[0])]"
```

This is byte-reproducible: it reads the JSON/CSV in
`scripts/solver_study/data/` and draws. `fig6` is
`plots/pareto_makespan_vs_time.png`, the makespan-vs-solve-time Pareto chart.

## Re-measure from scratch (hours, and *not* bit-reproducible — see below)

The Pareto chart needs two measurement passes on one fixed instance
(`networks_periodic_dronet50ms_yolov8_firesim_q31`, 242 operations, dronet
pinned to one instance):

```bash
cd "$XPURT"
export XPURT_CPSAT_PYTHON=/tmp/cpsat-venv/bin/python

# 1. Budget ablation: cpsat + the three cvxpy backends at 15/60/120/300/600 s.
#    ~75 minutes. Writes incrementally, so an interrupted run keeps its rows.
$PY scripts/solver_study/time_ablation.py \
    --spec networks_periodic_dronet50ms_yolov8_firesim_q31 \
    --instance-map 'dronet=1' --budgets 15,60,120,300,600 \
    --out scripts/solver_study/data/ablation_q31.json

# 2. The remaining points: the other constructive pickers, PSO/SA at three
#    budgets, and warm-started CP-SAT at four. ~10 minutes.
$PY scripts/solver_study/pareto_fill.py

# 3. Redraw.
$PY -c "import sys; sys.path.insert(0,'scripts/solver_study'); import make_plots as m; print(m.fig6()[0])"
```

## What is and isn't deterministic

| component | reproducible? | why |
| --- | --- | --- |
| the plotting itself | **yes, exactly** | pure function of the committed data files |
| `greedy*`, `decomposed`, `heft`, `heft_edf` | **yes** | deterministic construction, no RNG |
| `pso`, `sa` | **yes, given the seed** | seeded `np.random.default_rng(seed)`; `pareto_fill.py` fixes `seed=0` |
| `milp:*` | close | MOSEK/HiGHS are deterministic given the model, but a *time limit* truncates the search at a wall-clock point, so a loaded machine can change which incumbent is returned |
| **`cpsat`** | **no** | see below |

**CP-SAT is not reproducible as run here.** It defaults to 8 search workers,
and the answer depends on how those threads interleave. Repeated runs of an
identical configuration on the 242-op instance returned 45.33, 45.40, 46.91
and 46.95 ms — a spread of about ±1.5 ms on a 46 ms schedule. Two consequences
for reading the chart:

- Differences below ~1.5 ms between CP-SAT points are noise. The frontier's
  last step (`cpsat-warm@60s` 45.00 → `cpsat-warm@300s` 44.98) is inside that
  band and should not be read as an improvement.
- `cpsat-warm@120s` (45.32) being *worse* than `cpsat-warm@60s` (45.00) is the
  same noise landing the other way, not evidence that more time hurts.

For a bit-reproducible CP-SAT run, pass `workers=1` — `cpsat_schedule` takes
`workers` and `random_seed`, and a single-worker solve with a fixed seed is
deterministic. It is also meaningfully weaker, so the committed numbers use
the 8-worker default that anyone would actually run.

## What the figures are

| file | what it shows |
| --- | --- |
| `plots/pareto_makespan_vs_time.png` | makespan vs solve time, all solvers and budgets, Pareto frontier highlighted (`fig6`) |
| `plots/spike3_makespan_comparison.png` | makespan by method on the spike 3-model workload (`fig1`) |
| `plots/spike3_quality_vs_cost.png` | the same instance as quality-against-cost (`fig2`) |
| `plots/solver_backends_by_size.png` | cvxpy backends across three instance sizes (`fig3`) |
| `plots/spike3_lane_occupancy.png` | which lane each schedule uses over time (`fig4`) |
| `plots/solver_validity_by_target.png` | share of workloads with a valid schedule, split by whether the target has a Gemmini lane (`fig5`) |

`docs/scheduler_solver_study.md` is the write-up these figures illustrate.
