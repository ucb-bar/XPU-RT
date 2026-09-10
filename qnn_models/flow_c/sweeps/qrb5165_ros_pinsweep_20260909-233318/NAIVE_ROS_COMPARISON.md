# XPU-RT against the ROS deployment a team would actually write

Every workload cell run twice on the same QRB5165, both timed from their own first
dispatch: once under XPU-RT's per-operation scheduler solving with `cpsat:warmbest`,
and once as ordinary ROS 2 nodes with each network pinned whole to the backend it is
fastest on **in isolation**. No placement search on the ROS side — that is the point.

The ratio is **ROS ÷ XPU-RT** on the non-periodic makespan, so **above 1.00 the
scheduler is ahead**. Medians of 3 reps.

## Where it lands

| scope | cells | median | scheduler ahead | pinning ahead | within noise |
|---|---:|---:|---:|---:|---:|
| `wl_sweep`, quad (the real machine) | 7 | **1.1795** | 6 | 1 | 2 |
| `wl_sweep`, all lane subsets | 26 | **1.0768** | 15 | 11 | 6 |
| `3net` | 7 | **0.9488** | 2 | 5 | 3 |
| **Every comparable cell** | 33 | **1.0235** | 17 | 16 | 9 |

A cell is comparable only if it has aperiodic work to time and both runtimes ran the
same number of instances of it — **33 of 51** qualify.

## What each side is

**XPU-RT** places individual operations across HTA, Hexagon DSP, CPU and GPU lanes and
gates each dispatch to a scheduled start. The schedule comes from `cpsat:warmbest` —
CP-SAT warm-started from the best feasible heuristic — which the solver study recommends
for the build-time path.

**Naive ROS** is one node per network, one executor, the whole dispatch graph on one
backend, chosen by that network's own cost in isolation. No search, no coordination.
An earlier version of this comparison let ROS pick the best of every legal placement;
that is a placement oracle nobody has, and removing it reversed the conclusion.

## Every cell

Rows marked *excluded* are not part of any median above. `±9.18%` is this board's
rep-to-rep spread on the objective; a difference inside it is a direction, not a result.

| cell | lanes | ROS ms | XPU-RT ms | ratio | |
|---|---|---:|---:|---:|---|
| **wl_sweep · cg** | *cpu + gpu* | | | | |
| `bimodal_cg` | cg | 19.19 | 20.68 | **0.928** | within noise |
| `control_mix_cg` | cg | 4.72 | 5.86 | **0.804** | pinning |
| `depth_chain_cg` | cg | 19.03 | 14.74 | 1.291 | *excluded — no aperiodic work, unequal timed work* |
| `depth_contended_cg` | cg | 8.71 | 5.88 | **1.483** | scheduler |
| `depth_nav_cg` | cg | 26.89 | 27.45 | 0.979 | *excluded — no aperiodic work, unequal timed work* |
| `perception_heavy_cg` | cg | 10.36 | 12.02 | **0.862** | pinning |
| `saturation_cg` | cg | 27.75 | 20.32 | 1.365 | *excluded — unequal timed work* |
| `scale_ladder_cg` | cg | 22.46 | 8.84 | **2.540** | scheduler |
| `tight_loop_cg` | cg | 27.34 | 25.90 | 1.056 | *excluded — no aperiodic work, unequal timed work* |
| `vint_intro_cg` | cg | 125.76 | 99.92 | **1.259** | scheduler |
| `vint_multi_cg` | cg | 146.23 | 100.03 | **1.462** | scheduler |
| **wl_sweep · dc** | *dsp + cpu* | | | | |
| `bimodal_dc` | dc | 6.75 | 6.88 | **0.981** | within noise |
| `control_mix_dc` | dc | 3.93 | 3.48 | **1.129** | scheduler |
| `depth_chain_dc` | dc | 12.55 | 12.74 | 0.985 | *excluded — no aperiodic work, unequal timed work* |
| `depth_contended_dc` | dc | 5.88 | 3.35 | **1.754** | scheduler |
| `depth_nav_dc` | dc | 31.51 | 31.41 | 1.003 | *excluded — no aperiodic work, unequal timed work* |
| `perception_heavy_dc` | dc | 4.95 | 4.83 | **1.024** | within noise |
| `saturation_dc` | dc | 9.10 | 4.95 | 1.836 | *excluded — unequal timed work* |
| `scale_ladder_dc` | dc | 3.41 | 6.07 | **0.562** | pinning |
| `tight_loop_dc` | dc | 23.59 | 23.48 | 1.005 | *excluded — no aperiodic work, unequal timed work* |
| `vint_intro_dc` | dc | 118.62 | 43.41 | **2.732** | scheduler |
| `vint_multi_dc` | dc | 110.62 | 44.87 | **2.466** | scheduler |
| **wl_sweep · hd** | *hta + dsp* | | | | |
| `bimodal_hd` | hd | 7.00 | 10.20 | **0.687** | pinning |
| `control_mix_hd` | hd | 4.60 | 7.84 | **0.587** | pinning |
| `depth_chain_hd` | hd | 12.60 | 12.25 | 1.029 | *excluded — no aperiodic work, unequal timed work* |
| `depth_contended_hd` | hd | 3.60 | 3.81 | **0.945** | within noise |
| `depth_nav_hd` | hd | 31.93 | 33.10 | 0.965 | *excluded — no aperiodic work, unequal timed work* |
| `perception_heavy_hd` | hd | 4.92 | 6.89 | **0.715** | pinning |
| `saturation_hd` | hd | 9.89 | 17.39 | 0.569 | *excluded — unequal timed work* |
| `scale_ladder_hd` | hd | 3.41 | 6.45 | **0.529** | pinning |
| `tight_loop_hd` | hd | 47.44 | 48.14 | 0.986 | *excluded — no aperiodic work, unequal timed work* |
| **wl_sweep · quad** | *hta + dsp + cpu + gpu* | | | | |
| `bimodal_quad` | quad | 6.69 | 6.72 | **0.995** | within noise |
| `control_mix_quad` | quad | 3.96 | 3.36 | **1.179** | scheduler |
| `depth_chain_quad` | quad | 12.61 | 12.30 | 1.026 | *excluded — no aperiodic work, unequal timed work* |
| `depth_contended_quad` | quad | 5.97 | 4.22 | **1.416** | scheduler |
| `depth_nav_quad` | quad | 31.58 | 31.94 | 0.989 | *excluded — no aperiodic work, unequal timed work* |
| `perception_heavy_quad` | quad | 5.02 | 4.85 | **1.035** | within noise |
| `saturation_quad` | quad | 8.79 | 4.28 | 2.054 | *excluded — unequal timed work* |
| `scale_ladder_quad` | quad | 3.43 | 3.06 | **1.118** | scheduler |
| `tight_loop_quad` | quad | 47.44 | 49.48 | 0.959 | *excluded — no aperiodic work, unequal timed work* |
| `vint_intro_quad` | quad | 121.03 | 30.46 | **3.973** | scheduler |
| `vint_multi_quad` | quad | 126.51 | 30.45 | **4.155** | scheduler |
| **3net · all lanes** | *hta + dsp + cpu* | | | | |
| `3net_dronet1_mlp2` | all lanes | 10.23 | 9.98 | 1.026 | *excluded — no aperiodic work* |
| `3net_dronet1_mlp2_yolo1` | all lanes | 26.77 | 30.23 | **0.886** | pinning |
| `3net_dronet2_mlp8_yolo1` | all lanes | 28.63 | 30.18 | **0.949** | within noise |
| `3net_dronet4_mlp4_yolo1` | all lanes | 27.01 | 31.14 | **0.867** | pinning |
| `3net_dronet4_mlp8_yolo1` | all lanes | 28.90 | 30.50 | **0.948** | within noise |
| `3net_dronet8_mlp16_yolo1` | all lanes | 34.97 | 30.57 | **1.144** | scheduler |
| `3net_fused2_mlp8_yolo1` | all lanes | 34.78 | 29.81 | **1.167** | scheduler |
| `3net_fused4_mlp4_yolo1` | all lanes | 28.95 | 29.37 | **0.986** | within noise |
| `3net_mlp2` | all lanes | 10.16 | 8.09 | 1.256 | *excluded — no aperiodic work* |

## Why the two arms disagree

On `wl_sweep` at quad the naive rule costs a median **1.17×** against the best placement
available, and up to **1.86×** on `depth_contended_quad`, where all three networks prefer
the DSP alone and all three land there. On `3net` the same rule is nearly optimal — a
median gap of **1.018×** — because those networks spread naturally across lanes.

**So whether naive pinning is good enough is not a property of the scheduler. It is a
property of whether your networks contend for the same preferred lane.**

The sharpest case is `3net_fused2_mlp8_yolo1`, where both placements run the timed
`yolov8n` on the DSP and the naive one is still 17% slower:

```
naive    mlp_control@cpu   24 inst x 0.127 ms    yolov8n@dsp  34.676 ms
oracle   mlp_control@dsp   24 inst x 0.580 ms    yolov8n@dsp  28.811 ms
```

Moving `mlp_control` to the DSP makes mlp itself 4.6× slower per instance and the timed
network 17% faster, because QnnCpu builds its thread pool with full-machine affinity and
CPU work starves the DSP lane's host-side driver thread. **An isolation cost model cannot
see this**: `mlp_control` genuinely is faster on the CPU alone, 110.2 µs against 523.6.
The naive rule picks correctly by its own criterion and still loses, because the cost
lands on a different network on a different lane.

## Read with these

* **The band is wide.** 9 of 33 comparable cells sit inside
  ±9.18%. Those are directions, not results.
* **Only `quad` is a real machine.** A QRB5165 always has all four backends; `hd`, `dc`
  and `cg` model lane subsets you cannot buy and distort individual cells badly — in `hd`
  there is no CPU lane, so `mlp_control_sd` is forced onto the DSP at 523.6 µs though CPU
  runs it in 110.2. They are kept as a lane-scarcity sensitivity study, not the headline.
* **`scale_ladder_quad` is sampled, not enumerated** — 64 of 729 legal placements.
  Widening that sample already improved the best known placement by 10.7%, which bounds
  how wrong the sampled cells can be.
* **Both sides are offset-corrected.** Each run is timed from its own first dispatch, so
  neither runtime is charged for the other's start barrier. Raw and corrected figures are
  both in `results/analysis.json`; only `perception_heavy` moves more than the band.

---

Generated from `results/analysis.json` and `results/analysis_3net.json` by
`scripts/mk_naive_comparison.py`. Performance governor, one tenant at a time behind the
board lock. Full method and corrections: `SETUP.md` and `ANALYSIS.md` in this directory.
