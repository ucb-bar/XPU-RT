# Actuator energy vs. policy schedule — 4 tasks x 9 arms

Data: `traces_torque2/` — 36 cells, 24 episodes each, `init_rng=100`, produced by
`g5fine/job_torque2.sh` on the 6 g5 workers. Figure: `paper/fig_energy.png`
(script `paper/fig_energy.py`).

Supersedes `traces_torque/`, which covered only eggplant+drawer and carried the OLD
mislabelled pipelined latencies (`pipe110` modelled at 117.7 ms when its true
per-inference span is 385.1 ms — a 3.3x understatement). See
`archive/2026-09-06_pipelined-latency-mislabelled/`.

## The model

| term | definition | what it sees |
|---|---|---|
| `∫Στᵢ²dt` | joint forces from the sim's own `get_qf()`, summed over joints, integrated over the episode | copper loss (I²R) proxy — **the only term that sees a stall or collision**, because a servo pushing into a rigid constraint has large τ and ~zero ω |
| `∫Σω_arm²dt` | joint velocities, **arm joints only** | motion effort / thrashing |

Fingers excluded (damped at 8.0 vs 275–1060 for the arm, and near-massless — unweighted
they dominate by ~100x while doing no mechanical work). google_robot's 2 head joints
excluded for the same reason. Ratios between arms are meaningful; **absolute joules are
not** — there is no motor constant in the model.

Both integrate over the **whole episode**, so an arm that succeeds quickly is charged
less than one that flails to the horizon. Deliberate: the question was total mission
energy, not power. Mission time is plotted separately in `paper/fig_metrics.png`.

**Unconditioned on success**, deliberately. Eggplant `cpu685` succeeds 4.2% of the time,
so conditioning on success keeps the 1-in-24 episode that behaved like the ideal arm and
discards all the flailing — which is the entire effect.

Not modelled: contact impulses (τ² is a proxy, and `get_qf()` is sampled at tick
boundaries so brief impacts are under-counted), drivetrain and electrical losses.

## The finding: the payoff from a faster schedule is capped by the robot's control rate

The two embodiments have different native control grids, and that — not grasp-vs-push —
is what separates the rows.

| embodiment | native control | grid | inferences per issued command, across the 9 arms |
|---|---|---|---|
| widowx | 25 Hz (fine actuation) | 40 ms | **0.36 → 0.06** — never saturated; each result is held for 3–18 commands |
| google_robot | 3 Hz (`--actuation native`) | 333 ms | **2.98 → 0.49** — saturated by every arm from 111 to 283 ms cadence |

**WidowX: cost tracks cadence across the whole range.** Spoon `∫Στ²dt` is monotone in
cadence: 1.00 → 1.38 → 1.37 → 1.28 → 1.22 → 1.42 → 1.62 → 2.03 → **2.59x**. Motion
effort reaches 3.74x (spoon) and 5.29x (eggplant) at `cpu685`, both at **0% success** —
the arm is burning five times the effort to fail.

**google_robot: flat across the entire saturated band.** Every arm from `pipe110`
(111 ms cadence) to `serial283` (283 ms) issues ≥1 inference per command — `pipe110`
computes **2.98 inferences per command and discards two of every three**. Across that
band the observation age still varies 307–411 ms, yet both energy metrics sit within
±25% of the ideal arm and success is flat within the n=24 noise band. Only `fp32_555`
and `cpu685`, which fall *below* the grid (0.60 / 0.49 inferences per command, age
780–1010 ms), cost more — and there success does collapse (coke 33% → 12%).

On a 3 Hz arm **the ideal 0 ms schedule is also the cheapest**: it computes exactly
1.00 inferences per command. Every scheduled arm buys 1.2–3.0x the accelerator energy
for a command stream the robot cannot distinguish.

### Consequences

1. **This reinterprets the earlier "grasp vs push" boundary.** `close_drawer` being flat
   was read as "pushing tolerates latency". But `pick_coke_can` — a grasping task — is
   equally flat, and it shares an embodiment with the drawer, not a task class. The
   controlling variable is the 333 ms control grid, not the manipulation primitive.
2. **It ranks the earlier per-task effect sizes.** Cadence 150 vs 283 ms gave +35.4
   points on eggplant (widowx, unsaturated), +12.9 on spoon (widowx), but only +7.4 on
   coke and −0.8 on drawer (both google, both saturated). Saturation predicts the google
   tasks respond *weakly*, not that they respond zero — see point 3 for the two channels
   that stay open.
3. **Co-design number — but scoped to energy.** Below the actuation period, extra
   cadence buys no extra *actuations* and no measurable change in either energy metric.
   It does NOT follow that it buys nothing, and two channels stay open:

   - **Residual staleness.** A result waits ~C/2 in the hold buffer before the robot can
     consume it, so effective age is `schedule_latency + hold`, not `schedule_latency`.
     Measured on coke: age@act is 353.4 ms at cadence 219 vs 411.4 ms at cadence 283,
     despite schedule latencies of 260.5 and 283.4 ms.
   - **Ensemble depth.** The harness runs stock action ensembling
     (`finegrain_eval.py:566`, `ActionEnsembler(CHUNK, 0.0)`, applied at line 606), so
     *every* inference contributes a chunk to the average whether or not the robot could
     act on it — 2.98 chunks per command at cadence 111 vs 1.17 at cadence 283.

   Both are invisible to `∫Στ²dt` and `∫Σω²dt`, so the correct claim is "**energy** goes
   flat", not "scheduling faster is wasted" — the metrics in this figure cannot see
   either channel.

   As of the wide sweep's coke cells, however, **neither channel has produced a
   demonstrated success effect inside the band**. A −6.7 point drop across cadence
   220→283 at fixed schedule age looked monotone in the pooled means (39.6 → 35.8 →
   32.9%) but fails a per-seed trend test: ρ = −0.11 [−0.77, +0.54], 5 seeds down and 4
   up. Pooled monotonicity from noise. So on google_robot the saturated band is currently
   flat in **both** energy and success, which is what saturation predicts. The widowx
   tasks are where the effect lives, and their cells had not landed at time of writing.

**Caveat.** The wide E2E sweep now running carries explicit cadence ladders at fixed
latency and latency ladders at fixed cadence, at n=10 seeds. Point 2 above is a
prediction from n=24-single-seed energy traces; that sweep is what tests it.

## Table (medians, ratios vs. the ideal 0 ms arm)

```
task    arm         cad/age ms   int-tau2      int-omega2    success  inf/cmd
egg     lat0          -- /   0    82.1 1.00x   16.86 1.00x    37.5%     0.20
egg     pipe110fix   111 / 385    76.5 0.93x   13.84 0.82x    37.5%     0.36
egg     p105w300     125 / 259    76.2 0.93x   23.97 1.42x    50.0%     0.32
egg     p130w275     130 / 272    93.8 1.14x   30.37 1.80x    37.5%     0.31
egg     p150w300     150 / 282    66.7 0.81x   10.29 0.61x    50.0%     0.27
egg     pipe200fix   219 / 260    83.7 1.02x   36.24 2.15x    33.3%     0.18
egg     serial283    283 / 283    97.5 1.19x   25.04 1.49x    29.2%     0.14
egg     fp32_555     555 / 555   102.1 1.24x   83.01 4.92x     0.0%     0.07
egg     cpu685       685 / 685   118.2 1.44x   89.19 5.29x     0.0%     0.06

spoon   lat0          -- /   0    35.8 1.00x    1.98 1.00x    62.5%     0.20
spoon   pipe110fix   111 / 385    49.3 1.38x    1.98 1.00x    33.3%     0.36
spoon   p105w300     125 / 259    49.1 1.37x    1.50 0.76x    33.3%     0.32
spoon   p130w275     130 / 272    45.9 1.28x    1.95 0.99x    33.3%     0.31
spoon   p150w300     150 / 282    43.8 1.22x    1.71 0.86x    45.8%     0.27
spoon   pipe200fix   219 / 260    50.7 1.42x    2.70 1.36x    20.8%     0.18
spoon   serial283    283 / 283    57.9 1.62x    3.89 1.97x    12.5%     0.14
spoon   fp32_555     555 / 555    72.5 2.03x    5.16 2.61x     0.0%     0.07
spoon   cpu685       685 / 685    92.8 2.59x    7.40 3.74x     0.0%     0.06

coke    lat0          -- /   0  7854.3 1.00x    0.99 1.00x    33.3%     1.00
coke    pipe110fix   111 / 385  6292.0 0.80x    0.49 0.50x    41.7%     2.98
coke    p105w300     125 / 259  5958.9 0.76x    0.75 0.76x    50.0%     2.66
coke    p130w275     130 / 272  6814.6 0.87x    0.61 0.62x    37.5%     2.55
coke    p150w300     150 / 282  5668.0 0.72x    0.73 0.74x    45.8%     2.21
coke    pipe200fix   219 / 260  6462.2 0.82x    0.87 0.88x    37.5%     1.52
coke    serial283    283 / 283  6191.6 0.79x    0.90 0.91x    37.5%     1.17
coke    fp32_555     555 / 555  7513.6 0.96x    1.50 1.51x    16.7%     0.60
coke    cpu685       685 / 685  8154.1 1.04x    1.61 1.63x    12.5%     0.49

drawer  lat0          -- /   0  6754.5 1.00x    1.82 1.00x    45.8%     1.00
drawer  pipe110fix   111 / 385  6235.4 0.92x    1.12 0.62x    41.7%     2.98
drawer  p105w300     125 / 259  7916.1 1.17x    2.11 1.16x    41.7%     2.66
drawer  p130w275     130 / 272  7851.3 1.16x    1.46 0.80x    33.3%     2.56
drawer  p150w300     150 / 282  7099.4 1.05x    1.36 0.75x    37.5%     2.21
drawer  pipe200fix   219 / 260  8077.5 1.20x    1.88 1.04x    41.7%     1.52
drawer  serial283    283 / 283  7743.9 1.15x    2.26 1.25x    33.3%     1.17
drawer  fp32_555     555 / 555  6437.1 0.95x    2.32 1.28x    33.3%     0.60
drawer  cpu685       685 / 685  6595.1 0.98x    2.06 1.13x    50.0%     0.49
```

Cadence/age are MEASURED on the QRB5165, ungated. `inf/cmd` = policy inferences produced
per command the robot can actually issue; >1 means surplus results are discarded.
n=24 episodes per cell, ONE seed — success rates here carry a ~±20 point band and are
context for the energy numbers, not a success result. The success result is the closed-
loop sweep.
