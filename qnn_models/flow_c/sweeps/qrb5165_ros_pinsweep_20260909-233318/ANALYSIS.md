# ROS 2 whole-network-pinning baseline over `sched_algo_sweep10` — QRB5165 — RESULTS

Written after the campaign. `SETUP.md` is the contract and is **not** revised;
everything that turned out differently from what it assumed is in §0.

Everything below is measured. 256 assignments, 3 reps each, 768 board runs,
all complete, none discarded.

---

## 0. Corrections to SETUP.md

**0.1 — The headline objective changed, and the headline number with it.**
SETUP.md §2 framed the deliverable as "per-cell ROS-pinned makespan against
measured XPU-RT makespan", with the non-periodic makespan reported alongside.
That is the wrong way round. The quantity both runtimes are actually scored on
is **the makespan of the non-periodic work while the periodic tasks'
constraints are honoured** — which is what `sweep10_runner`'s
`evaluate(ctx, t, alpha, True)` minimises. A schedule that finishes the
non-periodic work having had to run *fewer periodic instances* is a correct and
better solution to the same problem, not a different workload. So the
non-periodic makespan is the comparison and the all-operations wall clock is
secondary. This moves the headline from **1.0042** (wall clock, 42 cells) to
**0.9762** (non-periodic, 26 cells).

**0.2 — Four cells are excluded from the headline because the *non-periodic*
work differs.** Fewer periodic instances is fine; fewer instances of the
*aperiodic* network is not, because that is the work being timed. `saturation`
declares `yolov8_nano_se` with `num_instances: 2` and XPU-RT scheduled **one**
(verified in the trace of `saturation_dc__greedy`: `{'dronet_sf': 6,
'mlp_control_sf': 16, 'yolov8_nano_se': 1}`). The pinning baseline runs both.
`saturation_{cg,dc,hd,quad}` are therefore reported but kept out of the
headline aggregate. **On all 26 other cells with an aperiodic network the
non-periodic instance counts match exactly** — checked per cell against the
XPU-RT sweep's own trace blocks, not assumed.

**0.3 — A harness bug was found after SETUP.md was written, and every
measurement postdates the fix.** The one-shot "kick" timer was created as
`kick_ = create_wall_timer(1ns, [this]{ kick_->cancel(); ... })` while the
node's executor was already spinning, so the callback could run before the
assignment to `kick_` landed and dereference a null `shared_ptr`. It killed 8
of the first 68 runs, and because stdout was block-buffered through `ssh` the
crash destroyed results that had already been printed. Fixed two ways:
executor threads now park at a barrier so **every pass's timers are created
while nothing is spinning**, and stdout is line-buffered. The 68 pre-fix runs
were **discarded, not reused** — the whole campaign was re-run on the fixed
harness. Nothing in this document comes from before the fix.

**0.4 — Scope grew by the 3net arm and its comparator.** SETUP.md's deliverable
list did not include `plans3net/`, `specs3net/`, `data/toplevel/rospin3net/`,
`results/xpurt3net.json` or `scripts/{pin3net,xpurt3net}.py`. RoSE's fifteen
3-network configs had no measured XPU-RT number on this board, so this campaign
emitted, solved and ran them here too (§10).

**0.5 — Everything else in SETUP.md held.** The expressibility verdicts, the
`n_legal` distribution, the harness limits, the 172-assignment measurement plan
and the noise floor are as written, and `reproduce.py` re-derives all of them
from the frozen inputs.

## 1. The noise floor, and this baseline's own

SETUP.md §8 quoted the XPU-RT sweep's measured rep spread before using it:
**7.62 %** median on the wall clock, **9.18 %** on the non-periodic objective.
Those bands are drawn on every figure and every "inside/outside" verdict below
is against them.

This campaign's own spread is much tighter: **2.46 % median** over 256
assignments. It is not uniformly tight — the maximum is 110 %
(`depth_contended_dc__a3`, reps `[30.6, 14.9, 14.2]` ms), and every one of the
eight worst is a placement with **two or more networks on the CPU lane**. The
QNN CPU backend runs its own thread pool, unmasked (the condition
`cost_model.json` was captured under), so two CPU-pinned nodes fight for the
same cores and the tail is long. Read every CPU-heavy cell — the whole `cg`
column especially — with that in mind.

## 2. Expressibility, as classified

| verdict | cells |
|---|---|
| `OK` — expressed as written, ≥2 legal assignments | **31** |
| `DEGENERATE (1-LANE)` — expressed, one legal assignment | **11** |
| `INEXPRESSIBLE` | **0** |

The 11 degenerate cells are the 11 `cg` cells: `cg` declares CPU + GPU, GPU is
not a pinning candidate (SETUP.md §1), so everything lands on the CPU and there
is no placement decision to make. That is a real narrowing and it shows: the
`cg` column holds 5 of the 8 noisiest runs in the campaign and both of the
losses to XPU-RT that are not about `vint`.

Three things the reference could not do, and this could:

* **The `fastdepth → dronet_sf` edge is wired, not dropped.** The micro-ROS
  reference had to run the depth families as independent timers and label them
  EDGE-DROPPED. Here `fastdepth`'s node publishes its instance index and
  `dronet_sf`'s node is driven by that subscription, so **all 12 depth cells
  are genuine `depth_*` results**. The wiring costs a publish→execute hop of
  **141.7 µs at p50** over 342 measured instances (p95 2.58 ms, but the tail is
  the downstream node being busy, not DDS).
* **`scale_ladder` is expressible.** The reference excluded it because of a
  compile-time 2–3 node cap; nothing here has one. All 4 cells ran, and
  `scale_ladder_quad` is the largest legal space in the matrix at 729
  assignments.
* **No cell sits below the executor floor.** One `SingleThreadedExecutor`
  tracks a 0.5 ms period at p50 0.5000 / p95 0.5030 ms
  (`results/floor.json`); the tightest period in the matrix is 0.670 ms. The
  reference's ~200 µs floor degraded two of its families; nothing is degraded
  here.

## 3. `n_legal`, reported rather than assumed

| n_legal | 1 | 2 | 4 | 6 | 8 | 9 | 12 | 18 | 27 | 64 | 729 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| cells | 11 | 5 | 8 | 3 | 6 | 1 | 1 | 3 | 1 | 2 | 1 |

min 1, median 4, max 729, **1078 legal assignments in total**. 33 cells were
enumerated in full on hardware (`n_legal ≤ 8`); 9 were sampled — the ranked
head of both objectives, both uniform placements and the bottom-ranked
assignment, so the measured span brackets the predicted range rather than only
its optimistic end. Every cell records which it got.

## 4. The headline: pinning vs scheduling on the objective

**26 cells, best legal placement against the best measured solver, medians of
3 reps.**

| | |
|---|---|
| median ROS ÷ XPU-RT | **0.9762** |
| pinning faster | 15 cells |
| scheduler faster | 11 cells |
| inside the ±9.18 % noise floor | 8 cells |
| worst for pinning | `vint_intro_quad`, **3.54×** |
| best for pinning | `control_mix_hd`, **0.52×** |

**At the median, whole-model pinning and per-op scheduling are the same speed
on this board's own workload matrix — and the median hides everything
interesting.** The distribution is bimodal, and both tails have a mechanism.

### 4.1 Where the scheduler wins, and why

| cell | ROS ÷ XPU-RT | mechanism |
|---|---|---|
| `vint_intro_quad` | **3.54×** | per-tile placement |
| `vint_multi_quad` | 3.10× | per-tile placement |
| `vint_intro_dc` | 2.45× | per-tile placement |
| `vint_multi_dc` | 2.41× | per-tile placement |
| `scale_ladder_cg` | 2.73× | GPU excluded → one lane for six networks |
| `depth_contended_cg` | 1.43× | one lane for three networks |
| `vint_multi_cg` | 1.43× | per-tile placement + one lane |
| `vint_intro_cg` | 1.25× | per-tile placement |

**Five of the eight are `vint`, and `vint` is the cleanest forfeiture in the
matrix.** Its encoder tile composes on {dsp, cpu}, its decoder on {cpu, gpu};
the intersection is CPU alone, so a whole-model pin pays 84.163 + 37.826 =
**121.989 ms**, while the scheduler puts the encoders on the DSP at 14.213 ms.
SETUP.md §3.2 predicted this before anything ran and the board confirms it:
measured `vint_intro_quad` is 107.8 ms pinned against 30.5 ms scheduled. This
is the single result that most justifies a per-op scheduler on this hardware,
and it has nothing to do with scheduling *policy* — it is about being allowed
to cut the model at all.

The other three are the GPU exclusion biting: with `cg` reduced to a single
lane, six `scale_ladder` networks serialise on one CPU.

### 4.2 Where pinning wins, and why

| cell | ROS ÷ XPU-RT | best placement |
|---|---|---|
| `control_mix_hd` | **0.52×** | everything on DSP |
| `scale_ladder_hd` | 0.62× | all six on DSP |
| `scale_ladder_dc` | 0.63× | all six on DSP |
| `bimodal_hd` | 0.72× | both on DSP |
| `perception_heavy_quad` | 0.74× | both on DSP |
| `depth_contended_quad` | 0.76× | fastdepth@cpu, dronet_sf@hta, yolo@dsp |
| `perception_heavy_dc` | 0.78× | yolo@dsp, mlp@cpu |
| `control_mix_cg` | 0.80× | everything on CPU |

**On eleven of the fifteen wins, every aperiodic network is pinned to the DSP
— "give the timed work the fast lane to itself and keep out of its way".** The
other three of the four are `cg` cells, where CPU is the only lane there is. The non-periodic makespan is the completion of one
network; a pinned node runs it as a single uninterrupted dispatch chain, while
the scheduler interleaves periodic instances into the same lane and pays a gate
per entry. Where the periodic work has somewhere else to go — and on this board
it usually does, because every `mlp_control` prefers the CPU by 3–9× — the
placement decision alone captures most of what there is to capture, and the
scheduler's extra freedom buys less than its extra overhead costs.

This is the result the brief asked not to be advocated away, so it is stated
plainly: **on 15 of 26 cells a team writing ordinary ROS nodes and choosing the
right lane per network beats the measured scheduler**, and on 7 of those the
margin is outside the noise floor.

### 4.3 The twelve cells with no aperiodic network

`depth_chain`, `depth_nav` and `tight_loop` are all-periodic, so the
non-periodic objective degenerates to the all-operations makespan exactly as
`evaluate()` does. For these the wall clock *is* the comparison, and it is a
valid one: both sides executed identical entry counts (checked per cell).
Median ROS ÷ XPU-RT **1.0029**, 4 of 12 faster, 10 of 12 inside the noise
floor. A dead heat, which is the honest reading — with every network periodic
and every release fixed, there is very little for either runtime to decide.

## 5. The wall clock, reported second

Over all 42 cells the all-operations makespan gives median **1.0042**, 15 cells
faster, 27 slower, 25 inside the noise floor. It is the weaker comparison and
the reason is structural: on a cell with a long periodic tail the wall clock is
pinned by the last periodic **release**, `(n−1)·period`, which is a property of
the workload and not of either runtime. `bimodal_dc` is the clean example —
32.533 ms pinned against 32.424 ms scheduled, and 31·1.045 = 32.4 ms of that is
just the clock ticking. Quoting it as the headline would have reported a
near-perfect tie on 25 cells where nothing was being measured.

## 6. What choosing the lane is worth, measured

Median placement spread (worst ÷ best measured legal placement) is **1.04** on
the wall clock and the maximum is 4.26×. On the objective it is far larger —
up to **6.74×**:

| cell | n_legal | measured | best np | worst np | spread |
|---|---|---|---|---|---|
| `control_mix_quad` | 18 | 7 | 3.414 | 23.023 | **6.74×** |
| `depth_contended_dc` | 8 | 8 | 3.641 | 17.781 | 4.88× |
| `control_mix_dc` | 8 | 8 | 3.492 | 15.477 | 4.43× |
| `scale_ladder_dc` | 64 | 5 | 3.463 | 14.754 | 4.26× |

**So ranking the placements is not a formality — it is worth up to 6.7× on the
same cell, and a baseline that picked a placement carelessly would have
flattered the scheduler by more than any scheduling effect in this matrix.**
That is exactly the reason the reference insists on the best permutation rather
than an arbitrary one.

## 7. The ranking: what the cost model got right, and where the rule matters

**The whole-model cost model picks the placement.** Its predicted-best
assignment was the measured-best on **87.1 %** of cells by wall clock and
**93.5 %** by the non-periodic objective. Where it missed, it missed by little.

**The reference's §6.5 replicates.** Ranking on makespan alone is not the same
as ranking on what you would ship, and on **7 of 42 cells the two rules
disagree**:

| cell | fastest placement | its window misses/rep | feasible-first placement | its misses | cost |
|---|---|---|---|---|---|
| `control_mix_quad` | `a0` 42.802 ms | **2.7** | `a16` 43.610 ms | **0** | +1.9 % |
| `saturation_quad` | `a1` 20.605 ms | 1.0 | `a6` 21.473 ms | **0** | +4.2 % |
| `saturation_hd` | `a0` 19.907 ms | 9.7 | `a3` 20.709 ms | 6.0 | +4.0 % |
| `saturation_dc` | `a2` 20.811 ms | 7.7 | `a1` 20.850 ms | 0.7 | +0.2 % |
| `depth_nav_hd` | `a1` 31.960 ms | 5.3 | `a3` 31.974 ms | 0.7 | +0.04 % |
| `control_mix_hd` | `a0` 42.780 ms | 3.3 | `a1` 42.848 ms | 1.0 | +0.2 % |
| `bimodal_quad` | `a0` 32.473 ms | 0.7 | `a1` 32.531 ms | **0** | +0.2 % |

On `control_mix_quad`, **1.9 % more makespan buys the elimination of every
window miss.** Window feasibility is deliberately kept out of the ranking key
and reported separately, exactly as the brief requires — but a reader choosing
a placement from this data should read both columns.

**Starvation, by contrast, did not replicate, and could not have.** SETUP.md
§5.3 said so before the campaign: this harness runs a finite taskset to
completion, so no instance can be dropped. **0 of 256 assignments starved a
network**, and the `(starved, makespan)` key therefore reduces to makespan on
every completed run. The reference's starvation arose because its window was
closed by a one-shot and a co-resident could vanish; nothing here can. Reported
as a structural difference, not as a null result.

## 8. Where the whole-model cost model breaks

Median absolute error **2.26 %** over 256 assignments, p90 **23.3 %**, range
**−68.5 % .. +126.4 %**. The median is excellent and the tails are the whole
story.

**It over-predicts when it charges for contention that does not happen:**

| assignment | predicted | measured | err |
|---|---|---|---|
| `scale_ladder_hd__a63` | 14.622 | 4.604 | **−68.5 %** |
| `scale_ladder_quad__a728` | 14.622 | 7.370 | −49.6 % |
| `depth_contended_hd__a7` | 19.264 | 13.814 | −28.3 % |
| `saturation_hd__a3` | 28.551 | 20.709 | −27.5 % |

All four pile several networks onto **HTA**. The model serialises them at their
measured solo cost; the hardware does not, because a QNN HTA context's
host-side call overlaps another context's accelerator time. The model's
serial-FIFO assumption is simply wrong for that lane.

**It under-predicts when several networks share the CPU:**

| assignment | predicted | measured | err | rep spread |
|---|---|---|---|---|
| `scale_ladder_cg__a0` | 9.936 | 22.496 | **+126.4 %** | 99.5 % |
| `scale_ladder_quad__a2` | 2.664 | 5.174 | +94.2 % | 27.1 % |
| `scale_ladder_dc__a2` | 3.031 | 5.831 | +92.4 % | 23.9 % |
| `depth_contended_dc__a7` | 15.154 | 28.417 | +87.5 % | 56.8 % |

Every one is two or more networks on the CPU lane. The per-network cell was
measured with QnnCpu's thread pool having the machine to itself; two such pools
do not add, they multiply, and the rep spread goes with them. **This is the
same confound the QRB5165 sweep already has on record** — "the CPU-lane
contention term the cost model has no way to express" — reproduced from the
other side, and it is why the cost model is used here only to *rank* candidates
and never to report a number.

Split by structure: median absolute error **2.28 %** where two networks share a
backend, **2.22 %** where each has its own. The tails, not the medians, carry
the failure.

## 9. Window feasibility, reported separately

Never folded into any makespan. The cells that miss windows at their best
placement are `saturation_{cg,hd,dc}` (11.7 / 9.7 / 7.7 instances per rep),
`bimodal_hd` (9.0), `depth_nav_hd` (5.3) and `control_mix_{hd,quad}` (3.3 /
2.7). Every one is a cell where a periodic network's whole-model latency on its
pinned backend is a large fraction of its own period — `mlp_control_sf` at
0.527 ms on the DSP against a 0.791 ms period, for instance. Whole-model
pinning cannot shorten a network to fit; that is the constraint being modelled.

## 10. The 3net arm

RoSE's fifteen 3-network configs
(`/scratch/dima/rose-infra/RoSE/soc/sw/xpu-rt/data/toplevel/networks_3net_*.json`
and `3net_pairs/`) are FireSim workloads over `mlp_control`, `dronet`,
`yolov8_nano` and `fused_full`. What ports is the workload shape; what does not
is the machine (`gemmini_q31` / `V256D128_rvv`) and the target bitstream.
**They collapse to 9 distinct shapes here**, because FireSim's
`gempair`/`rvvpair`/`hetero` axis counts identical cores and this board has
exactly one of each kind — so `gempair`, `hetero`, `milpfair` and `rvvpair` are
one workload, and `armA`/`armB` another. All 15 are accounted for; none is
dropped.

Both sides were measured **in this campaign, on the same three lanes**: 84
pinning assignments (full enumeration everywhere) and XPU-RT schedules emitted,
solved and run here. All three solvers (`greedy`, `heft_edf`, `cpsat`) produced
**identical schedules on every shape** — these are 2–3 network problems with
one tile each, so there is nothing for a better solver to find.

| shape | n | ROS np | XPU-RT np | ROS ÷ XPU-RT | placement spread |
|---|---|---|---|---|---|
| `3net_dronet4_mlp4_yolo1` | 12 | 27.060 | 49.929 | **0.542** | 4.07× |
| `3net_dronet8_mlp16_yolo1` | 12 | 28.490 | 38.762 | **0.735** | 4.88× |
| `3net_dronet4_mlp8_yolo1` | 12 | 28.436 | 37.996 | **0.748** | 3.84× |
| `3net_dronet1_mlp2_yolo1` | 12 | 26.190 | 31.912 | **0.821** | 3.31× |
| `3net_dronet2_mlp8_yolo1` | 12 | 28.677 | 31.143 | 0.921 | 3.21× |
| `3net_fused4_mlp4_yolo1` | 8 | 28.671 | 30.816 | 0.930 | 3.47× |
| `3net_fused2_mlp8_yolo1` | 8 | 28.979 | 30.172 | 0.961 | 3.26× |
| `3net_dronet1_mlp2` † | 6 | 10.209 | 10.056 | 1.015 | 1.28× |
| `3net_mlp2` † | 2 | 10.316 | 10.067 | 1.025 | 1.02× |

† no aperiodic network; the objective degenerates to the wall clock, quoted
only because both sides executed identical entry counts.

**Pinning wins all seven shapes that have an aperiodic network, median 0.82×,
four of them outside the noise floor.** The mechanism is §4.2's, in its purest
form: the objective is one `yolov8n` completion, pinning gives it the DSP to
itself (28.6 ms whole-model) and sends the periodic `mlp_control` and `dronet`
to the CPU and HTA, and there is nothing left for a scheduler to improve. The
placement decision, meanwhile, is worth **3.2–4.9×** on every one of them —
which is the same point as §6: the baseline is only fair because the placement
was ranked.

XPU-RT's schedules trim periodic instances that fall after the non-periodic
makespan (8 entries against 25 declared on `3net_dronet8_mlp16_yolo1`). That is
a better solution to the same problem, not a different workload, and it is why
only the non-periodic column is compared.

## 11. What was skipped, and why

* **The shard arm.** Inexpressible under whole-model pinning by construction —
  and it does not arise here anyway, because the QRB5165 port of
  `sched_algo_sweep10` has no shard arm to port. A loss the port already took,
  not one this baseline introduces.
* **The 8 `*_dcg` and `*_dg` cells.** Generated by the XPU-RT sweep but never
  measured by it, so there is nothing to compare against.
* **`n_legal > 8` cells were sampled, not exhausted.** 9 cells; 729 legal
  assignments on `scale_ladder_quad` alone, 1078 across the matrix against 172
  measured. The sampled set is the ranked head of both objectives plus the
  uniform and bottom-ranked placements, so the measured range brackets the
  predicted one — but a better placement may exist in the unmeasured tail of
  those 9 cells, which would only make the baseline stronger.
* **GPU as a pinning lane.** Excluded by design (SETUP.md §1). Its cost is
  concentrated and reported: it is what makes all 11 `cg` cells degenerate.
* **Per-solver XPU-RT numbers on the main arm** are whatever the XPU-RT sweep
  measured — 12 solvers on 12 cells, 1–2 on the rest. The comparison uses each
  cell's best measured solver, which is the most favourable reading for the
  scheduler.

## 12. Board discipline and provenance

Every board interaction was `timeout -s KILL … ssh -n … "flock -w 900
/tmp/qnn_board.lock -c '…'"`, batched 4 assignments to a lock acquisition: 64
batched calls, **12.8 s median, 26.5 s maximum** wall per call including lock
wait — this board had no other heavy tenant during the campaign. The CPU
governor was read before the campaign (`performance`), forced to `performance`
on all 8 cores, and restored to `performance` after, matching the conditions
`cost_model.json` was captured under and the conditions the XPU-RT runs used.
The board's root filesystem went from 32 G free to **19 G free** over the
session; this campaign's own footprint is 1.1 M of configs, and its runtime
directories were removed afterwards. Host `/scratch2` is at 96 %.

`reproduce.py` re-derives `model_costs.json`, the expressibility verdicts, the
enumeration, every plan and every median in `measured.json` from the frozen
inputs, and re-checks the two noise-floor figures against the XPU-RT sweep's
own `phase4_results.json`. It needs no hardware.

## 13. Figures

| file | what |
|---|---|
| `plots/ros_vs_xpurt_objective.png` | **the headline, and the one to report**: ROS ÷ XPU-RT on the objective, log ratio axis, the 26 compared cells separated from the 16 excluded, both tails' mechanisms on the figure (`scripts/plot_comparison.py`) |
| `plots/ros_vs_xpurt_nonperiodic.png` | the same data, first draft — linear axis and all 42 cells in one ranking with the excluded greyed. Superseded; kept because §4's cell-by-cell text was written against it |
| `plots/ros_vs_xpurt.png` | the same on the all-operations wall clock |
| `plots/placement_value.png` | worst ÷ best measured legal placement, per cell |
| `plots/costmodel.png` | predicted vs measured, split on whether two networks share a backend |
| `plots/ros_vs_xpurt_3net.png` | the 3net arm |
