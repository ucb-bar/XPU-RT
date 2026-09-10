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
emitted, solved and ran them here too (§10). The follow-up in §0.8 added
`results/undeclared_3net.json`, `results/verify3net.json` and three
`xpurt3net.py` subcommands (`undeclared`, `verify`, and a `--solvers` flag on
the rest) — none of them in SETUP.md either.

**0.5 — The opponent is now the *recommended* solver, and it makes XPU-RT look
WORSE, not better.** The first version of §4 scored pinning against
`xrt_np_best_ms` — the best *measured* solver per cell. Phase 4 of the XPU-RT
sweep was tiered by board time (`scripts/drive.py:121`): tier A ran all twelve
solvers on twelve cells, tier B ran only winner+greedy on the other thirty. So
on thirty cells "the best measured solver" was chosen from two candidates, and
on five of them (`bimodal_hd`, `control_mix_hd`, `depth_nav_dc`,
`tight_loop_{dc,hd}`) from greedy alone. **`cpsat:warmbest` has now been
measured on all 42 cells** (§13), so the comparison can be stated against the
solver that sweep actually recommends for the offline/build-time path. Both
readings are reported, and each is labelled:

| opponent | median ROS ÷ XPU-RT | pinning / scheduler | inside ±9.18% |
|---|---|---|---|
| **`cpsat:warmbest`** — the recommendation | **0.9281** | 17 / 9 | 6 |
| best measured solver over all twelve — an oracle | 0.9762 | 15 / 11 | 8 |
| *greedy alone* — what the tiering left on 5 cells | *0.7651* | *19 / 7* | — |

**The premise this campaign was launched on turned out to be wrong, and that is
the result.** Comparing against greedy really would have understated XPU-RT
badly — 0.7651 against 0.9281 — but §4 was never against greedy; it was against
an oracle over every solver that happened to be measured, which is the most
favourable reading available. Naming the actual recommendation moves the median
the other way, from **0.9762 to 0.9281**, because on this board
**`cpsat:warmbest` is the best measured solver on only 18 of 42 cells** despite
being ranked first by predicted objective. Its median penalty against the
per-cell measured best is 1.0010×, but its tail is 1.5217× (`saturation_dc`:
6.379 ms against `heft`'s 4.192 ms). Against greedy specifically it is a median
0.9898× — 22 cells faster (down to 0.2838× on `saturation_dc`), 10 tied and
**10 slower** (up to 1.1632× on `scale_ladder_hd`). That is the sweep's own §7.3
caveat — "60% of pairwise solver comparisons are inside the noise" — showing up
as a headline number rather than as a footnote.

Two cells change direction against the previous headline, both because the
opponent changed rather than because new data contradicted old:
`depth_contended_hd` 1.0235 → **0.8812** and `control_mix_dc` 1.0054 →
**0.9847** (the latter inside the band on both readings). One cell moved
materially in XPU-RT's favour on *new data*: `control_mix_hd`, whose only
XPU-RT measurement had been greedy at 8.861 ms, now has `cpsat:warmbest` at
7.879 ms, so **0.5246 → 0.5899**. The all-operations wall clock moves from
1.0042 to **1.0048** (15 → 13 cells pinning faster) as the new schedules enter
the "best measured" pool.

**0.6 — `prune_periodic` is applied to the reported schedules, and it does not
move the objective.** Every ported spec carries `scheduler.prune_periodic: true`
(`mk_workloads_qrb5165.py:282`) and nothing in the XPU-RT sweep read it: the
trim lives in `scripts/run_xpurt_schedule.py`, which that sweep bypasses in
favour of `sweep10_runner.make_solver`, so all 199 measured schedules are
untrimmed. `postprocessing.trim_periodic_after_nonperiodic_makespan` is purely
post-hoc — it takes an already-computed `(t, alpha)` and drops periodic
operations whose window does not overlap `[0, non-periodic makespan)` — so it
cannot move a placement and cannot move the non-periodic makespan. **Asserted,
not assumed: on all 42 cells the objective is bit-identical before and after,**
while the all-operations makespan shortens on 21 of them (`bimodal_dc`: 33
operations and 32.454 ms of table down to 10 and 8.691 ms). The board agrees:
four trimmed schedules were built and run for 3 reps each and the measured
non-periodic makespan is unchanged or marginally *shorter*, never longer
(0.982×, 0.847×, 0.905×, 1.021×; three of the four rep ranges overlap the
untrimmed ones outright). Where it is shorter it is a runtime effect — a
dispatch table with fewer entries walks fewer gates during the interval being
timed — not a scheduling one. The comparison in §4 stays on the untrimmed
measurements, because that is what the other 199 points measured; the trimmed
schedules are emitted alongside (`schedules/pruned/`) and are what the traces
and gantts should be read from.

**0.7 — Everything else in SETUP.md held.** The expressibility verdicts, the
`n_legal` distribution, the harness limits, the 172-assignment measurement plan
and the noise floor are as written, and `reproduce.py` re-derives all of them
from the frozen inputs.

**0.8 — The 3net arm's comparator was wrong twice over, and fixing it moves
that arm's headline from 0.82× to 0.93×.** §10 originally scored pinning
against cold `cpsat` — not `cpsat:warmbest`, the solver the sweep10 study
recommends and the one §4 is now scored against — so the two arms were not
comparable. Closing that gap turned up a second and larger defect.

*The dedupe key hashed a field that does not exist.* `xpurt3net.py`'s
`cmd_run` deduped identical schedules on
`sha256(json.dumps(json.load(sched).get("schedule")))`, and
`postprocessing.output_scheduled_json` emits no `"schedule"` key — it emits
`dot_file` / `dispatches` / `metadata`. Every schedule therefore hashed to
`sha256("null")`, every solver after the first was recorded as a duplicate of
it, and **`heft_edf` and `cpsat` were never run on this arm at all**: all three
columns of the old §10 table were `greedy`. The note that "all three solvers
produced identical schedules on every shape" was an artefact of that constant.
They do not: on **7 of the 9 shapes the three disagree**, and on three of
those `heft_edf` predicts 28.644 ms where `greedy` predicts 36.022–46.022 ms. The key is now
the one Phase 4 uses — op → (combination, start, duration) — so two solvers
share board time only when they really produced the same table.

*What that costs the arm's claim.* Measuring the schedules the broken key had
hidden gives **median 0.9305 against `cpsat:warmbest`, 7 of 7 shapes still
pinning-faster, but only 2 of the 7 outside the ±9.18% noise floor** — against
the old **0.82× with four outside**. The direction survives the change of
opponent; the margin does not, and it was never a margin against the
recommended solver in the first place. Details and the per-shape table are
in §10.

*The `--mask-undeclared` fix applies here too, and this arm has its own
undeclared cells.* §13's defect is a property of the pipeline, not of the two
`vint` cells, so it was checked rather than assumed. This arm has **three**
cells the profile tree prices and the binding manifests forbid —
`mlp_control/mlp_control_full@hta`, `yolov8n/yolov8n_head@hta` and
`fused_full/fused_full_net@hta` (`results/undeclared_3net.json`, derived by
`xpurt3net.py undeclared` from the bindings and `gen/profile/`, not from the
main arm's cost model, which covers a different network set). **No solver took
the bait**: across the 36 schedules this arm measured, 149 operations match an
undeclared cell and **none is placed on the forbidden lane**, and the same
holds for all 72 emitted under `verify` masked and unmasked. Nothing subtle is
going on — on each of those three tiles the HTA is **257× to 1038× slower**
than that tile's best declared lane (`mlp_control` 68.500 ms against 0.066 on
the CPU, `yolov8n_head` 3946.462 against 15.377 on the DSP, `fused_full`
473.700 against 1.147 on the CPU). The mask is applied anyway, because it is a constraint and not a tuning
knob; it changes no objective and no `greedy`, `heft_edf` or `cpsat:warmbest`
schedule on any shape.

*And the two solve paths were checked against each other before any of this was
believed.* `cpsat:warmbest` is not one of the ten solvers
`scripts/run_xpurt_schedule.py` exposes, so it has to come from
`sched_algo_sweep10`'s `fpga/emit_schedule.py`, while every 3net schedule
already measured came from `run_xpurt_schedule.py`. Comparing across builders
would be a harness difference dressed as a solver difference.
`xpurt3net.py verify` re-emits every expressible solver both ways:
**`greedy` and `heft_edf` are byte-identical on all 9 shapes**, so the emitter
route is the same experiment. Cold `cpsat` is the one exception — it differs
from its own on-disk schedule on 4 of 9 and from itself masked-vs-unmasked on
4 — **with an identical objective every time** (28.644 or 10.066), i.e. an
arbitrary draw from a set of tied optima under a different time budget, not a
different answer. `results/verify3net.json` records all of it.

## 1. The noise floor, and this baseline's own

SETUP.md §8 quoted the XPU-RT sweep's measured rep spread before using it:
**7.62 %** median on the wall clock, **9.18 %** on the non-periodic objective,
over that sweep's 199 Phase-4 points. Those bands are drawn on every figure and
every "inside/outside" verdict below is against them.

Extending that sweep by 30 points (§13) makes the same statistic **6.83 % /
7.93 %** over 229. **The pre-registered 9.18 % is kept**, because it is the
wider of the two and a wider band calls fewer differences results; narrowing it
after the fact would promote borderline cells into findings, which is what a
pre-registered floor exists to prevent. `reproduce.py` checks both and says
which is used.

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

**26 cells, best legal placement against XPU-RT, medians of 3 reps.** Two
opponents, both reported, each named (§0.5). The noise floor is quoted first:
the XPU-RT sweep's own measured rep spread is **9.18 % median on this
objective**, and every "inside/outside" verdict below is against that band.

| | vs `cpsat:warmbest` (the recommendation) | vs best measured over 12 solvers (an oracle) |
|---|---|---|
| median ROS ÷ XPU-RT | **0.9281** | 0.9762 |
| pinning faster | 17 cells | 15 cells |
| scheduler faster | 9 cells | 11 cells |
| inside the ±9.18 % noise floor | 6 cells | 8 cells |
| worst for pinning | `vint_intro_quad`, **3.51×** | `vint_intro_quad`, 3.54× |
| best for pinning | `scale_ladder_hd`, **0.53×** | `control_mix_hd`, 0.59× |

**`cpsat:warmbest` is the number to quote**, because it is what a team would
actually run at build time; the oracle column needs twelve solvers measured on
hardware per cell to be available at all, and this campaign is the only reason
it exists on twelve of the 42.

**Inside the band on 6 of 26** — `bimodal_{cg,dc,quad}` (0.919 / 0.961 / 0.937),
`control_mix_{dc,quad}` (0.985 / 0.987) and `depth_contended_dc` (1.068). Those
six are not results in either direction. Of the remaining twenty, **twelve are
outside the band with pinning faster** and **eight outside with the scheduler
faster**; the two tails are §4.1 and §4.2 and both keep their mechanism.

**At the median, whole-model pinning is about 7 % ahead of the recommended
scheduler on this board's own workload matrix — and the median hides everything
interesting.** The distribution is bimodal, and both tails have a mechanism.

### 4.1 Where the scheduler wins, and why

Against `cpsat:warmbest`, all eight outside the noise band:

| cell | ROS ÷ XPU-RT | mechanism |
|---|---|---|
| `vint_intro_quad` | **3.51×** | per-tile placement |
| `vint_multi_quad` | 3.13× | per-tile placement |
| `scale_ladder_cg` | 2.52× | GPU excluded → one lane for six networks |
| `vint_intro_dc` | 2.45× | per-tile placement |
| `vint_multi_dc` | 2.41× | per-tile placement |
| `vint_multi_cg` | 1.46× | per-tile placement + one lane |
| `depth_contended_cg` | 1.43× | one lane for three networks |
| `vint_intro_cg` | 1.26× | per-tile placement |

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

Against `cpsat:warmbest`; twelve are outside the noise band, the eight largest:

| cell | ROS ÷ XPU-RT | best placement |
|---|---|---|
| `scale_ladder_hd` | **0.53×** | all six on DSP |
| `scale_ladder_dc` | 0.56× | all six on DSP |
| `control_mix_hd` | 0.59× | everything on DSP |
| `perception_heavy_hd` | 0.64× | both on DSP |
| `bimodal_hd` | 0.69× | both on DSP |
| `perception_heavy_quad` | 0.74× | both on DSP |
| `depth_contended_quad` | 0.76× | fastdepth@cpu, dronet_sf@hta, yolo@dsp |
| `perception_heavy_dc` | 0.78× | yolo@dsp, mlp@cpu |

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
plainly: **on 17 of 26 cells a team writing ordinary ROS nodes and choosing the
right lane per network beats the scheduler running its recommended solver**,
and on 12 of those the margin is outside the noise floor. Against the
twelve-solver oracle it is 15 of 26 with 7 outside — better for XPU-RT, but
only available to someone willing to run twelve board campaigns per cell.

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
solved and run here.

**The noise floor first.** The comparison band is the XPU-RT sweep's own
measured rep spread on the objective, **±9.18%** (§1). This arm's own new runs
sit inside it at the median: non-periodic rep spread **4.08% median, 19.47%
max** over the 11 schedules measured here, the max on
`3net_dronet8_mlp16_yolo1__cpsat`. The pinning side's np spread is 0.1–11.4%
per shape, and the one shape above the band —
`3net_dronet1_mlp2_yolo1` at 11.4% — is also the shape with the largest
apparent win, which is the reason the band is quoted before the ratios and not
after them.

**Four solvers, on all nine shapes, scored against the recommended one.**
`greedy`, `heft_edf`, `cpsat` and **`cpsat:warmbest`** — the last being what
§4 scores the main arm against, so the two arms now face the same opponent.
The primary column is `cpsat:warmbest`; the secondary is the best measured
solver per shape, an oracle over those four. Neither stands in for the other.

| shape | n | ROS np | `cpsat:warmbest` | **ROS ÷ warmbest** | best-of-4 | *greedy alone* | placement spread |
|---|---|---|---|---|---|---|---|
| `3net_dronet1_mlp2_yolo1` | 12 | 26.190 | 30.270 | **0.865** | 30.270 | *31.912* | 3.31× |
| `3net_dronet4_mlp4_yolo1` | 12 | 27.060 | 31.182 | **0.868** | 31.182 | *49.929* | 4.07× |
| `3net_fused4_mlp4_yolo1` | 8 | 28.671 | 30.816 | 0.930 | 30.410 | *30.816* | 3.47× |
| `3net_dronet8_mlp16_yolo1` | 12 | 28.490 | 30.617 | 0.930 | 30.617 | *38.762* | 4.88× |
| `3net_dronet4_mlp8_yolo1` | 12 | 28.436 | 30.531 | 0.931 | 30.531 | *37.996* | 3.84× |
| `3net_dronet2_mlp8_yolo1` | 12 | 28.677 | 30.221 | 0.949 | 30.221 | *31.143* | 3.21× |
| `3net_fused2_mlp8_yolo1` | 8 | 28.979 | 30.172 | 0.961 | 30.172 | *30.172* | 3.26× |
| `3net_dronet1_mlp2` † | 6 | 10.209 | 10.056 | 1.015 | 10.056 | *10.056* | 1.28× |
| `3net_mlp2` † | 2 | 10.316 | 10.067 | 1.025 | 10.067 | *10.067* | 1.02× |

† no aperiodic network; the objective degenerates to the wall clock, quoted
only because both sides executed identical entry counts. Bold = outside the
noise band.

| opponent | median ROS ÷ XPU-RT | pinning / scheduler | outside ±9.18% |
|---|---|---|---|
| **`cpsat:warmbest`** — the recommendation | **0.9305** | 7 / 0 | 2 |
| best measured solver over the four — an oracle | 0.9314 | 7 / 0 | 2 |
| *greedy alone* — what the old §10 actually measured | *0.821* | *7 / 0* | *4* |

**The conclusion survives in direction and not in magnitude.** Pinning is still
faster on all seven shapes with an aperiodic network, against the recommended
solver and against the oracle alike — that part holds. But the median moves
from **0.82× to 0.9305×**, and the number of shapes outside the noise floor
falls from four to **two** (`3net_dronet1_mlp2_yolo1` 0.865,
`3net_dronet4_mlp4_yolo1` 0.868). **Five of the seven are inside the band, so
on those five this arm shows a direction and not a result.** "Pinning wins all
seven shapes, median 0.82×, four outside the noise floor" is no longer a
supportable sentence; "pinning is ahead on all seven, by a margin that is
inside the noise on five of them" is.

**Almost none of that move is the change of opponent.** It is §0.8's dedupe
defect: the old table's "XPU-RT" column was `greedy` on every shape, because
`heft_edf` and `cpsat` had been recorded as duplicates of it without ever
running. `cpsat:warmbest` turns out to emit the **same schedule as `heft_edf`
on five of the nine shapes and the same as `greedy` on the other four**, and it
is `heft_edf`'s schedule that closes most of the gap — on
`3net_dronet4_mlp4_yolo1` from 49.929 ms to 31.182 ms. Comparing against
greedy really would have understated XPU-RT here, exactly as §0.5 warned for
the main arm, and on this arm that is what the old number did.

**The mechanism is unchanged and is still §4.2's, in its purest form.** The
objective is one `yolov8n` completion, and on all seven shapes the np-best
pinning puts `yolov8n` on the **DSP** (13.267 + 15.377 = 28.644 ms whole-model)
and the periodic work where it interferes least — `mlp_control` on the CPU at
0.066 ms on five shapes, `dronet` on the CPU at 6.998 ms on three. The HTA is
never used by the winning placement on any of them.

What a better solver recovers is a *lane assignment* the greedy one gets
backwards, and `3net_dronet4_mlp4_yolo1` shows it exactly. `yolov8n` is two
tiles and only the backbone has an HTA context; the head composes on
`{dsp, cpu}` alone. Greedy puts the backbone on the HTA (13.909 ms) and the
periodic `dronet` instances on the DSP, so the head — which now has nowhere
else to go — waits behind `dronet3`'s 30 ms release and starts at 30.645 ms.
`heft_edf`, and `cpsat:warmbest` with it, puts the whole of `yolov8n` on the
DSP back to back and the `dronet` instances on the HTA at 2.030 ms each: the
objective falls from a predicted 46.022 ms to 28.644 ms and a measured
49.929 ms to 31.182 ms. So the scheduler ends up at the placement pinning was
already at, and what it does not recover is the last ~7% — the per-entry
dispatch gate XPU-RT pays and a pinned thread does not.

The placement decision, meanwhile, is worth **3.2–4.9×** on every one of these
shapes — the same point as §6: the baseline is only fair because the placement
was ranked.

**`cpsat:warmbest` is the measured np-best on 8 of the 9 shapes**, not all of
them. On `3net_fused4_mlp4_yolo1` it emits greedy's schedule (30.816 ms) while
`heft_edf` measures 30.410 ms, which is why the oracle column is 0.943 there
against warmbest's 0.930. Same finding as §13's Defect 2, on a second arm.

**Cold `cpsat` is where the cost model breaks worst on this arm.** It proves the
same 28.644 ms objective as everyone else and then measures **64.275 ms on
`3net_dronet4_mlp8_yolo1` and 73.537 ms on `3net_dronet8_mlp16_yolo1`** —
2.2–2.6× its own prediction, and 1.7–1.9× worse than the greedy schedule it
was supposed to improve on. It is also the only solver whose schedule is not
reproducible across the two solve paths (§0.8), for the same underlying reason:
these problems have many tied optima and a cold search returns an arbitrary one,
which the cost model cannot distinguish and the board can.

**What it cost.** Emitting `cpsat:warmbest` on all 9 shapes and hashing against
the schedules already measured found that **all 9 dedupe onto another
schedule** — 4 onto that shape's `greedy` run, which was already on record and
consumed no board time, and 5 onto a `heft_edf` schedule that the broken key
had hidden and that therefore had to be measured. **Nine warmbest points, zero
of them needing a board run of their own; five needing the board because what
they dedupe onto had never been run.** Measuring every distinct schedule the
four solvers produce came to **11 unique tables and 33 board runs at 3 reps
each**, all `N/N entries executed`, none discarded, against 16 points resolved
as dedupes. Lock wait **0.27 s median / 0.34 s max** over 33 probes.

XPU-RT's schedules trim periodic instances that fall after the non-periodic
makespan (8 entries against 25 declared on `3net_dronet8_mlp16_yolo1`). That is
a better solution to the same problem, not a different workload, and it is why
only the non-periodic column is compared.

**No shape reversed direction**, so no 3net gantt was drawn: the one drawn for
a reversal would have nothing to show that the bar chart does not. The largest
single move is `3net_dronet4_mlp4_yolo1`, 0.542 → 0.868, and it is a change of
margin rather than of sign.

One artefact to know about when reading these schedules by hand: every
`dispatches[*].module_name` says `_HTA_` regardless of the lane, because
`output_scheduled_json`'s legacy p/e bucketing has no name for a third machine
kind. `hardware_target` and `duration` are correct and are what the codegen and
the dedupe hash read; it is identical on both solve paths and on every schedule
this arm ever measured, so it moves nothing here. It is noted only so the next
reader does not take it for a placement bug.

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
* **Per-solver XPU-RT numbers on the main arm.** The XPU-RT sweep measured 12
  solvers on 12 cells and winner+greedy on the other 30. This campaign added
  **`cpsat:warmbest` on all 42** (§13), so the headline is against the solver
  that sweep recommends; the twelve-solver oracle is still only available on
  twelve cells and is reported as the secondary column. On the other 30 cells
  the oracle is a best-of-three, not a best-of-twelve, and is quoted as such.
* **Solvers beyond four on the 3net arm.** That arm now measures `greedy`,
  `heft_edf`, `cpsat` and `cpsat:warmbest` on all 9 shapes (§10), which is
  enough to name the recommendation and an oracle over it. The remaining eight
  are not measured there, so its secondary column is a best-of-four and is
  labelled that way — the same honesty the main arm's best-of-three gets.

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
directories were removed afterwards. Host `/scratch2` is at 96 %. The follow-up
campaign in §13 found the board at 81 % / 19 G and **45 G of it in
`/data/tombstones/cdsp/pd_dump_`** — 881 CDSP process-domain core dumps
accumulated across this and the preceding sessions. Clearing them returned the
filesystem to 36 % / 63 G before any new run started. They regrow: 1.9 G
accumulated again over the 87 runs of §13, so this is a per-session chore and
not a one-off.

The 3net follow-up in §0.8 is a third session on the same discipline. It does
not take an outer `flock`: `flow_c.py run` acquires `/tmp/qnn_board.lock`
itself inside `qnn_models/runtime/deploy_and_run.sh`, so wrapping it would make
its own lock wait out its own 900 s. The lock is instead **probed with the same
flock on the same path immediately before every rep** —
`xpurt3net.py::lock_wait_s`, verbatim `sched_algo_sweep10`'s, so the two arms'
recorded waits mean the same thing. **33 runs, 33 probes, 0.27 s median /
0.34 s max**; no contention. The governor was read (`performance`), forced to
`performance` on all 8 cores, and restored to `performance` after. The board
started at **36% / 63 G free** and finished at **39% / 61 G**, with
`/data/tombstones/cdsp` regrown from 4 K to **2.4 G** over the 33 runs —
the same per-session chore §13 found, at the same rate.

`reproduce.py` re-derives `model_costs.json`, the expressibility verdicts, the
enumeration, every plan and every median in `measured.json` from the frozen
inputs, and re-checks the two noise-floor figures against the XPU-RT sweep's
own `phase4_results.json`. It needs no hardware.

## 13. The `cpsat:warmbest` campaign, and two defects it exposed

`sched_algo_sweep10`'s Phase 4 was tiered by board time, which left its own
recommended solver measured on 12 of 42 cells. This campaign closed that gap so
§4 could be stated against the recommendation rather than against a best-of-two.

**What it cost.** 30 cells lacked `cpsat:warmbest`. Emitting all 30 and hashing
them against the 199 schedules already measured (the sweep's own Phase 4 dedupe,
on the op → (combination, start, duration) map) found **5 that were byte-identical
to a schedule already on record** — `depth_chain_{cg,hd}`,
`depth_contended_cg`, `perception_heavy_{dc,quad}`, every one of them resolving
through a `decomposed` hop onto that cell's `greedy` run. Those five consumed no
board time and are reported as dedupes, not as measurements. The other **25
needed board runs: 75 runs at 3 reps each**, all `N/N entries executed`, none
discarded. Lock wait 0.28 s median / 0.39 s max. Their non-periodic rep spread
is **4.67 % median, 37.8 % max** — inside the 9.18 % band the comparison uses,
at the median. A further 12 runs went to §0.6's prune check, for **87 in total**;
one of those waited 228 s on the board lock because another tenant held it, which
is the only contention this campaign saw.

**Defect 1 — the cost model's own exclusion never reaches the solver.**
`build_cost_model.py` drops any cell the binding manifest cannot execute, and
says exactly why: *"The scheduler has no `forbidden` flag — it will happily
place a tile on the cheapest lane it is offered."* It drops one:
`vint/vint_encoders@gpu`. But the **solver's** workload is built by
`load_profiled_processing_times` straight out of `gen/profile/`, which never
sees that decision — the two paths meet only at `flowc/schedule.py::ingest`,
i.e. *after* the solve. Re-solving `vint_intro_cg` and `vint_multi_cg` today,
CP-SAT proves an **OPTIMAL 72.279 ms** that puts the ViNT encoders on the GPU at
a measured-but-unbuildable 55.854 ms, and codegen refuses it. Phase 3 missed
this by two hours of luck: its solves ran 2026-09-08 16:31 and
`gen/profile/GPU/qrb5165_flowc/vint/.../results.csv` was not written until
18:21, so the cell was invisible to the solver then and is visible now. The fix
is `emit_schedule.py --mask-undeclared`, which masks the cost model's own
`dropped_undeclared_cells` to +inf **before** the search rather than rejecting
them after it. With it, both cells return Phase 3's 100.588 ms. Re-emitting all
30 cells with the mask changes **nothing on the other 28** (identical content
hashes), including the two `vint_*_quad` cells where the mask applies but the
solver had not taken the bait — so the mask is a constraint, not a tuning knob.

**Defect 2 — `cpsat:warmbest` leads on prediction and not on measurement.**
It is the measured np-best solver on **18 of 42 cells**. Median penalty against
the per-cell measured best is 1.0010×; the tail is 1.5217× (`saturation_dc`,
6.379 ms against `heft`'s 4.192 ms). Against greedy it is a median 0.9898× —
22 cells faster, 10 tied, **10 slower**, worst 1.1632× on `scale_ladder_hd`.
Several of those losses are cells where the two solvers' *predicted* objectives
are identical to three decimals (`bimodal_hd`: 10.001 both ways) and only the
board separates them. This is not new — it is `sched_algo_sweep10`'s own §7.3,
"60 % of pairwise solver comparisons are inside the noise" — but it is the
reason §4's two columns differ, and it is why the recommendation column is the
honest one to quote.

## 14. Figures

| file | what |
|---|---|
| `plots/ros_vs_xpurt_objective_warmbest.png` | **the headline, and the one to report**: ROS ÷ XPU-RT on the objective against `cpsat:warmbest`, log ratio axis, the 26 compared cells separated from the 16 excluded, both tails' mechanisms on the figure (`scripts/plot_comparison.py --against warmbest`) |
| `plots/ros_vs_xpurt_objective.png` | the same against the best measured solver over all twelve — the secondary reading (`--against best`). The two files are produced by one script from one `results/analysis.json`, and each states its opponent on the figure |
| `plots/ros_vs_xpurt_nonperiodic.png` | the same data, first draft — linear axis and all 42 cells in one ranking with the excluded greyed. Superseded; kept because §4's cell-by-cell text was written against it |
| `plots/ros_vs_xpurt.png` | the same on the all-operations wall clock |
| `plots/placement_value.png` | worst ÷ best measured legal placement, per cell |
| `plots/costmodel.png` | predicted vs measured, split on whether two networks share a backend |
| `plots/ros_vs_xpurt_3net.png` | **the 3net arm, both readings on one figure**: bars are ROS ÷ XPU-RT against `cpsat:warmbest`, open markers the same against the best measured solver per shape, so the gap between the recommendation and the oracle is the visible thing rather than a second file a reader might quote instead. Log ratio axis with reciprocal limits (0.78, 1/0.78) — this arm spans 0.865–1.025 and the main arm's wide axis would compress every bar into a smear at 1.0. The two shapes with no aperiodic network are a separate panel, not greyed rows (`scripts/analyse.py plots`) |
| `plots/gantt_<cell>.png` | XPU-RT vs ROS execution traces, both from the measured trace blocks, with the aperiodic-completion marker the ratio is computed on (`scripts/plot_gantt_compare.py --prefer cpsat:warmbest`). Six cells across **both** directions: `control_mix_hd` (0.59×), `perception_heavy_hd` (0.64×) and `bimodal_hd` (0.69×) where pinning wins; `depth_contended_cg` (1.43×), `vint_multi_cg` (1.46×) and `vint_intro_dc` (2.45×) where scheduling wins. Every one now draws `cpsat:warmbest`, so the "not measured on this cell" caveat the script prints no longer fires |
| `plots/gantt_scale_ladder_{hd,dc}.png` | the same, for the two cells the 3-slot palette used to refuse. See below |

**Why `scale_ladder_{hd,dc}` are drawn differently.** `SLOTS` is a
**categorical** palette of three, and three is where it stops: that is how far
the reference palette is validated on the all-pairs list, and a gantt places
colours at arbitrary spatial positions, so the adjacent-pair rule does not buy
a fourth slot. `scale_ladder_hd` and `scale_ladder_dc` have six networks each
and were skipped outright, which threw away the interesting part of the cell.

Those six are not six unrelated networks. `dronet_sb` … `dronet_sg` are **one
architecture at six sizes** — 23 IR ops each; 2.64, 3.98, 6.86, 13.06, 15.94
and 21.26 M MACs — so they are **ordinal, not categorical**, and a categorical
palette was the wrong tool rather than a palette one slot too short. They get a
**single-hue light-to-dark ramp keyed to size**, which encodes the ladder
truthfully and sidesteps the cap by not being a categorical encoding at all.
Faceting by rung was the alternative and is half-applied anyway (there is
already one sub-row per network), but it would have left the ladder — the thing
this cell is about — carried by nothing but the reading order of the y labels.

The ramp is not applied on faith. `ladder()` requires that every network share
one base name, that each carry a MAC count in its binding manifest (written
there by `phase1_bindings.py` from the board's own compose verdict, so it is a
property of the model that ran and not an inference from the name), and that
**sorting by MACs reproduce sorting by rung letter**. If a family is recognised
but not monotone the ramp would assert an order the data does not have, so it
falls back to one flat colour; a >3-network cell that is not a ladder at all is
still skipped. **Identity is never carried by colour alone on any path** —
every bar sits in a sub-row labelled `<LANE> · <network>` and each legend entry
spells out its rung and size, so the ramp is redundant encoding rather than the
encoding. On these two cells every network is aperiodic, so the
aperiodic hatch would separate nothing and six hatched bars would fight the
ramp; it is dropped and the legend says so instead.

What the figures show. Both cells are the same six networks on a different lane
pair — `hd` is HTA + DSP, `dc` is DSP + CPU — and pinning answers both the same
way: all six on the DSP, overlapping, aperiodic work done at **3.46 ms**. The
scheduler's answer differs by cell and loses in both. On `hd` it sends
`dronet_sg`, the largest rung, to the HTA for a measured 1.5 ms and serialises
the other five on the DSP, ending at **6.48 ms** — the DSP queue, not the HTA
excursion, is what it is paying for. On `dc` it moves `dronet_sd` and
`dronet_se` to the CPU, and `dronet_sd` alone runs **6.1 ms there — the entire
makespan**, four times the 1.474 ms the cost model prices it at, while the DSP
sits idle from 3.8 ms onward. Ending at **6.14 ms**, it is a placement the
model could not have known was bad.

The ramp also makes visible something the ladder does not predict: **the size
order is not the cost order** — `dronet_sf` is 15.9 M MACs against
`dronet_se`'s 13.1 M and is *cheaper* on the DSP, 0.810 ms against 0.838 ms,
and on the `hd` pinning panel it finishes first. A flat palette would have
hidden that; a categorical one would have made it look like a coincidence
between two unrelated networks.
