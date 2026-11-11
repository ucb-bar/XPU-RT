# ROS 2 whole-network-pinning baseline over `sched_algo_sweep10` — QRB5165 — RESULTS

Written after the campaign. `SETUP.md` is the contract and is **not** revised;
everything that turned out differently from what it assumed is in §0.

Everything below is measured. **379 assignments, 3 reps each, 1104 board runs,
all complete, none discarded** — 256 in the first two sessions and a further
123 in the follow-up of §0.11, of which 112 needed the board and 11 resolved as
content-duplicates of a run already on record.

**The headline changed twice since the first version of this document, and
both changes make the pinning baseline look worse.** §0.9 replaces the
best-of-enumeration placement with the one a ROS user actually deploys; §0.12
stops charging XPU-RT for a start-barrier delay it was being charged for. The
previous numbers are all still here, labelled, because a reader has to be able
to see what moved.

---

## 0. Corrections to SETUP.md

> **§0.1 to §0.8 are the corrections made in the first three sessions and they
> are kept as written.** Their numbers were the headline *at the time* and are
> superseded by §0.9 (the baseline is now the placement a user deploys, not the
> best one measured) and §0.12 (both sides are re-timed from their own first
> dispatch). §4.0 carries the current reading, with each superseded one beside
> it.

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

*(All three rows are the ORACLE ROS baseline, RAW. §0.9 and §0.12 supersede
them; the first row still re-derives exactly and `reproduce.py` checks that it
does, so the move to §4.0's 1.1795 is checkable rather than asserted.)*

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
the "best measured" pool. (§0.11's coverage later takes it to 1.0045; §5.)

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

**0.9 — The primary baseline is now the placement a ROS user would actually
choose, not the best one measured. This is the single largest change in this
document and it reverses the headline.**

§4's original ROS number was `ros_np_best_ms`: enumerate every legal
`network -> backend` map, measure them all on hardware, and quote the fastest.
That was chosen to make the baseline *fair* rather than a straw man (§6), and
as an upper bound on pinning it still is — but it is **a placement oracle no
user has**. Nobody enumerates 27 placements and runs each one three times on
the board before deploying. It flatters ROS by exactly the amount that search
is worth, which on this matrix is up to **1.86×** on a single cell.

**The primary baseline is now `isolation-best`: each network pinned to the
backend it is fastest on *by itself*, chosen from a per-model benchmark with no
knowledge of what else is running.** That is the deployment rule — read the
per-model numbers, pick the best lane for each model, ship it. It yields
exactly one placement per cell, needs no search, and it is what the whole-model
cost table in SETUP.md §3.2 already says: every `mlp_control` wants the CPU,
everything else wants the DSP.

*How the lane is chosen, precisely.* Strict argmin of the whole-model cost —
the sum of that network's binding tiles on one lane — out of `model_costs.json`,
which is derived from the same frozen `cost_model.json` the XPU-RT sweep solved
against and which already excludes the `cpu@int8` alias. Three things about
the rule, all checked rather than assumed:

* **Ties never arise.** The stated rule breaks an exact tie by a fixed lane
  order (`dsp, cpu, hta, gpu`), and **it never fires**: the smallest margin
  between any network's best lane and its runner-up is **12.3 %**
  (`yolov8_nano_sc`, DSP 3.461 ms against CPU 3.887 ms), which is outside the
  ±9.18 % band. No selection in this matrix is a coin flip.
* **GPU never wins an unrestricted argmin**, for any of the 17 networks. So
  excluding it as a pinning candidate (SETUP.md §1) changes no selection, and
  the rule would pick the same lanes if the GPU were on the table.
* **Where the isolation-best lane is not legal in a cell, the network falls
  back to its best legal lane and that is recorded** (`iso_fallbacks` in
  `results/analysis.json`). This happens only on the two-lane configs — in
  `hd` there is no CPU, so `mlp_control_sd` goes to the DSP at 523.6 µs where
  the CPU runs it in 110.2 µs. **On every `quad` cell there are no fallbacks
  at all**, which is one more reason the `quad` column is the headline scope
  (§0.10). A user on a two-lane machine makes the same substitution, so this
  is the rule behaving correctly and not an approximation of it.

*It piles networks onto one lane, and that is the result.* On
`depth_contended_quad` the rule puts `fastdepth`, `dronet_sf` and
`yolov8_nano_sc` all on the DSP, because each prefers it alone; they serialise
and the timed `yolov8_nano_sc` finishes at 6.04 ms where the scheduler — which
moves `fastdepth` to the GPU — finishes at 4.22 ms. That is the realistic
failure mode of naive per-model lane selection, not a defect in the baseline,
and `plots/gantt_depth_contended_quad.png` shows it directly.

**Both baselines are reported and each is labelled.** `oracle` is kept as the
secondary reading and as the upper bound on what pinning could reach with
perfect knowledge; the **gap between the two is itself a result** — it is what
a naive user leaves on the table (§4.4). Job §0.11's enumeration is what makes
that bound meaningful.

**0.10 — The headline scope is `quad`, and the other three configs are a
labelled sensitivity study.** The config axis is which lane *subset* is
available: `hd` = hta+dsp, `dc` = dsp+cpu, `cg` = cpu+gpu, `quad` = all four.
**Only `quad` describes hardware that exists** — a QRB5165 always has all four
backends — so the other three model machines nobody can buy, and they distort
individual cells badly. `hd` has no CPU lane, so `mlp_control_sd` is forced
onto the DSP at 523.6 µs though the CPU runs it in 110.2 µs; that congests the
DSP and pushes the timed `yolov8` onto the HTA, and the cell reads as a 1.70×
pinning win. The same family in `quad` reads 0.987 — a tie. Every table below
gives the `quad` reading first and the all-configs one second, and the
all-configs figures are still produced.

**0.11 — The `quad` column is now enumerated, and `scale_ladder_quad` has a
stated budget rather than an unstated sample.** SETUP.md §5.4 enumerated in
full up to 8 legal placements and sampled above it, which left **7 of the 11
`quad` cells sampled and 788 legal placements unmeasured**. On a sampled cell
the "best placement" is the best of a targeted sample, which risks understating
what pinning can do — and that number is now the *secondary* reading, so it had
to be a real bound.

`scripts/pinsweep.py enumerate --coverage extended` (the default, and what the
committed plans hold) raises the full-enumeration threshold to **27 for the
`quad` column only**, which takes in six of the seven; `--coverage setup`
re-emits SETUP.md's pre-registered 172-assignment plan byte-for-byte and
`reproduce.py` checks that it does. **Nothing about scoring changes between the
two profiles: every assignment keeps the id, rank and predicted cost it had.**
The `hd`, `dc` and `cg` columns are untouched, because they are the sensitivity
study and re-measuring them would have cost 354 board runs to sharpen a number
that is not the headline.

| cell | n_legal | was | now | mode |
|---|---|---|---|---|
| `control_mix_quad` | 18 | 7 | **18** | full |
| `depth_chain_quad` | 9 | 5 | **9** | full |
| `depth_contended_quad` | 27 | 7 | **27** | full |
| `depth_nav_quad` | 18 | 6 | **18** | full |
| `saturation_quad` | 18 | 8 | **18** | full |
| `tight_loop_quad` | 12 | 5 | **12** | full |
| `scale_ladder_quad` | 729 | 5 | **64** | **sampled** |

*`scale_ladder_quad`'s budget and rule, in SETUP-style prose.* Exhaustive is
729 × 3 = 2187 runs, a different order of campaign. The budget is **64
placements**, and 64 is not a round number picked for its own sake: it is
exactly `n_legal` of `scale_ladder_dc` and `scale_ladder_hd`, both of which are
enumerated in full, so the third rung of the same family gets the same number
of measured placements as its two siblings and the three are comparable at
equal measurement effort. It is 8.8 % of the space. The set is chosen in five
parts, four deterministic and one seeded:

1. **The ranked head of both objectives, 8 deep each** — the cost model picked
   the measured best on 90.3 % of cells by the non-periodic objective, so the
   head is where the answer usually is; 8 rather than SETUP.md's 3 because the
   budget allows it.
2. **Every uniform placement** — all-CPU, all-DSP, all-HTA. What a team does
   when it does not think, and they bracket the space.
3. **The bottom-ranked placement**, so the measured span brackets the predicted
   range at the pessimistic end and not only at the optimistic one.
4. **The complete one-swap neighbourhood of the predicted best** — every
   placement differing from rank 0 in exactly one network's lane, 6 × 2 = 12.
   This is a full local-optimality test on the incumbent: if the placement the
   cost model recommends is beaten by moving a single network, this finds it,
   and a sample without it could not. The centre is the *predicted* best, not
   the measured one, so the plan stays a pure function of the frozen inputs.
5. **A seeded stratified fill to 64.** Strata are the number of networks on the
   DSP, 0..6 — every network in this family prefers the DSP by 2.5–4.4×, so DSP
   occupancy is the contention axis the cell turns on, and stratifying on it
   stops a flat random draw spending the whole budget in the middle of the
   binomial. Strata are visited round-robin, each drawing from its own members
   shuffled by `random.Random(20260909)`.

**`scale_ladder_quad` is reported as `sampled`, not `full`, and the tables and
figures say so.** A capped sample is not an enumeration however wide the cap.

*What the widened coverage found.* On the six now-enumerated cells, **five had
already found their best placement in the 5-to-8-point sample** and one moved:
`saturation_quad`, 8.835 → 8.675 ms, **1.85 % better** — inside the noise
floor, and that cell is excluded from the headline anyway (§0.2). On
`scale_ladder_quad` the widened sample **did** beat the old best: **2.638 →
2.383 ms, 10.7 % better**, just outside the ±9.18 % band. That 10.7 % is the
honest bound on how wrong the old sampled cells could have been, and it is
quoted rather than the reassuring five.

**0.12 — Both runtimes are now timed from their own first dispatch. This is a
harness correction, not a modelling choice.**

XPU-RT's measured non-periodic makespan is taken from t0 of its run loop, and
on some runs the run loop's start gate releases late. On
`perception_heavy_quad` the trace shows **both** first entries scheduled at
`predicted_start_ms = 0.000`, on two different lanes, both actually starting at
~1.62 ms, with `dep_wait_ms` of 0.002 and 0.028 — so nothing was blocked on a
dependency. The wait sits in `gate_ms` (1.606 and 1.617). yolov8's actual
execution is 6.462 − 1.622 = **4.84 ms** against ROS's ~4.73 ms, so the compute
agrees to ~2 % and **essentially the entire 1.35× "pinning win" on that cell was
the head start**.

It is **intermittent**, which is worse than constant: `perception_heavy_dc`
measures 0.025 ms on one rep and 1.638 ms on the next, so it moves a median of
3 rather than shifting every number equally.

The correction, applied **symmetrically on both sides and by one shared module**
(`scripts/offsets.py`):

    np_corrected = (end of the last aperiodic operation)
                 - (start of the FIRST dispatch anywhere in that run)

per rep, then the median over reps as before. Raw and corrected are **both**
reported everywhere; nothing is replaced. Neither runtime is charged for the
other's startup — that is the whole content of it. No schedule, no placement,
no cost model and no solver ranking is touched.

**Root cause is deliberately not chased here.** The runtime does two iterations
(`FLOWC_ITERATIONS=2`) and the trace is the second, so first-touch on the
fastRPC path or SCHED_FIFO lane spin-up bleeding across the iteration boundary
are the likely candidates. The evidence this campaign has is that the delay is
in `gate_ms` and not `dep_wait_ms`; **future work**, recorded in §15 with the
distribution that motivates it.

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

This campaign's own spread is much tighter: **3.86 % median** over 379
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
is no placement decision to make. That is a real narrowing and it shows: **the scheduler beats the
deployable pinning baseline on every one of the 7 non-degenerate `cg` cells**,
by 1.06× to 2.54×, which is the largest clean sweep of any config — with one
lane there is no placement to get right and nothing for the baseline to do.
The `cg` column also holds the second-worst rep spread in the campaign
(`scale_ladder_cg__a0`, 99.5 %), for the CPU-thread-pool reason in §1.

Three things the reference could not do, and this could:

* **The `fastdepth → dronet_sf` edge is wired, not dropped.** The micro-ROS
  reference had to run the depth families as independent timers and label them
  EDGE-DROPPED. Here `fastdepth`'s node publishes its instance index and
  `dronet_sf`'s node is driven by that subscription, so **all 12 depth cells
  are genuine `depth_*` results**. The wiring costs a publish→execute hop of
  **135.9 µs at p50** over 558 measured instances (p95 2.23 ms, but the tail is
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

min 1, median 4, max 729, **1078 legal assignments in total**, of which
**295 are measured** on hardware.

**39 of the 42 cells are now enumerated in full**, including ten of the eleven
`quad` cells (§0.11). The three still sampled are `scale_ladder_{dc,hd}` — 64
legal, 5 measured, left at SETUP.md's plan because they belong to the
lane-scarcity sensitivity study and not to the headline — and
`scale_ladder_quad`, 729 legal and **64 measured under the stated budget of
§0.11**. Every cell records which it got, and a capped sample is recorded as
`sampled` however wide the cap.

## 4. The headline: pinning vs scheduling on the objective

**The noise floor first.** The XPU-RT sweep's own measured rep spread is
**9.18 % median on this objective** (§1), and every "inside/outside" verdict
below is against that band.

**The scope is `quad`** — the only lane config that describes hardware that
exists (§0.10). **The baseline is `isolation-best`** — each network on the lane
it is fastest on alone, which is what a ROS user deploys (§0.9). **Both sides
are timed from their own first dispatch** (§0.12). Medians of 3 reps.

### 4.0 The number

| `quad`, 7 cells | vs `cpsat:warmbest` (the recommendation) |
|---|---|
| **isolation-best pinning ÷ XPU-RT, corrected — THE HEADLINE** | **1.1795** |
| pinning faster / scheduler faster | 1 / 6 |
| inside the ±9.18 % noise floor | 2 |
| *the same, raw (uncorrected)* | *1.1571 — 2 / 5, 1 inside* |
| **best-of-enumeration ÷ XPU-RT, corrected — the placement oracle** | **0.9976** |
| pinning faster / scheduler faster | 4 / 3 |
| inside the ±9.18 % noise floor | 3 |
| *the same, raw* | *0.9374 — 5 / 2, 2 inside* |

**Read that as: on the machine that exists, a team writing ordinary ROS nodes
and picking each model's best lane from a per-model benchmark is a median 18 %
SLOWER than XPU-RT running its recommended solver — and a team that somehow
knew the best joint placement would be at a dead heat with it (0.9976, three
of seven inside the noise band).** The whole of the difference between those
two rows is the value of the placement search, and no user has it.

The all-configs reading is the secondary scope, and it is closer because the
two-lane configs remove the alternative the scheduler exploits:

| 26 cells, all configs | corrected | raw |
|---|---|---|
| isolation-best ÷ `cpsat:warmbest` | **1.0768** (11 / 15, 6 inside) | 1.0396 (13 / 13, 4 inside) |
| oracle ÷ `cpsat:warmbest` | 0.9880 (16 / 10, 8 inside) | 0.9281 (17 / 9, 6 inside) |
| oracle ÷ best measured solver | 0.9989 (13 / 13, 9 inside) | 0.9762 (15 / 11, 8 inside) |

The bottom-right cell, **0.9762**, is what the previous version of this
document quoted as the headline, and the top-left, **1.0768**, is the same data
read two corrections later. Both are here on purpose.

### 4.0.1 Per cell, `quad`, primary reading

`iso` is the isolation-best placement; `wb` is `cpsat:warmbest`; both columns
are corrected, with the raw ratio beside for comparison.

| cell | iso ms | `wb` ms | **iso ÷ wb** | raw | band | oracle ÷ wb |
|---|---|---|---|---|---|---|
| `bimodal_quad` | 6.686 | 6.719 | **0.995** | 0.937 | inside | 0.995 |
| `perception_heavy_quad` | 5.018 | 4.847 | **1.035** | 0.772 | inside | 0.998 |
| `scale_ladder_quad` | 3.425 | 3.063 | **1.118** | 1.128 | outside | 0.773 |
| `control_mix_quad` | 3.963 | 3.360 | **1.180** | 1.157 | outside | 1.004 |
| `depth_contended_quad` | 5.972 | 4.217 | **1.416** | 1.410 | outside | 0.768 |
| `vint_intro_quad` | 121.034 | 30.465 | **3.973** | 3.939 | outside | 3.536 |
| `vint_multi_quad` | 126.508 | 30.445 | **4.155** | 4.131 | outside | 3.148 |

**Only `bimodal_quad` has pinning ahead, and it is inside the band, so it is a
tie rather than a win.** The two cells inside the band are the two where the
isolation rule happens to land on a placement with no contention: `bimodal` and
`perception_heavy` are each one vision network plus one `mlp_control`, and the
rule puts them on different lanes.

### 4.1 Where the scheduler wins, and why

All configs, isolation-best, corrected; the eight largest of the fourteen
outside the noise band:

| cell | ROS ÷ XPU-RT | mechanism |
|---|---|---|
| `vint_multi_quad` | **4.16×** | per-tile placement |
| `vint_intro_quad` | 3.97× | per-tile placement |
| `vint_intro_dc` | 2.73× | per-tile placement |
| `scale_ladder_cg` | 2.54× | GPU excluded → one lane for six networks |
| `vint_multi_dc` | 2.47× | per-tile placement |
| `depth_contended_dc` | 1.75× | **three networks on one lane, because each prefers it alone** |
| `depth_contended_cg` | 1.48× | one lane for three networks |
| `vint_multi_cg` | 1.46× | per-tile placement + one lane |

**Five of the eight are `vint`, and `vint` is the cleanest forfeiture in the
matrix.** Its encoder tile composes on {dsp, cpu}, its decoder on {cpu, gpu};
the intersection is CPU alone, so a whole-model pin pays 84.163 + 37.826 =
**121.989 ms**, while the scheduler puts the encoders on the DSP at 14.213 ms.
SETUP.md §3.2 predicted this before anything ran and the board confirms it:
`vint_intro_quad` is 121.0 ms pinned against 30.5 ms scheduled. This is the
single result that most justifies a per-op scheduler on this hardware, and it
has nothing to do with scheduling *policy* — it is about being allowed to cut
the model at all.

**`depth_contended_*` is the new entry and it is the naive-placement failure
mode itself.** `fastdepth`, `dronet_sf` and `yolov8_nano_sc` each prefer the
DSP in isolation by 1.33×, 2.46× and 2.29×, so the rule puts all three there
and they serialise. On `depth_contended_quad` the timed `yolov8_nano_sc`
finishes at 5.97 ms; the oracle placement — `fastdepth@cpu`, `dronet_sf@hta`,
`yolo@dsp` — finishes at 3.24 ms, and the scheduler, which moves `fastdepth` to
the GPU, at 4.22 ms. **The scheduler beats the deployable baseline by 1.42× and
loses to the oracle by 1.30×**, and the whole of that band is the placement
decision. `plots/gantt_depth_contended_quad.png` and its `_oracle` companion
show both.

### 4.2 Where pinning still wins, and why

All configs, isolation-best, corrected; seven are outside the band with pinning
faster, and **every one of them is a two-lane config**:

| cell | ROS ÷ XPU-RT | isolation placement |
|---|---|---|
| `scale_ladder_hd` | **0.53×** | all six on DSP |
| `scale_ladder_dc` | 0.56× | all six on DSP |
| `control_mix_hd` | 0.59× | everything on DSP |
| `bimodal_hd` | 0.69× | both on DSP |
| `perception_heavy_hd` | 0.71× | both on DSP |
| `control_mix_cg` | 0.80× | everything on CPU (the only lane) |
| `perception_heavy_cg` | 0.86× | both on CPU (the only lane) |

**Not one is a `quad` cell.** That is the finding, and it is the reason §0.10
makes `quad` the headline scope: the cells where naive pinning beats the
scheduler are the cells where the scheduler had nowhere to move the work
either. On `hd` there is no CPU lane, so the periodic `mlp_control` is forced
onto the DSP for both runtimes; the pinned node then runs the timed network as
one uninterrupted dispatch chain while the scheduler interleaves periodic
instances into the same lane and pays a gate per entry. Give the scheduler the
fourth backend back and the advantage disappears — `perception_heavy` reads
0.71× on `hd` and 1.04× on `quad`, on the same two networks.

So the honest statement is no longer "pinning beats the scheduler on 17 of 26
cells". It is: **on hardware that exists, choosing each model's best lane and
deploying loses to the recommended solver at a median 1.18× on the `quad`
column and 1.08× over all 26 cells; where it wins, it wins on machines that do
not exist.**

### 4.3 The twelve cells with no aperiodic network

`depth_chain`, `depth_nav` and `tight_loop` are all-periodic, so the
non-periodic objective degenerates to the all-operations makespan exactly as
`evaluate()` does. For these the wall clock *is* the comparison, and it is a
valid one: both sides executed identical entry counts (checked per cell).
Isolation-best ÷ `cpsat:warmbest`: median **0.9811** raw and **0.9958**
corrected, 11 of 12 inside the noise floor corrected. A dead heat, which is the
honest reading — with every network periodic and every release fixed, there is
very little for either runtime to decide, and the isolation rule and the oracle
pick nearly the same placement because there is no aperiodic network whose lane
is worth fighting for.

**A caveat specific to these twelve, and the one place the offset correction is
not obviously right.** Subtracting the head start is correct when the timed
interval *ends* on work that inherited it. On an all-periodic cell the last
operation is often pinned by a periodic **release**, and the release clock is
**not** delayed by the start gate — in the `perception_heavy_quad` trace, entry
2 is scheduled at 4.189 ms and starts at 4.196 ms, on time, in the very run
whose first dispatch was 1.6 ms late. Where that happens the end does not move
and subtracting the offset flatters that run. `offsets.release_bound()` flags
those runs from the trace and `results/analysis.json` records them
(`release_bound_reps`); five of the twelve have at least one flagged rep, all
in `depth_chain` and `tight_loop_dc`. **The two cells whose corrected number
moves more than the band — `depth_chain_dc` +12.5 % and `depth_chain_quad`
+12.6 % — are both flagged, so they are reported and not treated as results.**
None of this touches the headline: all twelve are outside it by construction.

### 4.4 What the placement search is worth — the gap between the naive rule and the oracle

This is the quantity the two baselines bracket, and it is worth reporting on
its own: **isolation-best ÷ best-of-enumeration, per cell, both measured.**

| scope | median | max | naive rule already optimal |
|---|---|---|---|
| `quad`, 7 cells | **1.172** | **1.865** (`depth_contended_quad`) | 1 of 7 |
| all configs, 26 cells | 1.000 | 1.865 | 15 of 26 |

**On the machine that exists, the naive rule leaves a median 17 % and a worst
case of 86 % on the table.** On the two-lane configs it leaves nothing at the
median — 15 of 26 cells have only one sensible placement or none at all — which
is another way of saying the same thing as §4.2: lane scarcity removes the
decision, and removing the decision is what made the old baseline look strong.

The five `quad` cells where the gap is real: `depth_contended_quad` 1.865×,
`scale_ladder_quad` 1.456×, `vint_multi_quad` 1.320×, `control_mix_quad`
1.172×, `vint_intro_quad` 1.123×. On `bimodal_quad` the isolation rule *is* the
oracle, and on `perception_heavy_quad` it is 1.038× off it.

**This is also why §0.11's enumeration mattered.** The oracle column is only an
upper bound if it is a real one; on six of the seven sampled `quad` cells it now
comes from a complete enumeration, and on the seventh from a 64-placement
budgeted sample that found a placement 10.7 % better than the old five-point
one.

## 5. The wall clock, reported second

Over all 42 cells the all-operations makespan gives median **1.0045**, 13 cells
faster, 29 slower, 25 inside the noise floor. It is the weaker comparison and
the reason is structural: on a cell with a long periodic tail the wall clock is
pinned by the last periodic **release**, `(n−1)·period`, which is a property of
the workload and not of either runtime. `bimodal_dc` is the clean example —
32.533 ms pinned against 32.424 ms scheduled, and 31·1.045 = 32.4 ms of that is
just the clock ticking. Quoting it as the headline would have reported a
near-perfect tie on 25 cells where nothing was being measured.

## 6. What choosing the lane is worth, measured

Median placement spread (worst ÷ best measured legal placement) is **1.04** on
the wall clock and the maximum is 9.44×. On the objective it is far larger —
up to **9.44×**, and the widened coverage of §0.11 is what exposes the top of
that range:

| cell | n_legal | measured | best np | worst np | spread |
|---|---|---|---|---|---|
| `scale_ladder_quad` | 729 | 64 | 2.383 | 22.496 | **9.44×** |
| `control_mix_quad` | 18 | 18 | 3.414 | 24.080 | 7.05× |
| `depth_contended_dc` | 8 | 8 | 3.641 | 17.781 | 4.88× |
| `control_mix_dc` | 8 | 8 | 3.492 | 15.477 | 4.43× |
| `scale_ladder_dc` | 64 | 5 | 3.463 | 14.754 | 4.26× |

**So the placement decision is worth up to 9.4× on the same cell.** That is why
this document reports two ROS baselines rather than one: quoting only the best
measured placement would credit a deployment with a search worth up to 9.4×
that nobody runs, and quoting only the naive one would leave the reader unable
to tell a scheduling effect from a placement effect. The spread here is the
full width of the space; §4.4 is the part of it a naive user actually loses,
which is smaller — a median 1.17× on `quad` — because the isolation rule is a
good rule, just not an omniscient one.

## 7. The ranking: what the cost model got right, and where the rule matters

**The whole-model cost model picks the placement.** Its predicted-best
assignment was the measured-best on **83.9 %** of cells by wall clock and
**90.3 %** by the non-periodic objective. Where it missed, it missed by little.
Both rates fell slightly against the pre-extension figures (87.1 % / 93.5 %),
and that is the expected direction: enumerating the rest of the `quad` column
gave the board more chances to beat the prediction, and on two cells it took
them.

**The reference's §6.5 replicates.** Ranking on makespan alone is not the same
as ranking on what you would ship, and on **7 of 42 cells the two rules
disagree**:

| cell | fastest placement | its window misses/rep | feasible-first placement | its misses | cost |
|---|---|---|---|---|---|
| `saturation_quad` | `a1` 20.605 ms | 1.0 | `a6` 21.473 ms | **0** | +4.2 % |
| `saturation_hd` | `a0` 19.907 ms | 9.7 | `a3` 20.709 ms | 6.0 | +4.0 % |
| `saturation_dc` | `a2` 20.811 ms | 7.7 | `a1` 20.850 ms | 0.7 | +0.2 % |
| `bimodal_quad` | `a0` 32.473 ms | 0.7 | `a1` 32.531 ms | **0** | +0.2 % |
| `control_mix_hd` | `a0` 42.780 ms | 3.3 | `a1` 42.848 ms | 1.0 | +0.2 % |
| `control_mix_quad` | `a6` 42.587 ms | **3.7** | `a9` 42.652 ms | **0** | +0.2 % |
| `depth_nav_hd` | `a1` 31.960 ms | 5.3 | `a3` 31.974 ms | 0.7 | +0.04 % |

Still 7 of 42, and the enumeration made the trade *cheaper* rather than
rarer: `control_mix_quad`'s full 18 placements contain a zero-miss placement
only 0.2 % slower than the fastest, where the 7-point sample's cheapest
zero-miss option cost 1.9 %. **0.2 % more makespan buys the elimination of
every window miss on that cell.** Window feasibility is deliberately kept out of the ranking key
and reported separately, exactly as the brief requires — but a reader choosing
a placement from this data should read both columns.

**Starvation, by contrast, did not replicate, and could not have.** SETUP.md
§5.3 said so before the campaign: this harness runs a finite taskset to
completion, so no instance can be dropped. **0 of 379 assignments starved a
network**, and the `(starved, makespan)` key therefore reduces to makespan on
every completed run. The reference's starvation arose because its window was
closed by a one-shot and a co-resident could vanish; nothing here can. Reported
as a structural difference, not as a null result.

## 8. Where the whole-model cost model breaks

Median absolute error **3.02 %** over 379 assignments, p90 **47.5 %**, range
**−70.4 % .. +127.1 %**. The median is excellent and the tails are the whole
story — and the widened `quad` coverage made both tails longer, because the
placements it added are exactly the mixed-lane ones the model has no term for.

**It over-predicts when it charges for contention that does not happen:**

| assignment | predicted | measured | err |
|---|---|---|---|
| `scale_ladder_quad__a685` | 9.911 | 2.936 | **−70.4 %** |
| `scale_ladder_hd__a63` | 14.622 | 4.604 | −68.5 % |
| `scale_ladder_quad__a679` | 9.836 | 3.324 | −66.2 % |
| `scale_ladder_quad__a727` | 12.699 | 4.837 | −61.9 % |

All four pile several networks onto **HTA**. The model serialises them at their
measured solo cost; the hardware does not, because a QNN HTA context's
host-side call overlaps another context's accelerator time. The model's
serial-FIFO assumption is simply wrong for that lane.

**It under-predicts when several networks share the CPU:**

| assignment | predicted | measured | err | rep spread |
|---|---|---|---|---|
| `scale_ladder_quad__a487` | 6.734 | 15.291 | **+127.1 %** | 96.8 % |
| `scale_ladder_cg__a0` | 9.936 | 22.496 | +126.4 % | 99.5 % |
| `scale_ladder_quad__a687` | 9.936 | 22.496 | +126.4 % | 99.5 % |
| `scale_ladder_quad__a4` | 2.664 | 5.332 | +100.1 % | 2.9 % |

(`scale_ladder_quad__a687` is the all-CPU placement, byte-identical to
`scale_ladder_cg__a0` as a harness input; it was resolved as a content
duplicate and consumed no board time — §0.11, §12.)

Every one is two or more networks on the CPU lane. The per-network cell was
measured with QnnCpu's thread pool having the machine to itself; two such pools
do not add, they multiply, and the rep spread goes with them. **This is the
same confound the QRB5165 sweep already has on record** — "the CPU-lane
contention term the cost model has no way to express" — reproduced from the
other side, and it is why the cost model is used here only to *rank* candidates
and never to report a number.

Split by structure: median absolute error **3.54 %** where two networks share a
backend, **2.37 %** where each has its own. The tails, not the medians, carry
the failure.

**This is the mechanism behind §4.4's gap, seen from the model's side.** The
isolation rule is a pure function of the solo costs, and the solo costs are the
one thing the model gets right; what it cannot express is what happens when the
rule sends three networks to the same lane. So the naive rule is not naive
because the numbers it reads are wrong — they are the same numbers the
scheduler solves against — but because the quantity it optimises has no
contention term at all.

## 9. Window feasibility, reported separately

Never folded into any makespan. The cells that miss windows at the **oracle**
placement are `saturation_{cg,hd,dc}` (11.7 / 9.7 / 7.7 instances per rep),
`bimodal_hd` (9.0), `depth_nav_hd` (5.3) and `control_mix_{quad,hd}` (3.7 /
3.3). Every one is a cell where a periodic network's whole-model latency on its
pinned backend is a large fraction of its own period — `mlp_control_sf` at
0.527 ms on the DSP against a 0.791 ms period, for instance. Whole-model
pinning cannot shorten a network to fit; that is the constraint being modelled.

**The naive placement is not uniformly worse here, and on four cells it is
better.** Isolation-best against the oracle, misses per rep: `saturation_dc`
**1.3 against 7.7**, `depth_nav_hd` **0.7 against 5.3**, `control_mix_quad`
**0 against 3.7**, `control_mix_hd` **1.0 against 3.3**; against it,
`saturation_hd` 15.3 against 9.7 and `saturation_quad` 2.3 against 1.0. The
reason is §7's: the oracle is ranked on makespan with feasibility deliberately
outside the key, so it will trade window misses for a faster makespan and the
isolation rule — which is not optimising the makespan at all — sometimes will
not. **A reader choosing a deployment from this data should read both columns,
and the faster baseline is not always the one that holds its cadence.**

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

### 10.0 The same two changes, on this arm

**The primary baseline here is isolation-best too, and both sides are timed
from their own first dispatch.** This arm needed no new board time for either:
it was already fully enumerated, so the isolation placement is a re-selection
over runs already on record, and the correction is derived from traces already
on disk (`xpurt3net.py rescan`).

| 7 shapes with an aperiodic network | vs `cpsat:warmbest`, corrected | raw |
|---|---|---|
| **isolation-best ÷ XPU-RT — the primary** | **0.9488** (5 / 2, 4 outside the band) | 0.9482 |
| best-of-enumeration ÷ XPU-RT — the oracle | 0.9311 (7 / 0, 2 outside) | 0.9305 |

**On this arm the direction does not change and pinning stays ahead**, unlike
the main arm's `quad` column. Two reasons, and both are properties of the
workload rather than of the runtimes:

* **The isolation rule is nearly optimal here.** On 2 of the 7 shapes it *is*
  the oracle, and the median gap is only **1.018×** (max 1.229× on
  `3net_dronet8_mlp16_yolo1`). These shapes are three networks with strongly
  separated preferences — `yolov8n` and `dronet` want the DSP, `mlp_control`
  wants the CPU by 10× — and only `3net_dronet8_mlp16_yolo1` and
  `3net_fused2_mlp8_yolo1` load the DSP hard enough for the naive choice to
  hurt. Where it does hurt it hurts a lot: those two go from 0.930 / 0.970 at
  the oracle to **1.144 / 1.167** naive, i.e. they cross over to the scheduler.
* **The offset correction barely moves this arm.** Its XPU-RT offsets are a
  median 0.033 ms against the main arm's 0.064, and the shapes with a real head
  start (`3net_fused4_mlp4_yolo1` 1.354 ms, `3net_fused2_mlp8_yolo1` 1.292 ms)
  are timed against a ~30 ms objective, so 1.3 ms is 4 %. The main arm's
  affected cells were timed against 5–7 ms objectives, where the same 1.6 ms is
  a third of the number.

The one shape whose correction is large is `3net_mlp2` — 10.067 → 8.087 ms,
−20 % — and it is **all-periodic**, so it is outside the comparison for the same
reason §4.3's twelve are, and it is release-bound in the sense §4.3 describes.

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

(That table is the **oracle** ROS baseline, raw — the reading this section was
originally written against. §10.0 has the primary.)

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
* **Three cells are still sampled, not exhausted, and one of them is in the
  headline.** After §0.11, 39 of 42 cells are enumerated in full and 295 of the
  1078 legal assignments are measured. What remains:
  `scale_ladder_{dc,hd}` — 64 legal, 5 measured each, deliberately left at
  SETUP.md's plan because they are lane-scarcity sensitivity cells and not the
  headline; and **`scale_ladder_quad`, 729 legal and 64 measured**, which *is*
  in the headline and is reported as `sampled` everywhere. On that cell the
  oracle number is the best of 64 placements chosen by a stated rule, not a
  minimum over the space, and the honest bound on the error is the 10.7 % the
  widened sample itself found against the previous 5-point one. A still-better
  placement in the unmeasured tail would move the oracle column further from
  the naive one — i.e. it would make §4.4's gap larger, not smaller, and would
  not touch the primary reading at all, because the isolation placement on that
  cell is a *named* placement (`a143`, all six on the DSP) and is measured.
* **Root cause of the start barrier.** Diagnosed to the gate (§0.12, §15) and
  deliberately not chased further; recorded as future work with the evidence.
* **GPU as a pinning lane.** Excluded by design (SETUP.md §1). Its cost is
  concentrated and reported: it is what makes all 11 `cg` cells degenerate.
* **A second measured point for the isolation baseline.** The isolation rule
  yields one placement per cell, so its number is one placement's median over
  3 reps and it has no within-cell spread to quote. That is not a gap in the
  measurement — it is what the baseline *is* — but it does mean a reader cannot
  see how sensitive the primary reading is to placement, and §4.4 is the answer
  to that question rather than an error bar on it.
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

**The `quad` coverage campaign of §0.11 is the fourth session, and the only one
that needed the board.** Same discipline: `timeout -s KILL` around `ssh -n`
around `flock -w 900 /tmp/qnn_board.lock -c`, 4 assignments batched to a lock
acquisition. **112 new placements, 336 runs, 92 batched calls, 12.5 s median /
26.5 s maximum wall per call including lock wait**; no contention from another
tenant. All 336 reported `N/N entries executed`; none discarded. The governor
was read (`performance`), forced to `performance` on all 8 cores, and restored
to `performance` after. `/data/tombstones/cdsp` was at **2.4 G** and was cleared
before the first run, taking the board from 39 % / 61 G to **36 % / 63 G**; it
regrew to **3.4 G** over the 336 runs and the board finished at 40 % / 60 G.
The same per-session chore, at the same rate, for the fourth time.

**11 further placements consumed no board time because they were the same
experiment.** The harness config is never told which lanes a cell *declares* —
only where each network goes — so a `quad` placement using no HTA is
byte-identical to the `dc` placement of the same family. `drive.py::cfg_hash`
hashes the config with its provenance comments stripped, and a tag whose hash
matches one already measured records `duplicate_of` and reads that run's log.
The dedupe is cross-cell and it is recorded per assignment (`measured_via` in
`measured.json`), so no number is silently shared between two rows:
`control_mix_quad__a{12,13}`, `saturation_quad__a{3,4,11,12,14}`,
`scale_ladder_quad__a{48,82,687}` and `tight_loop_quad__a8`. **123 placements
marked, 112 measured, 11 deduped.**

**The offset correction of §0.12 needed no board time on either arm.** Both
sides' traces were already on disk; `scripts/offsets.py` re-reads them and
`drive.py collect` / `xpurt3net.py rescan` re-derive from them.

`reproduce.py` re-derives `model_costs.json`, the expressibility verdicts, the
enumeration under **both** coverage profiles, every plan, the isolation-best
placement of every cell, every median in `measured.json` and the offset
correction on both sides, from the frozen inputs and the run logs — and
re-checks the two noise-floor figures against the XPU-RT sweep's own
`phase4_results.json`. It needs no hardware.

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

`scripts/plot_comparison.py` takes `--against {warmbest,best}`,
`--config <lane config>`, `--baseline {isolation,oracle}` and
`--timing {raw,corrected}`, and every combination writes its own file whose
name and on-figure text state what it is. The defaults are the headline:
`--against warmbest --baseline isolation --timing corrected`.

| file | what |
|---|---|
| `plots/ros_vs_xpurt_objective_warmbest_iso_corrected_quad.png` | **the headline, and the one to report**: isolation-best pinning ÷ XPU-RT running `cpsat:warmbest`, both re-timed from their own first dispatch, `quad` only. Log ratio axis, the 7 compared cells separated from the 4 excluded with their reasons named (`--config quad`) |
| `plots/ros_vs_xpurt_objective_warmbest_corrected_quad.png` | the same scope and timing against the **oracle** placement — the upper bound on pinning (`--baseline oracle`) |
| `plots/ros_vs_xpurt_objective_warmbest_iso_quad.png`, `..._warmbest_quad.png` | the same two, **raw**, so the effect of the offset correction is a file comparison and not a claim |
| `plots/ros_vs_xpurt_objective_warmbest_iso_corrected.png` | the headline over **all 26 cells** — the secondary scope, with the three lane-scarce configs in it |
| `plots/ros_vs_xpurt_objective_warmbest.png` | oracle, raw, all configs: **the reading the previous version of this document quoted as its headline**, kept unchanged so the move is visible |
| `plots/ros_vs_xpurt_objective_iso_corrected_quad.png`, `..._objective_corrected_quad.png` | the headline scope against the **best measured solver** instead of the recommendation (`--against best`) |
| `plots/ros_vs_xpurt_nonperiodic.png` | the same data, first draft — linear axis and all 42 cells in one ranking with the excluded greyed. Superseded; kept because §4's cell-by-cell text was first written against it |
| `plots/ros_vs_xpurt.png` | the all-operations wall clock |
| `plots/placement_value.png` | worst ÷ best measured legal placement, per cell |
| `plots/costmodel.png` | predicted vs measured, split on whether two networks share a backend |
| `plots/ros_vs_xpurt_3net.png` | **the 3net arm, both baselines on one figure**: bars are isolation-best ÷ `cpsat:warmbest` corrected, open markers the oracle placement on the same axis, so **the gap between marker and bar is what the naive rule leaves on the table** rather than a second file a reader might quote instead. Log ratio axis with reciprocal limits — this arm spans 0.87–1.17 and the main arm's wide axis would compress every bar into a smear at 1.0 (`scripts/analyse.py plots`) |
| `plots/gantt_<cell>.png` | XPU-RT vs ROS execution traces, both from the measured trace blocks. **The ROS panel draws the isolation-best placement** — the one the headline ratio is computed on — and says so in its title; `--baseline oracle` writes `gantt_<cell>_oracle.png` instead of overwriting it, because the two panels show different placements. **The pre-first-dispatch dead time is shaded on both panels and the first dispatch is marked**, so the start barrier is visible rather than described, and the title carries the corrected ratio with the raw one beside it (`--prefer cpsat:warmbest --baseline isolation`) |

**Which cells get a gantt, and why.** Six `quad` cells, chosen for the primary
reading: `depth_contended_quad` (1.42×) is the naive-placement failure mode
itself — three networks on the DSP because each prefers it alone — and its
`_oracle` companion is the same cell with the placement a search would find;
`perception_heavy_quad` (1.04×) is the start-barrier cell §0.12 is about, and
is the figure that previously illustrated a startup artifact as a scheduling
difference; `control_mix_quad` (1.18×) and `scale_ladder_quad` (1.12×) are the
two other cells where naive placement loses outside the band;
`vint_intro_quad` (3.97×) is the per-tile forfeiture; `bimodal_quad` (0.995×)
is the one cell where the naive rule ties the scheduler, drawn so the tie has a
mechanism on the page too. Eight more cover the lane-scarcity study
(`control_mix_hd`, `perception_heavy_hd`, `bimodal_hd`, `depth_contended_cg`,
`vint_multi_cg`, `vint_intro_dc`, `scale_ladder_{hd,dc}`).

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

## 15. The start barrier: the evidence, and what the correction did

§0.12 states the correction. This section is the evidence for it, because a
correction applied to a headline has to be checkable.

### 15.1 The per-run offset distribution, both harnesses

First dispatch of a run, in that run's own time base, every rep counted
separately:

| harness | runs | median | p75 | p90 | max | runs > 1 ms |
|---|---|---|---|---|---|---|
| **XPU-RT**, main arm | 366 | **0.064 ms** | 0.236 | 0.770 | 2.120 | **30** |
| ROS pinning, main arm | 885 | **0.032 ms** | 0.050 | 0.137 | 1.993 | 4 |
| ROS pinning, 3net arm | 252 | 0.045 ms | 0.060 | 0.159 | 2.114 | 2 |

**Both harnesses have the delay; XPU-RT has it twice as often at the median and
eight times as often past 1 ms.** That asymmetry is why correcting only one
side would be wrong, and why correcting neither was also wrong.

It is concentrated, not spread: ten XPU-RT points carry a median offset above
1 ms, and they are `perception_heavy_{quad,dc,hd}`, `depth_chain_{dc,quad}`,
`saturation_dc` (two solvers), `tight_loop_{hd,quad}`. **On the rest of the
matrix the correction is worth less than the third decimal.**

### 15.2 That it is the gate, not a dependency

On `perception_heavy_quad__greedy` rep 1, the two entries scheduled at
`predicted_start_ms = 0.000` sit on different lanes and both start at ~1.62 ms:

| entry | lane | predicted start | actual start | `dep_wait_ms` | `gate_ms` |
|---|---|---|---|---|---|
| `mlp_control_sd#0` | cpu | 0.000 | 1.615 | **0.002** | **1.606** |
| `yolov8_nano_sf#0` | dsp | 0.000 | 1.622 | **0.028** | **1.617** |

Nothing was dependency-blocked and nothing was queued behind anything: the wait
is at the gate. And the gate delay does **not** propagate to the release clock
— entry 2 of the same run is scheduled at 4.189 ms and starts at 4.196 ms, on
time — which is both the reason §4.3's all-periodic cells need the caveat they
have and the reason the delay looks like a one-off barrier rather than a clock
skew.

**Future work, not chased here.** The runtime does two iterations
(`FLOWC_ITERATIONS=2`) and the trace is the second, so first-touch on the
fastRPC path or SCHED_FIFO lane spin-up bleeding across the iteration boundary
are the candidates. Neither is tested by this campaign.

### 15.3 What the correction moved

**Three of the 26 comparable cells move by more than the ±9.18 % noise floor,
and all three are the same family:**

| cell | isolation raw → corrected | move | oracle raw → corrected | move |
|---|---|---|---|---|
| `perception_heavy_quad` | 0.772 → **1.035** | **+34.2 %** | 0.743 → **0.998** | +34.2 % |
| `perception_heavy_dc` | 0.776 → **1.024** | **+32.0 %** | 0.776 → **1.024** | +32.0 % |
| `perception_heavy_hd` | 0.635 → **0.715** | **+12.5 %** | 0.635 → 0.715 | +12.5 % |

On the `quad` scope alone it is **1 of 7** (`perception_heavy_quad`). Every
other comparable cell moves by less than the band; the median move is +0.42 %
over all 26 and +0.85 % over the 7.

**Cells that change direction** — which is not the same question, because a
cell sitting at 1.00 can cross on a sub-noise move:

* isolation, `quad`: **`perception_heavy_quad`** (0.772 → 1.035, pinning-faster
  to scheduler-faster; corrected it is inside the band, so it becomes a tie).
* oracle, `quad`: **`control_mix_quad`** (0.987 → 1.004) — inside the band on
  both readings, so nothing turns on it.
* over all 26: isolation adds `perception_heavy_dc`; oracle adds
  `depth_contended_dc` (1.068 → 0.996) and `perception_heavy_dc`.

**`perception_heavy` is the whole story, and it is the family that was drawn.**
Its cells are two networks and a 5–7 ms objective, so a 1.6 ms head start is a
third of the number. `plots/gantt_perception_heavy_quad.png` previously showed
a 1.35× "pinning win" that was 1.6 ms of dead time on the XPU-RT panel; it now
shades that dead time, marks both first dispatches, and reads **1.03× corrected
against 0.77× raw**, with both printed on the figure.

### 15.4 What it did not move

* **No solver ranking.** The correction subtracts a per-run constant from a
  makespan; it does not touch a schedule, an objective or a placement.
* **The wall-clock comparison** (§5), which is release-bound on most cells and
  is reported raw.
* **The 3net arm's direction** (§10.0): its offsets are smaller and its
  objectives are ~30 ms, so the medians move by 0.06 % and 0.07 %.
* **The noise floor.** The ±9.18 % band is the XPU-RT sweep's own pre-registered
  rep spread on the *raw* quantity and it is kept (§1). Recomputing it on the
  corrected quantity would narrow it — the correction removes a source of rep
  variance — and a narrower band promotes borderline cells into findings, which
  is exactly the move a pre-registered floor exists to prevent.
