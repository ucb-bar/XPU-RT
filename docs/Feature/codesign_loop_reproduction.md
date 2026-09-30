# Running the co-design feedback loop

This is the document to hand someone who asks "where is the automated script?". Every
command below is committed and runs from a fresh clone; the ones that need the physical
K1 board or the cross toolchain say so.

Paths are repo-relative. `$PY` is the repo venv (see [Environment](#environment)).

---

## What the loop is, in one table

| stage | what decides | script | needs |
|---|---|---|---|
| solve | the scheduler | `scripts/run_xpurt_schedule.py` | — |
| advise | the profile + the solved schedule | `scripts/emit_compile_advice.py` | — |
| bridge | one verb's hint from the advice | `scripts/advice_to_{fusion,unfuse,split,shard,kernel_choice}_hint.py` | — |
| apply | ModelBlaster rewrites the IR | `ModelBlaster/pipeline/apply_*_hint.py` | — |
| graph gate | did the dispatch graph change | `scripts/diff_dispatch_graph.py` | — |
| correctness gate | is the rewrite bit-exact | `ModelBlaster/scripts/verify_ir_rewrite_host.py` | — |
| rebuild + measure | the board | `ModelBlaster/scripts/run_model_k1.sh` | **toolchain + K1** |
| re-solve | the scheduler, on measured costs | `scripts/run_xpurt_schedule.py` | — |
| verdict | `candidate_objective.accept()` — nine terms | `scripts/compare_candidates.py` | — |

Two drivers chain those for you, and one composes the figure.

---

## 1. The scheduling levers, one command, no board

```bash
$PY scripts/run_codesign_loop.py \
    --workload data/toplevel/_4w_networks_k1_sensor_sharded_rich_shard_ime_s4.0.json \
    --solver cpsat --time-limit 45 --objective lateness \
    --board-calibration results/codesign_feedback/k1_board_calibration.json \
    --board-solver cpsat --board-time-limit 200 \
    --out-dir results/codesign_loop
```

Starts from the workload with every lever stripped, proposes each not-yet-applied lever
(`shard`, `ime`, `unfuse`) as a candidate, solves each with the real profiled scheduler,
and accepts on **`candidate_objective.accept()`** — nine lexicographic terms, hard
deadline misses first and makespan seventh, each with a noise tolerance. Then the
board-feedback arm re-costs the accepted schedule under measured K1 multipliers and
re-solves against them until the misses clear.

What it prints, on that workload:

```
round 1 · try ime:    accepted -- hard deadline misses: ime=0 beats baseline=2
round 1 · try shard:  accepted -- hard deadline misses: shard=0 beats baseline=2
round 1 · try unfuse: lever changed nothing on this spec (inapplicable here)
round 1: ACCEPT +shard   total-lateness 3.226 -> 0.000
round 2 · try ime:    rejected -- max deadline lateness: shard=0 beats ime=11.29
round 2: no lever is accepted by the nine-term objective rule — CONVERGED
board arm: instance-miss arc  baseline->AOT->board-recost->board-resolve = 2 -> 0 -> 2 -> 0
```

Read that last line as the whole point: the offline loop clears the misses, the board
says two came back, and re-solving on the measured costs clears them again.

`loop_report.json` records, per round, the accepted lever, the **deciding term**, the
nine-term values, and every rejected candidate with its reason. `--accept-rule legacy`
reproduces pre-hardening runs, which judged on `misses_not_worse AND delta > 0.05 ms`
and could accept a lever with 93 deadline misses on a makespan win.

## 2. The graph-rewrite arm (advice → bridge → applier → gates)

```bash
$PY scripts/run_modelblaster_arm.py \
    --schedule schedules/scheduled_<stem>_greedy_profiled.json \
    --ir ffn_block=<ModelBlaster build>/k1_xpurt/ffn_block/int8/graph.json \
    --out-dir results/modelblaster_arm/<name>
```

Drives all five verbs and records, per verb, how far it got and **why it stopped** —
"the advisor did not fire", "the applier refused" and "the rewrite is not bit-exact" are
three different findings. Nothing is marked `eligible` without the bit-exact host verify
(`--no-require-verify` overrides, and the report then says so on every row).

`shard` is exempt from the numeric gate on purpose: `apply_shard_hint` is annotation-only
— same dispatch count, same ids, one extra `shard_factor` field — so numerics cannot
change and there is nothing for a numeric check to check. Its remaining gate is a board
reprofile at that width.

The IRs are ModelBlaster build outputs (`pipeline/extract_graph.py`), not committed.

## 3. A board round: rebuild, reprofile, re-solve, adjudicate — **needs the K1**

```bash
eval "$(scripts/setup_spacemit_toolchain.sh)"     # GCC 14.3 for riscv64
$PY scripts/run_board_round.py \
    --workload data/toplevel/_4w_networks_k1_sensor_sharded_rich_shard_ime_s4.0.json \
    --net ffn_block \
    --baseline-ir  <build>/k1_xpurt/ffn_block/int8/graph.json \
    --candidate-ir results/modelblaster_arm/<name>/ffn_block.split.graph.json \
    --seed-build-dir <build>/k1_xpurt/ffn_block/int8 \
    --baseline-tag bA --candidate-tag splitM2 \
    --out-dir results/board_loop/ffn_split
```

Both arms are profiled in one session with the same curated kernels and filed under their
own model identity (`<net>_<tag>`). That is not cosmetic: `gen/mb/profile` is a symlink
the profiler writes through, so a shared basename means the second run overwrites the
costs the first was solved from and the comparison becomes a schedule against itself.

The round refuses to use costs from a board run that did not report `max_abs_err=0`, and
warns when a curated kernel fell back to the scalar reference — both produce timings that
describe something other than the kernel under test.

Result on `ffn_block`, whose `linear_s8` overruns its slot (the scheduler's own advice):

| | blocking dispatch | total |
|---|---|---|
| baseline | 360,803 cy | 685,262 cy |
| split along M, ×2 | 185,611 cy | 659,129 cy |

and the re-solve: makespan 53.55 → 45.37 ms, worst lateness 22.65 → 14.54 ms, 2 → 1
instance-misses. **Verdict: ACCEPT** on term 2 of 9.

## 4. The figure, and its caption

```bash
XPURT_CPSAT_WORKERS=6 $PY scripts/gen_schedule_evolution.py --from-loop
$PY scripts/emit_figure_numbers.py \
    --metrics results/codesign_feedback/schedule_evolution_auto_metrics.json \
    --prefix schedEvo --out results/codesign_feedback/schedule_evolution_numbers.tex
```

`--from-loop` runs the loop and builds the four panels from its **actual trajectory**,
so panel 2's title names the levers the loop chose. `emit_figure_numbers.py` turns the
figure's own metrics into `\newcommand`s for the caption to quote, because a hand-typed
caption and a regenerated figure drifted apart once already: the paper said panel 3 had
four misses at 34 ms while the auto variant of the same figure read 2 misses at 36 ms.
Both were true about *some* figure.

## 5. How general is it

```bash
$PY scripts/loop_over_workloads.py --solver greedy \
    --board-calibration results/codesign_feedback/k1_board_calibration.json \
    --out-dir results/loop_sweep
```

Runs the same driver over every multi-net K1 spec with no per-workload tuning and reports
the fraction it strictly improves, listing the workloads where no lever helped. A run
that converges with nothing applied is not an improvement and is counted as one of the
failures.

---

## 6. Closing the loop the rest of the way: board costs inside the AOT search

Until now the outer loop could only ever **re-solve a decision the inner loop had already
committed to**. `--board-calibration` re-costs and re-schedules a fixed spec; the lever
and rewrite search that chose that spec always ran on the isolated profile database. So
the board could fix a placement, never a transformation.

`--search-calibration` runs the inner search itself against measured board costs:

```bash
$PY scripts/run_codesign_loop.py --workload data/toplevel/scaling/w5_ffn_dronet_yolo.json \
    --solver greedy --objective lateness --max-rounds 3 \
    --search-calibration results/codesign_feedback/k1_board_calibration.json \
    --out-dir results/codesign_feedback/search_board
```

Every lever, every graph rewrite and the baseline they are compared against are scored on
board costs. `loop_report.json` records `search_costs` so a board-honest run cannot later
be mistaken for an isolated-profile one -- the two are not comparable, and the driver says
so in its first line of output.

### It changes which transformations the loop adopts

Same workloads, same solver (greedy), same budget; only the costs the search sees differ:

| rung | AOT-cost search | board-cost search |
|---|---|---|
| `w2_ffn_tight` | `shard`, 5 -> **0** misses, lateness 83.05 -> 0 | `shard`, 5 -> **0** misses, lateness 99.74 -> 0 |
| `w3_ffn_dronet` | `shard`, 10 -> 5, lateness 88.06 -> **7.79** | `shard`, 10 -> 5, lateness 113.15 -> **35.64** |
| `w5_ffn_dronet_yolo` | `ime` **+ `shard`**, 11 -> **5** | `ime` only (**shard rejected**), 11 -> **11** |

Three different things to read off it:

1. **On w2 the decision is robust.** Sharding clears every deadline whichever cost model
   is used, so the AOT loop was right for the right reason.
2. **On w3 the lever is right and the confidence is not.** Both searches pick `shard` and
   both land on 5 misses, but the AOT view reports 7.79 ms of residual lateness where the
   board reports 35.64 ms -- **4.6x** more. A loop reading only the AOT number would
   report itself nearly finished.
3. **On w5 the decision flips.** The AOT search credits `shard` with halving the misses
   (11 -> 5) and deploys it. Under board costs `shard` is *rejected twice*, because it
   makes the deciding term worse: worst deadline lateness 38.62 -> 40.38 ms and misses
   11 -> 12. That is a transformation the inner loop would ship and the silicon would
   punish, and only a search that can see board costs declines it.

**The honesty limit on point 3.** `w5` is the rung containing `yolov8_nano_64x96`, and
the calibration in the repo has **no yolo in its measurement set** -- yolo dispatches are
costed by the op-kind fallback, i.e. extrapolated, and yolo dominates this rung. So the
*direction* of the flip is credible (the board inflates exactly the ops `shard` widens)
but its magnitude is not measured for this workload. Fixing that needs yolo board runs
and then a calibration built from them, which is now a command rather than a lost script:

```bash
$PY scripts/emit_board_calibration.py \
    --trace-glob 'results/<your run>/*_trace.csv' \
    --workload 'w5 ladder rung' --out results/codesign_feedback/k1_cal_w5.json
```

### Where the calibration comes from, and what could not be recovered

`results/codesign_feedback/k1_board_calibration.json` was an **orphaned output**: the
script that produced it was never committed, so the table could be consumed and never
regenerated -- which is the whole reason its `fallback_key` has to admit "EXTRAPOLATED
for nets not in calibration set, e.g. yolo".

`scripts/emit_board_calibration.py` rebuilds it from the traces and is checked against
the committed table rather than merely resembling it:

```bash
$PY scripts/emit_board_calibration.py \
    --trace-glob 'results/k1_feedback_exact/board_runs*/original_*_trace.csv' \
    --trace-glob 'results/k1_feedback_exact/board_runs*/feedback_*_trace.csv' \
    --validate-against results/codesign_feedback/k1_board_calibration.json \
    --out /tmp/cal.json
```

which reports **the same 48 per-dispatch keys, 40 of them within 2%**, and 7 of 11 op
keys within 2%. The recipe it recovered is: `actual/predicted` per dispatch, where
`actual` is `(end-start)` rdtime ticks at `k1_trace.K1_RDTIME_HZ` and `predicted` is the
`predicted_duration_ms` the runner stamps into the trace; arithmetic **mean** of
per-sample ratios; all 60 traces (both schedules, all three run directories); a 0.1 ms
floor on the pooled op tier so timer granularity cannot inflate it. Every alternative
tried is markedly worse -- RT traces only gives 25/48, dropping the cold first instance
13/48, trimming each key's top decile 9/48 -- which is the evidence that this is the
original recipe and not a coincidence.

Two things did not come back. Eight `dronet` keys sit 2-14% high, and the v1
`aggregate_multiplier` of 1.2608 matches no single statistic over these traces (mean
1.3348 under the floor, median 1.1647, mean unfloored 1.6858), so the original applied a
trim that went with the script. Both are recorded in the emitted table's `validation`
block instead of being smoothed over.

It needs no schedule, no IR and no join -- `op` and `predicted_duration_ms` are columns
the runner already writes -- so it runs on any workload's board traces, which is the
point. Queue delay is excluded: a dispatch that waited is not a dispatch that ran slowly,
and charging the wait to the op would make the scheduler pay twice for its own placement.

---

## 7. The whole loop, with the board in it — **needs the K1**

Sections 5 and 6 still start from a calibration someone measured earlier. This runs the
arc end to end, so the costs the second AOT search sees were measured from *the schedule
the first AOT search produced*:

```bash
$PY scripts/run_closed_board_loop.py \
    --workload data/toplevel/scaling/w3_ffn_dronet.json \
    --repeats 10 --solver greedy \
    --out-dir results/codesign_feedback/closed_w3_n10
```

Four stages, no hand-editing: AOT search on isolated profiles -> execute the converged
schedule on the K1 N times -> fit a calibration to those traces -> AOT search again
scored on it -> report which transformations the measured costs added or dropped.

Stage 3 is what makes stage 4 honest. The calibration covers exactly the networks that
just ran, and the driver checks the spec's networks against `coverage.nets_exact` and
names any that are still extrapolated rather than leaving it to a footnote. On this
workload it prints `every network in this workload is MEASURED — nothing extrapolated`.

### What it found (w3, 10 board runs, 214 dispatches each, SCHED_FIFO 80)

| | AOT-cost search | measured-cost search |
|---|---|---|
| transformations | `shard` | `shard` |
| instance misses | 10 -> 5 | 10 -> 5 |
| residual lateness | **7.79 ms** | **16.00 ms** (2.05x) |

So the decision is robust on this rung and the *confidence* is not: the isolated profile
database tells the loop it has 7.79 ms of lateness left where the silicon says 16.00 ms.

### The measurement is repeatable; the calibration is not transferable

Two comparisons, and the first is what licenses the second:

| comparison | shared keys within 10% | median ratio | worst key |
|---|---|---|---|
| this workload, 3 runs vs 10 runs | **31 / 33** | 0.994 | 20% |
| this workload (10 runs) vs the 60-run generic table | **16 / 33** | 0.955 | **119%** |

Three board runs already reproduce ten (`aggregate_multiplier` 1.196 vs 1.2032), so the
measurement is stable and the disagreement with the generic table is not sampling noise.
And the generic table is not a stranger to these networks -- `dronet`, `ffn_block` and
`mlp_control` are all *in* its measurement set; the 15 keys it has that this workload does
not are `fused_full`, which w3 does not run.

The reason is that a `network/dispatch_id` multiplier is **not a property of the
network**. It depends on the core width the schedule chose for that dispatch and on what
was co-resident while it ran. w3's converged schedule shards `ffn_block`; the schedule the
generic table was fit to did not shard it the same way. Hence `dronet/18` at 2.51x here
against 1.15x there, and `mlp_control/6` at 1.06x against 1.58x.

**What follows for the outer loop.** Re-solving against a calibration measured from a
*different* schedule is a systematically wrong cost model, not a slightly stale one. The
aggregate transfers (1.20 vs 1.26, median key ratio 0.96) and the per-dispatch tier does
not, which is exactly the tier the solver leans on. Calibrate the schedule you are about
to run.

### And the same arc on the 5-net rung, where the loop does not win

`w5_ffn_dronet_yolo` (five networks, `yolov8_nano_64x96` included, 5 board runs):

| | AOT-cost search | measured-cost search |
|---|---|---|
| transformations | `ime` (`shard` refused: not buildable) | `ime` |
| instance misses | 11 -> 11 | 11 -> 11 |
| baseline lateness | 109.81 ms | **164.75 ms** |
| residual lateness | 99.02 ms | **140.86 ms** (1.42x) |

The ratio is stable across disjoint samples of the same five board runs -- residual
lateness 141.32 / 140.48 / 140.86 ms from runs 0-2, runs 2-4 and all five, agreeing
within 0.6% -- so 1.42x is the number and not one draw of it.

This is the honest negative, and it is worth as much as w2's clean win. Nothing clears a
deadline on this rung. What the board adds is *why*: the isolated profile database
understates the baseline by 1.50x and the residual by 1.42x, so an AOT-only loop would
report itself much closer to feasible than it is.

**A 2%-of-time provenance error moved this by 20%, which is worth knowing before quoting
any lateness ratio.** A first pass measured 175.12 ms (1.77x) because the board ran the
STOCK COCO-80 detector while the schedule was costed from the deployed 2-class fine-tune
(see the contract's `staged_ir_networks`: a default extraction produces a different
network under the same name, with the same 98 dispatches and no error). Only 10 of those
dispatches differ, 2.02% of measured yolo time, and their per-op multipliers differ by
1-6% -- yet total lateness moved 20%, because the dispatches that differ are the detect
head at the END of yolo's chain, so extra time there pushes whole instances past their
deadlines. Total lateness is a critical-path quantity, not a sum, and it does not
degrade gracefully with small cost errors.

**yolo is costed by its op kinds here, not per dispatch, and that is deliberate.** The
runner records zero-cost ops (`chunk`, `split`, `slice`) with `dispatch_id = -1` and
numbers the remaining dispatches from zero, while the schedule numbers all of them. So
`yolov8_nano_64x96` traces 90 dispatches as 0..89 where the schedule spans 0..97, and a
per-dispatch key would be applied to a *different* dispatch. Nothing about that is visible
in the emitted table -- every ratio is computed inside one trace row and is correct, so
the multipliers look plausible and the file validates -- which is why
`emit_board_calibration.py` takes `--schedule`, checks the two numberings, and **drops a
misaligned network's per-dispatch keys** rather than keeping wrong ones that look right.
It cost a first pass at these numbers: with the offset keys this rung read 214.01 /
186.79 ms (1.89x). The four networks with no zero-cost ops are aligned and keep their
48 exact keys; w3's three are aligned too, so nothing there moves.

**The calibration is what makes this rung sayable at all.** `conv2d_batchnorm2d_silu_s8`
is 94.2% of yolo's runtime and is **absent from the generic table's op tier**, so every
earlier "outer loop" number here costed yolo's dominant op by the aggregate multiplier,
1.2608. Measured on the board it is **1.4060**, and yolo's per-dispatch multipliers run
from 0.99 to 12.31 (median 1.4143) — a distribution that one scalar was standing in for.
For a cross-check in the other direction, `linear_f16` comes back at 2.1544 here against
the generic table's 2.0496, on a network that table did measure.

### Two things it has to get right, and how you can tell it did

* `entries_done`. A run whose `core_kind` does not match the backend tag completes
  normally having executed **nothing** -- every worker refuses every entry, and the trace
  is all zeros. The driver copies each run's stdout beside its trace, sums the per-worker
  counts, and refuses to calibrate from a run reporting `entries_done=0`. On the w3 runs
  the counts sum to 214, the schedule's full entry count.
* `SCHED_FIFO 80`. The same schedule measures **1.35x** under CFS and **0.92x** under
  FIFO. That difference is preemption, not execution; calibrating on it would teach the
  solver that its own kernels are slow when what actually happened is that Linux
  descheduled them. Runs are RT-pinned, and `xpurt: observed_sched_policy` in the copied
  stdout is where you check it rather than trusting the request.

### If stage 2 cannot start

It needs two things this repo does not carry by default:

* **The GCC 14.3 cross toolchain**, fetched automatically. 13.2 -- what `CROSS` defaults
  to via chipyard -- miscompiles the RVV intrinsics into a `SIGILL` with no stdout at all.
* **An interpreter with torch** (and `ultralytics` for the detector), because stage 2 runs
  ModelBlaster's `extract_graph` and this repo's venv is installed `--no-deps` on
  purpose. The driver probes for one that has torch **and** resolves `modelblaster`
  inside this checkout -- both, because there is a second ModelBlaster clone on this
  machine that some environments import under the same module names -- and prints what it
  tried when none does. Override with `--board-py` or `MB_PY`.

---

## 8. The codegen contract: what the scheduler is allowed to recommend

The scheduler's option space was wider than the compiler's, and nothing connected the
two. `shard` machine-combination mode lets every periodic **instance** of a dispatch pick
its own aligned core block. For a convolution that cannot be generated: the packed weight
array is materialised per shard while emitting the skeleton, so the width has to be one
value per dispatch. The solver did not know, so it produced schedules that were valid for
the runtime and impossible for the compiler.

That is not a theoretical concern. It cost a board build, and it retracted a result.

**`ModelBlaster/cores/codegen_contract.json`** states the compiler's constraints as data,
each with its evidence and where it is enforced. It is the sibling of
`cores/spacemit_k1.json`, and the two are not interchangeable: that file says what the
**hardware can execute** (which ops a core runs, and that `smt.vmadot` traps outside
cluster 0), this one says what the **compiler can generate**.

| rule | severity | statement |
|---|---|---|
| `uniform_width_across_instances` | refuse | packed-weight (conv family) dispatches take one width across all instances |
| `width_divides_output_channels` | refuse | the width must divide OC exactly |
| `runtime_sliceable_ops` | none | `linear_s8`, `matmul_s8` may vary width freely — row-major weights are sliced at runtime |
| `linear_split_axis` | refuse | a linear splits on M; an N-split is refused when M > 1 |
| `ime` | refuse | int8 only, cluster 0 only — `smt.vmadot` does not degrade off-cluster, it dies |
| `machine_combinations` | refuse | one core kind per combination, as an aligned block |
| `staged_ir_networks` | refuse | `yolov8_nano_64x96` cannot be extracted by name |

`ModelBlaster/pipeline/schedule_shards.py` takes its packed-weight op list **from the
contract** rather than keeping a second copy — two lists in two languages would be free
to drift, and the drift fails in the worst direction: the scheduler believes an op is
unconstrained and the build refuses the result. A test asserts they agree.

### Two ways to use it, and the first is better

```bash
# check a solved schedule -- milliseconds, names the dispatch
$PY xpu-rt/codegen_contract.py schedules/scheduled_*.json
```

**Reject afterwards.** `run_codesign_loop.py` gates every candidate, lever and graph
rewrite alike, because *a lever that wins on paper and cannot be compiled is not a win*.
This is what retracted the w5 result: `shard` was credited with 11 → 5 instance misses on
a schedule where `dronet` dispatch 0 takes width 2 in one instance and 4 in another.
Gated, w5 converges on `ime` alone at 11 → 11. The previously reported halving on that
rung was never deployable. (Reassuringly, scored on measured board costs the nine-term
rule rejects the same lever independently, on worst deadline lateness 38.62 → 40.38 ms —
so the gate is not merely pessimistic.)

**Constrain beforehand**, which is better, because rejecting the solver's answer
understates what co-design can do:

```bash
XPURT_UNIFORM_PACKED_WIDTH=1 $PY scripts/run_xpurt_schedule.py --networks-json <spec> \
    --profiled --solver milp --scheduler cpsat
# [cpsat] codegen contract: 10 packed-weight dispatch(es) constrained to one width
#         across their instances
```

CP-SAT gets a per-(dispatch, width) indicator linked to its combination-presence
variables, so the solver **still chooses** the width — it just has to choose one. Pinning
a width instead would trade a correctness constraint for a policy decision, and the whole
point of shard mode is that the right width depends on what else is running. Off by
default so no existing result moves.

**What constraining costs.** Same spec (`w5_ffn_dronet_yolo`, shard mode), same 300 s
CP-SAT budget, 4 workers, one env var apart:

| | makespan | dispatch misses | contract |
|---|---|---|---|
| unconstrained | 93.81 ms | 210 | **refused** — 6 conv dispatches take 2–3 widths each |
| constrained (16 dispatches coupled) | 93.21 ms | 200 | **passes** |

Nothing measurable. The constrained solve came out marginally ahead, which at 4 workers
under a time limit is inside CP-SAT's own nondeterminism — the honest reading is that the
constraint is free here, not that it helps. What matters is the last column: the
unconstrained schedule is not a result at all, and rejecting it costs the whole candidate,
which on w5 is the difference between reporting `shard` and reporting nothing.

The op kind is not available from a dispatch graph — that carries only ids and
dependencies, so `operation_name` is `<net-instance>_dispatch_<id>` and names no op at
all. `run_xpurt_schedule._annotate_op_kinds` attaches it from the profile database, and
looks the base network up **by membership, never by trimming trailing digits**:
`yolov8_nano_64x96` ends in a digit, so a blind strip yields `yolov8_nano_64x` and every
one of its dispatches silently loses its op kind. That hazard bit three separate places
in one afternoon — this driver, the calibration emitter, and the board runner's ingest.

### Adding a rule

Edit the json. Both sides read it, `tests/test_codegen_contract.py` checks the two agree,
and a rule whose `checkable_from` is `"schedule"` is enforced automatically by
`violations()`. A missing contract degrades to **ungated and says so** — refusing every
candidate because a submodule is absent would be worse than not checking — but an empty
or unreadable one is an error, because a silently empty ruleset passes everything and the
build still fails.

---

## Environment

```bash
uv venv .venv --python 3.13
uv pip install --python .venv/bin/python --no-deps -e . -e ModelBlaster
uv pip install --python .venv/bin/python numpy scipy matplotlib pandas ortools \
    pyyaml pillow requests pytest cvxpy 'highspy==1.14.0'
git submodule update --init ModelBlaster
```

`--no-deps` for the two local packages on purpose: ModelBlaster declares `torch` and
`ultralytics`, which only the model-extraction path needs. `highspy` is pinned because
1.15's bundled HiGHS has a C++ ABI clash with ortools' `libortools` — import `cvxpy`
then `ortools` and it dies on an undefined symbol.

Then `$PY -m pytest -q` (771 passed) and, in `ModelBlaster`, `$PY -m pytest tests
pipeline/tests -q` (183 passed). CI runs both plus `examples/run_all.py`.

MOSEK is optional and not needed for any result here.

## Things that will bite you

- **`XPURT_CPSAT_WORKERS`.** `scheduler_cpsat` pins one worker and seed 42 so a cold
  rerun matches bit-exactly, and says in as many words that more workers under a time
  limit are not deterministic. The published solver numbers use `=0`/`=6`; the test
  suite pins `=1` via `conftest.py`. Record which you used.
- **A short profile CSV.** A dispatch absent from a profile that *exists* is costed
  0.0. It is reported now, listed in the schedule as `metadata.zero_costed_dispatches`,
  and fatal under `XPURT_STRICT_ZERO_COST=1`. On the sensor workload it flags 8 — yolo's
  `chunk2_c1` ops, which the codegen legitimately drops. A short CSV looks identical
  from inside the loader, which is why the list is in the artifact.
- **`--board-calibration` with a wrong path** exits 2 rather than silently producing an
  uncalibrated schedule under a calibrated name.
- **`pdb_hash` vs `solve_hash`.** `pdb_hash` fingerprints only the profile CSVs, so an
  additive solve and a calibrated re-solve of one spec share it — and `compare_candidates`
  refused that comparison, which is the one the board-feedback story rests on.
  `solve_hash` folds in the calibration content, solver and env.
- **Derived IME profiles.** `scripts/make_ime_profile.py` writes measured-RVV-divided-by-
  measured-speedup into the tree the loader reads as board measurement. Those rows carry
  `source=ime_derived(...)` and a `PROVENANCE.json` sidecar. They are predictions.
- **The board is shared.** Check before you take it (`docs/K1/k1_board.md`). `ping` to the
  old address fails while `ssh k1` works, so test with ssh.
