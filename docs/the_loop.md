# The loop, end to end

How a measurement becomes a compiler change, and how the compiler change gets
accepted or rejected. Every number here was measured on the physical SpaceMiT
K1; every command is one that has actually been run.

The one-line summary: **the loop searches on predictions and decides on
measurements, and those are different jobs done by different tools.** Most of
the failures this project has had came from letting one do the other's work.

---

## 1. The cycle

```
   ModelBlaster IR ──generate_kernels──▶ build ──profile_writer──▶ results.csv
         ▲                                                             │
         │                                                    profile_loader
   apply_*_hint                                                        │
         │                                                             ▼
    *_hints/v1 ◀──advice_to_*_hint── compile_advice.json ◀──  XPU-RT schedule
         ▲                                   ▲                         │
         │                          emit_compile_advice          harness_xpurt
         └────────── diff_dispatch_graph (gate) ◀─────────────── measured trace
```

Eight artefacts, each a file with a contract. Any stage can be run and debugged
alone, which is the reason for the file boundaries.

| stage | tool | artefact |
|---|---|---|
| IR | `extract_graph.py` | `build/k1*/<model>/int8/graph.json` |
| kernels | `generate_kernels.py` | `kernel_picks.json`, `kernels.c` |
| profile | `run_model_k1.sh` → `profile_writer` | `gen_mb/profile/<impl>/…/results.csv` |
| schedule | `run_xpurt_schedule.py` | `schedules/scheduled_*.json` |
| run | `run_xpurt_k1.sh` → `harness_xpurt` | `*_trace.csv` |
| advice | `emit_compile_advice.py` | `compile_advice.json` |
| hint | `advice_to_{fusion,split,unfuse}_hint.py`, `advice_to_kernel_choice.py` | `modelblaster.*_hints/v1` |
| rewrite | `apply_{fusion,split,unfuse}_hint.py` | `graph.<rewritten>.json` + `id_remap` |
| gate | `diff_dispatch_graph.py` | exit 0 / 3 / 4 |
| verdict | `compare_candidates.py` | accept / reject + the deciding term |

## 2. The four verbs

| verb | producer | bridge | consumer |
|---|---|---|---|
| `fuse_with_successor` | `overhead_advice` | `advice_to_fusion_hint.py` | `apply_fusion_hint.py` |
| `split` | `blocking_advice` | `advice_to_split_hint.py` | `apply_split_hint.py` |
| `unfuse` | `unfuse_advice` | `advice_to_unfuse_hint.py` | `apply_unfuse_hint.py` |
| `choose_implementation` | `implementation_advice` | `advice_to_kernel_choice.py` | `generate_kernels --keep-reference-ops` |
| `shard` | `shard_advice` | — | `MB_SHARD_FACTOR` (build-level) |

`shard` has no bridge because it is not a graph rewrite: the width is chosen by
the SCHEDULER, per dispatch, out of multi-core profiles, rather than by a hint
that changes the IR. See section 4c.

This entry used to read "deliberately unwired: it needs multi-core profiles
that do not exist, and B4 measured a 2.27x ceiling." Both halves are now
obsolete and are recorded here rather than deleted, because the second one was
quoted for a while as if it bounded sharding in general. The profiles exist for
three models at four widths each; the 2.27x came from a different mechanism;
and OC sharding measures **3.87x and 3.93x on four harts** for ffn_block's two
linears. What was actually missing was not profiles but a board harness -- the
Linux harness compiled no pool at all, so every `parallel_<op>` wrapper in
every K1 binary had taken its serial arm.

## 3. Search and decide are different

There are two loops, and conflating them is the mistake to avoid.

**The search loop** is cheap and predicted. `granularity_loop.py` builds the
workload in-process, generates fuse/split candidates with `rewrite.py`,
re-schedules each, and ranks them. `decision_loop.py` drives it over rounds.

Its ranking criterion is not uniform, and that is deliberate: **the predicted
cost model has no per-dispatch launch overhead**, so fusing tiny dispatches
leaves the predicted makespan unchanged — every fuse candidate scores
Δmakespan 0.00. Merges are therefore judged by the dispatches and cross-device
transitions they remove; splits by makespan delta, which prediction *can* see
because parallelism is visible in it.

**The decision loop** is expensive and measured: rewrite → rebuild → reprofile
on the board → re-solve → `candidate_objective.accept()`.

The acceptance rule is lexicographic over nine terms:

1. hard deadline misses 2. max lateness 3. frequency compliance
4. p99 response of critical tasks 5. heavy-model max latency
6. heavy-model throughput 7. makespan 8. utilisation
9. **standalone kernel cycles**

Term 9 is last on purpose. `candidate_objective.py`'s own worked examples:

> a split making a kernel 5% slower in total cycles but letting DroNet meet
> 30 Hz instead of missing 20% of deadlines is a **WIN**;
> a fusion making a model 10% faster in isolation but creating an 8 ms
> non-preemptible dispatch that breaks a 100 Hz MLP is a **LOSS**.

Neither is visible without scheduling the rewritten graph. A rung that stops at
"reprofiled on the board" has not been adjudicated at all.

## 4. A worked round trip, with the real numbers

DroNet's dispatch 0 (`conv2d_s8`, OC=32) split along OC, in a 3-model workload
(mlp_control 100 Hz, dronet 30 Hz, yolov8_nano 4 Hz) on 8 harts.

```bash
# 1. board-profile the rewrite WITHOUT re-extracting.
#    MB_IR is the whole point: copying the rewrite over graph.json and letting
#    step 1/5 re-extract profiles the BASELINE and files it under the
#    rewrite's name.
#    gen_mb/profile is a SYMLINK and the profiler writes in place -- back up
#    the baseline results.csv first or you destroy what everything else was
#    solved from.
eval "$(scripts/setup_spacemit_toolchain.sh)"   # sets CROSS; see below
MB_IR=artifacts/k1_run/round_B3_dronet_split/graph.split_x4.json \
PROFILE_OUT_ROOT=$PWD/gen_mb/profile \
  bash ModelBlaster/scripts/run_model_k1.sh dronet int8 rvv_x60 0
#    GATE: stdout must say max_abs_err=0. A rewrite that changes the answer is
#    ineligible and its timings mean nothing.

# 2. file it under its own BASENAME, not its own model name, so DroNet stays
#    in critical_models and the per-model terms remain name-comparable:
#    gen_mb/profile/rvv_x60/spacemit_x60/dronet/dronet.split_x4.int8/…

# 3. re-emit the dispatch graph. emit_dispatch_graph takes its output path from
#    ir["name"] and ir["quant"], so give the copy a distinct quant or it
#    OVERWRITES the baseline graph.

# 4. solve, with --max-periodic-iters 1 on BOTH sides: the refinement loop can
#    grow num_instances, and two candidates that scored different amounts of
#    work are not comparable.
python scripts/run_xpurt_schedule.py --networks-json <spec> \
    --solver greedy --profiled --max-periodic-iters 1

# 5. the verdict
python scripts/compare_candidates.py \
  --baseline-schedule  schedules/scheduled_..._greedy_profiled.json \
  --candidate-schedule schedules/scheduled_..._split_x4_greedy_profiled.json \
  --windows-from data/toplevel/networks_k1_mb_3model_4hz.json \
  --critical-models mlp_control,dronet --heavy-model yolov8_nano
```

Measured:

| rung | dispatches | service | vs baseline | board verify |
|---|---|---|---|---|
| baseline | 21 | 8.5900 ms | — | — |
| split ×2 | 22 | 9.7691 ms | +13.7% | `max_abs_err=0` |
| split ×4 | 24 | 12.4028 ms | +44.4% | `max_abs_err=0` |

Scored:

```
              misses  worst_late    p99     makespan   standalone
  baseline      0       0.000      8.230    902.652     996014
  split_x2      0       0.000      7.890    902.652    1007806
  split_x4      0       0.000      8.310    902.652    1034142

  x2  REJECT — heavy-model max latency: 152.65 beats 157.27
  x4  REJECT — heavy-model max latency: 152.65 beats 158.57
```

**Both rejected at term 5, not term 9.** And x2's shape is the opposite of the
story that had been recorded: it *improved* critical-task p99 (7.890 vs 8.230)
and lost only because it pushed yolov8_nano's max latency out by 4.6 ms.
"+13.7% slower, rejected" was a true statement about the term that ranks last
and was never the reason.

The sweep shows what one rung could not: x4 is worse on the deciding term *and*
gives back x2's p99 gain entirely, so granularity has an interior optimum below
x4 — and it is still not better than not splitting. Consistent with B4's +76%:
slicing OC adds work, and where nothing blocks, nothing buys it.

### The rung that was accepted

Three rejections do not demonstrate a loop; they demonstrate a filter. The
`unfuse` rung is the one where the loop changed the schedule for the better and
the objective said so.

`unfuse_advice` fires on a fused op whose implementation fell back to the
scalar reference — the historical 0.81× condition. That condition no longer
occurs on any profile in the tree, because the curated fused kernel now exists,
so the rung was **reconstructed**: `--keep-reference-ops
conv2d_batchnorm2d_silu_s8` forces the fallback, and the loop responds to it.
State that plainly — the loop's response is hardware-proven; the loop did not
spontaneously find the condition.

It then found a second thing, which was not reconstructed. Against the *curated
fused* control (`ctrl`, same toolchain, same graph shape), the unfused graph is
still faster:

```
              misses  worst_late    p99     makespan   standalone   heavy max
  ctrl          0       0.000      8.230    898.097     961067       148.10
  unfused       0       0.000      8.230    866.421     794037       116.42

  ACCEPT — heavy-model max latency: 116.42 beats 148.10
```

218.128 → 176.370 ms on the detector, **−19.1%**, and better on makespan and on
standalone cycles as well. Accepted at term 5.

**This is a kernel bug, not a fact about fusion.** Splitting conv+BN+SiLU into
three dispatches cannot be intrinsically cheaper than doing it in one pass. The
reason it wins is that `rvv_conv2d_batchnorm2d_silu_s8_rvv_oc_blocked_bn_silu_epilogue.c`
carries a slower conv inner loop than the standalone
`rvv_conv2d_s8_rvv_vsmul_vnclip.c`, so the fused kernel gives back more than the
fusion saves. The right response is to rebuild the fused kernel on the faster
inner loop, which should beat both. Recorded as the loop working — it found a
real 19% and named the term — not as a recommendation to stop fusing.

### The figure

```bash
S=schedules/scheduled_networks_k1_mb_3model_4hz
python scripts/plot_loop_iterations.py \
  --iteration "1 · baseline: dronet conv0 OC=32=${S}_greedy_profiled.json" \
  --iteration "2 · split dronet conv0 x2=${S}_split_x2_greedy_profiled.json" \
  --iteration "3 · split dronet conv0 x4=${S}_split_x4_greedy_profiled.json" \
  --iteration "4 · yolo control rebuild, same toolchain=${S}_yolo_ctrl_greedy_profiled.json" \
  --iteration "5 · pin maxpool2d to the scalar kernel=${S}_yolo_pinmaxpool_greedy_profiled.json" \
  --iteration "6 · unfuse yolo conv+BN+SiLU=${S}_yolo_unfused_greedy_profiled.json" \
  --control 4 --judge-against 5=4 --judge-against 6=4 \
  --windows-from data/toplevel/networks_k1_mb_3model_4hz.json \
  --critical-models mlp_control,dronet --heavy-model yolov8_nano \
  --window-ms 250 --out-dir out/figures --stem k1_loop_story \
  --title "Four rewrites adjudicated on the K1 by the nine-term objective"
```

→ `out/figures/k1_loop_story.png`, all four adjudicated rewrites in one
figure, plus the per-rung PNGs. Panel d is the control; panels e and f are
judged against it rather than against the shipping baseline.

→ `out/figures/k1_loop_evolution.png`, one panel per rung on the real machine
lanes, each labelled with its dispatch count, miss count, p99, makespan and the
term its verdict turned on. `diff_dispatch_graph` proves the *graph* changed;
this shows whether the *schedule* did, which is a different question and the one
that decides.

## 4b. The other axis: which UNIT runs a dispatch

Everything above rewrites the graph and re-schedules it. The second axis is
leaving the graph alone and changing the implementation — the K1 has an int8
MAC unit (`smt.vmadot`, IME) on cluster 0, and a dispatch can run there instead
of on the vector unit.

It is a scheduling decision rather than a build flag because **the accelerator
does not always win**, and where it stops winning is measurable:

| M (K=N=256) | RVV | IME | |
|---|---|---|---|
| 4 | 0.188 ms | 0.387 ms | RVV |
| 8 | 0.375 ms | 0.440 ms | RVV |
| 16 | 0.751 ms | 0.581 ms | **IME** |
| 64 | 3.007 ms | 1.379 ms | **IME** |
| 128 | 6.012 ms | 2.534 ms | **IME** |

Both are linear in M — `RVV = 0.047·M` and `IME = 0.0173·M + 0.297` — and the
fits say why. RVV needs no repacking, so it is pure per-row work with no fixed
cost. IME does 2.7× less work per row and pays a fixed ~0.30 ms packing its
operands into the 4×8 tiles the MAC unit requires. **They cross at M = 10.1.**

So attention (M=8) stays on the vector unit and a transformer MLP (M in the
hundreds) moves to the MAC unit, and only a scheduler holding both measured
costs can tell them apart.

```bash
# both sides, same graph, same dispatch ids
scripts/run_model_k1.sh ffn_block int8 rvv_x60 0
scripts/run_model_k1.sh ffn_block int8 ime_x60 0

python scripts/run_xpurt_schedule.py --networks-json \
    data/toplevel/networks_k1_ffn_ime.json --solver greedy --profiled \
    --max-periodic-iters 1        # scheduler.enable_impls: true

python scripts/plot_loop_iterations.py --color-by impl \
  --iteration "1 · all-RVV=schedules/scheduled_networks_k1_ffn_rvv_greedy_profiled.json" \
  --iteration "2 · impl-aware=schedules/scheduled_networks_k1_ffn_ime_greedy_profiled.json" \
  --windows-from data/toplevel/networks_k1_ffn_ime.json \
  --critical-models mlp_control --heavy-model ffn_block \
  --window-ms 100 --out-dir out/figures --stem k1_hetero_placement
```

→ `out/figures/k1_hetero_placement.png`. The FFN block goes from 43.4 ms
entirely on the vector unit to 30.6 ms with its two linears on the MAC unit,
while `mlp_control`'s 10 ms ticks are undisturbed. **ACCEPT on heavy-model max
latency.** `--color-by impl` colours by which unit ran a dispatch rather than
by network, because that is the question this figure answers.

**An `ime` COMBINATION is not the same as MAC-unit work.** With `enable_impls`
on, a core appears in several combinations, one per implementation, and a
combination is costed from its backend's profile whatever the op is. A
layernorm scheduled on an ime combination fell through to the identical RVV
kernel; only `linear_s8` and `matmul_s8` there are genuine accelerator work.
The figure distinguishes them, and conflating them would overstate how much of
the schedule the NPU carries.

## 4c. The third axis: how many HARTS run a dispatch

Section 4 changes the graph, 4b changes which unit runs a dispatch. The third
lever leaves both alone and changes how WIDE a dispatch runs -- one hart, or a
block of them sharing the work.

It is a scheduling decision for the same reason as 4b: **the extra harts do
not always pay**, and where they stop paying is measured, not derived. On
`rvv_x60`, medians of 6 warm reps:

| | 1 hart | 2 | 4 | 8 | 4-hart |
|---|---|---|---|---|---|
| ffn_block fc1 (M=128) | 338091 | 168644 | 87456 | 53162 | 3.87x |
| ffn_block fc2 (M=128) | 266293 | 128800 | 67815 | 54261 | 3.93x |
| ffn_block total | 681737 | 375542 | 233113 | 185161 | 2.92x |
| `layernorm_s8` | 61818 | 62125 | 61902 | 62007 | **1.01x** |

The last row is the control and it is why the others can be believed:
`layernorm_s8` has no pool path, so it must not move, and it does not -- within
0.5% across every width. A speedup that came from a quieter board would have
moved it too.

Whole models, four harts: **ffn_block 2.92x, yolov8_nano_64x96 1.96x, dronet
1.59x**. The headline number is the least useful thing in the data, because
within ONE model the per-dispatch gain runs from 4.02x (a wide-OC conv) down to
0.83x (a 1x1). Sharding pays in proportion to OC and against spatial extent, so
a model's total is set by whichever its largest dispatches happen to be -- not
by its size or its maximum OC. That spread, not a count of regressions, is the
argument for choosing a width per dispatch.

```bash
# profile one model at four widths. MB_CORES derives the pool size, the
# affinity mask and the topo tag from one place -- a run tagged topo_0_1_2_3
# whose pool was actually 1 thread is a serial measurement filed as a parallel
# one, and nothing downstream could tell.
for spec in "0:1" "0,1:2" "0,1,2,3:4" "0,1,2,3,4,5,6,7:8"; do
  MB_CORES="${spec%:*}" MB_SHARD_FACTOR="${spec#*:}" ITERS=7 \
    scripts/run_model_k1.sh dronet int8 rvv_x60 0
done

python scripts/plot_multicore_scaling.py --model dronet --model yolov8_nano_64x96
python scripts/plot_loop_iterations.py --color-by width \
  --iteration "the solver picks a width per dispatch=schedules/scheduled_networks_k1_multicore_shard_greedy_profiled.json" \
  --windows-from data/toplevel/networks_k1_multicore_shard.json \
  --critical-models dronet --heavy-model yolov8_nano_64x96 --window-ms 24
```

**`topo_tag_override: false` is load-bearing**, and its failure is silent. With
it true every combination is costed from `topo_0`, so a 4-hart block is charged
the SINGLE-hart time while occupying four harts. The solver then correctly
never picks one, and the run reports "sharding does not help" for a purely
clerical reason.

Conv sharding needs the weights RE-PACKED per shard, not a pointer offset: the
rvv backends pack conv weights IHWOC, so an OC slice is strided and no offset
expresses it. `shard_conv_weights` gives each shard its own array. Bit-exactness
is not a separate step -- `run_model_k1.sh` golden-compares in-binary every run,
so a shard reading the wrong weights fails the run that would have timed it.

## 4d. What the cost model knows beyond a solo profile

Two questions, same shape, opposite answers. Both were measured on the
ModelBlaster path because the earlier numbers for both came from
`iree-benchmark-module` over `.vmfb` files, and that path is retired.

**Co-runner contention: a NULL.** Does a dispatch slow down because something
else runs at the same time on another hart? Not measurably, up to four
co-runners -- the same-cluster and cross-cluster distributions straddle 1.0 and
overlap completely, and the arms are not monotonic in co-runner count. The
IREE-path figures (1.043x same-cluster, 1.185x cross-cluster) do not reproduce.
Install no contention model. `docs/k1_contention.md`.

**Producer-consumer edge cost: REAL, and network-dependent.** What does it cost
a dispatch to read what the previous one wrote, from another hart? All arms
disjoint, twice, on two networks:

| | same hart | same cluster | other cluster |
|---|---|---|---|
| dronet | 1.000 | 1.068 | 1.111 |
| yolov8_nano_64x96 | 1.000 | 1.033 | 1.070 |

Roughly 6% to leave the hart that produced your input, 10% to leave the
cluster -- and yolo pays ~4pp less than dronet in both classes, so the map is
keyed per network rather than being a global constant. `docs/k1_cost_by_pred.md`.

**The two are different mechanisms and only one is visible here.** On this
board the cross-cluster cost appears when a dispatch is placed away from its
PRODUCER, not when two dispatches sit on opposite sides at once. Three
independent measurements agree on the explanation: dronet is slower on 8 harts
than on 4 (5.32 vs 5.25 ms) and yolo is not; dronet pays more to cross the
edge; dronet's working set fits one L2 and so has more to lose. Looking for
this effect with a co-runner sweep finds nothing, and is right to.

## 5. Guard rails, and what each one caught

Every one of these exists because it failed silently at least once.

* **`check_rvv_avl.py` refuses a `vsetvl` whose AVL is another `vsetvl`'s
  result.** Every instruction is legal and the binary runs to completion; only
  `vl` is wrong, so `check_rvv_vtype.py` -- which reads the disassembly for
  instructions the hardware refuses -- structurally cannot see it. Two
  committed kernels declaring `accuracy_class: bit_exact` were not
  (`max_abs_err` 20 and 68); see section 6.
* **`check_schedule_feasibility` refuses an implementation the core cannot
  execute.** Every other finding it reports is a slowdown -- a double-booked
  core serialises, an overrun still produces numbers. This one does not: an
  `ime` dispatch on CPU_E takes SIGILL and writes nothing, so it surfaces as a
  missing results file rather than a wrong measurement.
* **`compare_candidates` refuses two schedules with the same `pdb_hash`.** They
  were solved against the same measured costs, so whatever the verdict is, it
  is not about the rewrite.
* **`compare_candidates` refuses two schedules with different INSTANCE
  COUNTS.** `pdb_hash` proves the two solves read different costs; nothing
  proved they scheduled the same amount of *work*, and a schedule does not
  record how many refinement iterations produced it. The 4 Hz baseline was
  re-solved without `--max-periodic-iters 1`, the loop grew `mlp_control` from
  32 instances to 91 and `dronet` from 10 to 28, and it was written under the
  baseline's own filename — after three verdicts had been recorded against the
  826-dispatch version. Every term still computed, `pdb_hash` still differed,
  and the figure rendered from it reported **ACCEPT** for a DroNet rung that
  had been adjudicated REJECT. A rewrite changes how many dispatches an
  instance is made of; it must not change how many instances there are.
  `plot_loop_iterations` makes the same check, from the same implementation
  (`schedule_scoring.instances_per_model`), and refuses to draw the panel.
* **A control build is not a rewrite.** The yolo `ctrl` rebuild is 4.5 ms
  faster than the shipping baseline on the same graph, because they came off
  different toolchains. Fed to the objective as a rung it "accepts", crediting
  a compiler to a rewrite that did not happen. Mark it `--control` in the
  figure and judge the real rung against it, not against the shipping build —
  `--judge-against 6=4`.
* **A per-dispatch profile miss costs ZERO, silently.** Check the row count
  against the dispatch count; `profile_loader`'s strict mode only catches a
  wholly absent CSV.
* **`diff_dispatch_graph` exits 4** when a rewriter's own `id_remap` disagrees
  with the op signatures — a broken rewriter, not a negative result.
* **The trace's `dispatch_id` is a record SLOT, not the IR id.** It drifts by
  the number of zero-cost ops (`view`, `chunk2_c1`) before it. On yolov8_nano,
  44 of 90 dispatches join to an op of a different kind, and it reads as a
  *prediction* error: d81 reported "predicted 17.465 ms, measured 0.577 ms".
  Pass `--ir` to `join_k1_trace.py`; the audit is always on and refuses by
  default.
* **A network name may end in a digit.** `yolov8_nano_64x96` split at the wrong
  place made instance 0 read as instance **960**, whose deadline is
  `960 × 50 ms = 48 s` — so the model became structurally incapable of missing
  a deadline and the scorer reported `misses: 0` with
  `response_p50 = −47954 ms`. `job_names.py` owns that split now; it had seven
  independent implementations that disagreed.
* **`impl` was read from a stale loop variable.** The dispatch dict is built
  in a SECOND pass over the operations, so `combo_idx` from the first pass
  holds whatever the last operation was assigned. Every dispatch in a
  heterogeneous schedule came out tagged `rvv` while its duration was plainly
  the IME cost. Caught by cross-checking the field against the report's own
  `combo_idx` rather than by reading it — a schedule that agrees with itself
  proves nothing.
* **A vector kernel that compiles to no vector instructions.** Asking the
  generator for the generic `direct` algorithm on `rvv_x60` produced four
  scalar transformer kernels; all four cross-compiled, verified bit-exact, and
  would have been committed as curated RVV kernels. `cross_compile_verify` now
  disassembles the object — reading the OBJECT, not the source, because
  GCC auto-vectorizes some plain C (softmax, layernorm) and IME's `vmadot` is
  a `.insn` that never looks like an intrinsic.
* **The window is the deadline.** `D = windows_ms.get(m, T)` — omit
  `--windows-from` and you score against the period, a more forgiving test than
  the workload declared.

## 6. Three traps that cost a board slot each

**Use GCC 14.3, not 13.2.** Get it with
`eval "$(scripts/setup_spacemit_toolchain.sh)"`, which finds an existing
install (including an old merlin checkout) or fetches from the vendor, and
REFUSES a 13.x it finds rather than letting you discover the problem on the
board. merlin is no longer a submodule; that script is the whole reason the
live path ever needed it. `run_xpurt_k1.sh` defaults `CROSS` to chipyard's 13.2, while the single-model profiling path uses the spacemit 14.3. GCC 13.2
reorders the `__riscv_vsetvl_*` intrinsics so a widening instruction runs under
the narrow vtype:

```
vsetvli e32,m4    ← sets SEW=32
vsetvli e8,m1     ← clobbers it to SEW=8
vle8.v ×2
vsext.vf4         ← ILLEGAL: widening 8→32 needs SEW=32
```

SIGILL with no stdout, `epc 0x17020`, `badaddr` equal to that instruction's own
encoding. It crashes rather than computing a wrong answer, so no past result is
invalidated.

**GCC 14.3 has its own bug, in the opposite direction, and it is worse.**
Given a `vsetvl` whose AVL is another `vsetvl`'s return value -- which reads as
correct RVV, since e8m1 and e32m4 hold the same number of elements -- it
substitutes an unrelated register. Measured in the avgpool kernel: the second
`vsetvl` was issued with the enclosing loop's BOUND as its AVL, `vl` came out 5
where the output row is 11 wide, and six of every eleven outputs were never
written. No crash, no warning, `max_abs_err=68`.

So the two compilers are wrong in opposite ways, and the mandate that fixed the
loud failure installed a quiet one:

```
GCC 13.2   reorders a vsetvl across a widening op   -> SIGILL, loud
GCC 14.3   wrong AVL on a chained vsetvl            -> wrong answer, silent
```

The only form correct under both is to pass the ELEMENT COUNT to every width,
every time. `scripts/check_rvv_avl.py` refuses the other one.

**`--staged-ir` for anything the loop produced.** A rewritten IR has no
`--model` that can regenerate it, which is the entire point of the loop. The
flag copies the graph, records source path and sha256 in `.staged_from`, and
renames the IR's `name` field to the network name — every generated C symbol
mangles from it.

## 7. Where each concept lives

The modules are not all named after the concepts.

| concept | module |
|---|---|
| granularity verdict | `xpu-rt/granularity_advisor.py` |
| candidate generation + predicted scoring | `xpu-rt/rewrite.py` |
| the search driver | `scripts/granularity_loop.py` |
| hint assembly (one schema, three callers) | `xpu-rt/bundle.py` |
| scoring a solved schedule (one scorer, three callers) | `xpu-rt/schedule_scoring.py` |
| the acceptance rule | `xpu-rt/candidate_objective.py` |
| the verdict CLI | `scripts/compare_candidates.py` |
| workload-spec reading | `xpu-rt/workload_spec.py` |
| job-name splitting | `xpu-rt/job_names.py` |
| trace reading, both producers | `xpu-rt/k1_trace.py` |
| board measurement of a candidate | `ModelBlaster/scripts/measure_candidate.sh --runner k1` |
| multi-core board runs (pool + topo tag) | `ModelBlaster/scripts/run_model_k1.sh` via `MB_CORES` |
| per-shard conv weight re-packing | `ModelBlaster/pipeline/generate_skeleton.py::shard_conv_weights` |
| co-runner contention (a null result) | `runtime/scripts/k1_contention_mb.py`, `docs/k1_contention.md` |
| producer-consumer edge cost | `runtime/scripts/k1_cost_by_pred.py`, `docs/k1_cost_by_pred.md` |
| which implementations a core can run | `xpu-rt/capabilities.py` |
| the chained-AVL kernel lint | `ModelBlaster/scripts/check_rvv_avl.py` |
| shard: advice -> hint | `scripts/advice_to_shard_hint.py` |
| shard: hint -> annotated IR | `ModelBlaster/pipeline/apply_shard_hint.py` |
| per-dispatch implementation, board side | `ModelBlaster/pipeline/ingest_xpurt_schedule.py` (`impl`) |
| runtime feedback, batch | `scripts/run_xpurt_schedule.py --emit-feedback` |
| runtime feedback, streaming | `xpu-rt/streaming_feedback.py`, `MB_XPURT_STREAM=1` |
| measured run -> back onto the advice | `xpu-rt/feedback_join.py`, `emit_compile_advice.py --feedback` |
| both feedback channels, explained | `docs/modelblaster_integration.md` |
| the solver registry | `docs/solvers.md`, `xpu-rt/schedulers.py` |
| what a spec's fields mean | `docs/workload_specs.md` |
| runnable walkthroughs | `examples/` |

## 8. Examples

Every arrow above has a runnable version under `examples/`, and the test suite
runs the subset that needs neither a board nor a licence — so an example
cannot rot quietly into a description of code that no longer exists.

```bash
.venv/bin/python examples/run_all.py
```

| example | shows |
|---|---|
| `examples/feedback_loop/one_revolution.py` | profile -> advice -> hint -> rewrite -> verdict, on measured data |
| `examples/verbs/all_five_verbs.py` | each verb's producer/bridge/consumer, and what each bridge refuses |
| `examples/workloads/anatomy_of_a_spec.py` | the six load-bearing spec fields and their failure modes |
| `examples/k1_board/board_flow.py` | the board flow and every precondition checkable from the host |
| `examples/solvers/compare_solvers.py` | the registry on one workload; slow, opt-in |

`rewrite.py`, `bundle.py` and `granularity_loop.py` were absent from this branch
for a while — added on `origin/xpurt-scheduler-advisor` and never merged
forward, so `decision_loop.py` shelled out to a file that was not there and the
automated loop could not run at all. Restored in `feat/k1-granularity-bridge`.
