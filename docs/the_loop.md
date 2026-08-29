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

`shard` is deliberately unwired: it needs multi-core profiles that do not
exist, and B4 measured 4-way OC sharding at **+76% total work**, a 2.27×
ceiling. Documented, not built.

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
export CROSS=<spacemit>/bin/riscv64-unknown-linux-gnu-   # NOT the default; see §6
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

### The figure

```bash
python scripts/plot_loop_iterations.py \
  --iteration "baseline · dronet conv0 OC=32=<baseline schedule>" \
  --iteration "split x2 · dronet conv0 OC=2x16=<x2 schedule>" \
  --iteration "split x4 · dronet conv0 OC=4x8=<x4 schedule>" \
  --windows-from <spec> --critical-models mlp_control,dronet \
  --heavy-model yolov8_nano --out-dir out/figures
```

→ `out/figures/k1_loop_evolution.png`, one panel per rung on the real machine
lanes, each labelled with its dispatch count, miss count, p99, makespan and the
term its verdict turned on. `diff_dispatch_graph` proves the *graph* changed;
this shows whether the *schedule* did, which is a different question and the one
that decides.

## 5. Guard rails, and what each one caught

Every one of these exists because it failed silently at least once.

* **`compare_candidates` refuses two schedules with the same `pdb_hash`.** They
  were solved against the same measured costs, so whatever the verdict is, it
  is not about the rewrite.
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
* **The window is the deadline.** `D = windows_ms.get(m, T)` — omit
  `--windows-from` and you score against the period, a more forgiving test than
  the workload declared.

## 6. Two traps that cost a board slot each

**Use GCC 14.3, not 13.2.** `run_xpurt_k1.sh` defaults `CROSS` to chipyard's
13.2, while the single-model profiling path uses the spacemit 14.3. GCC 13.2
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

`rewrite.py`, `bundle.py` and `granularity_loop.py` were absent from this branch
for a while — added on `origin/xpurt-scheduler-advisor` and never merged
forward, so `decision_loop.py` shelled out to a file that was not there and the
automated loop could not run at all. Restored in `feat/k1-granularity-bridge`.
