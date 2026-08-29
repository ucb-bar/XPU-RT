# What landed on `dev` and `main` — a reader's guide

Everything on this branch, in the order a new reader should meet it. Two repos
moved together and neither makes sense alone.

```
ucb-bar/XPU-RT        dev    935ce59 → 0904dde    198 commits
ucb-bar/ModelBlaster  main   dbbdcf0 → 2f1cec8    296 commits
```

**Start here:** [`the_loop.md`](the_loop.md) is the index — every arrow of the
compiler↔scheduler cycle and which script owns it. Then
[`environment.md`](environment.md) to get it running, then `examples/`.

---

## 1. The headline

The loop is closed, and it has been round once on real hardware. One
revolution on the K1, ~35 s of wall clock, ending in a verdict rather than a
percentage:

```
baseline   misses=0  worst_late=0.000  p99=26.850  makespan=56.846
candidate  misses=0  worst_late=0.000  p99= 6.500  makespan=36.496
VERDICT: ACCEPT — p99 response of critical tasks, 6.5 vs 26.85
```

Decided by **term 4** of the nine in `candidate_objective.py`. Not makespan
(term 7), and not standalone kernel cycles (term 9 — the one the module's own
docstring calls "never the deciding term", and the one every earlier rung was
adjudicated on by eye).

The rewrite behind it: `ffn_block`'s two linears sharded 8 ways on measured
per-width profiles, **644313 → 155903 ticks (4.13×), `max_abs_err=0`** on the
board. Worst CV was 48.7% at 3 reps — the 4.13× is real, the third digit is
not.

## 2. The best story in here is about testing

Running that revolution broke at **four separate points**, all green in 904
unit tests beforehand:

1. `--fp16-ops` registered twice in argparse — both branches added it, the
   merge kept both, and argparse raises at import. Every extract died before
   reading a model.
2. 28 `_weight_name()` calls not passing `backend`, so `model.c` referenced
   `ffn_block_ln_gamma` while `weights.h` declared `ffn_block_ln_gamma_rvv_x60`.
3. The Linux xpurt harness staged `test_io.h` with `COPYONLY`, leaving its own
   `#include "model.h"` dangling. The Zephyr harness has rewritten that since
   `84a98cc`; the Linux path never got the fix.
4. Six test assertions matching pre-suffix symbol names inside generated calls.

None of them is subtle. All of them needed a board, a cross-compiler and a
real model to surface. The suites test the pieces; only a revolution tests the
seams.

## 3. Everything that changed, by area

### The five verbs — all now have complete chains

`producer → bridge → consumer`. A verb missing any of the three is advice
nobody can act on, which is what `shard` was for most of this project's life.

| verb | producer | bridge | consumer |
|---|---|---|---|
| fuse | `compile_advice.overhead_advice` | `scripts/advice_to_fusion_hint.py` | `pipeline/apply_fusion_hint.py` |
| split | `blocking_advice` | `advice_to_split_hint.py` | `apply_split_hint.py` |
| unfuse | `unfuse_advice` | `advice_to_unfuse_hint.py` | `apply_unfuse_hint.py` |
| **shard** | `shard_advice` | **`advice_to_shard_hint.py`** (new) | **`apply_shard_hint.py`** (new) |
| choose_implementation | `implementation_advice` | `advice_to_kernel_choice.py` | `--keep-reference-ops`, and the schedule's `impl` |

**`shard` is the odd one out** and worth reading `advice_to_shard_hint.py`
for: it does not rewrite the graph. It annotates one dispatch with a core
width — same dispatch count, same ids, same edges. Everything else in that
table is a graph rewrite, and the contract spells the count `n_shards` rather
than `n_splits` so a hint fed to the wrong applier fails instead of quietly
doing the other verb.

Sharding is also now **per dispatch** rather than per model
(`MB_SHARD_FACTOR`), because measured scaling varies **4.8× within one model**
— 4.02× on a wide-OC conv down to 0.83× on a 1×1.

### Feedback — two channels, and they are not the same channel

Read [`modelblaster_integration.md`](modelblaster_integration.md).

| channel | says | file |
|---|---|---|
| compile advice | how to **rewrite** the graph | `compile_advice.json` |
| runtime feedback | how to **place and size** what exists | `xpurt_feedback.json` |

Runtime feedback has two producers — `run_xpurt_schedule.py --emit-feedback`
(batch, from the solver's own arrays) and `streaming_feedback.py` (live, from
the board) — and one consumer: **`emit_compile_advice.py --feedback`**.

The consumer is the advice producer rather than ModelBlaster directly, and
that is forced: turning "ran slower than predicted" into a split factor needs
the periodic budget, a fusion needs the graph, and `pin_target` names a
machine combination rather than a kernel. So the measured run **corroborates
or contradicts** advice derived from profiles, and never manufactures it.

### Per-dispatch implementation choice now reaches the binary

The solver could already emit it; ModelBlaster ignored the field entirely, so
a heterogeneous schedule produced a binary that quietly ran one backend
everywhere and reported the runtime it got. The walker selects on `impl` now
and **reboots loudly** if asked for one the build lacks.

### merlin is retired

The submodule, the IREE/VMFB runtime, and nine QRB5165 scripts that hardcoded
`/scratch2/agustin/merlin`. Flow B in the README is the K1 board now — the
same ModelBlaster codegen as Flow A, cross-compiled for Linux/riscv64.

**Kept on purpose:** `qnn_scheduler/` and `qrb5165_costs.json` (measured data
outlives the compiler that fed it), `workload_factory`'s `old_merlin_prefix`
rewrite (historical specs still resolve), and two schemas that keep merlin's
spelling because renaming would touch every reader to change nothing —
`results.csv` and the trace's `dispatch_key`/`run_us`/`queue_delay_us`.

The cross toolchain moved out of `merlin/build_tools/` into
`tools/riscv-tools-spacemit/`, so nothing depends on merlin being on disk.

### Board measurements

* [`k1_contention.md`](k1_contention.md) — do concurrent dispatches slow each
  other down? **Null result.** The distributions overlap and the arms are not
  monotonic in co-runner count. Worth reading before anyone models contention.
* [`k1_cost_by_pred.md`](k1_cost_by_pred.md) — what it costs to read what the
  previous dispatch wrote, from elsewhere. ~6% off-hart, ~10% cross-cluster,
  and it is a **model fitted to three measured classes**, not 64 independent
  measurements. The artifact says so in a `derivation` field; anything quoting
  it should too.

### Documentation

| doc | what it is for |
|---|---|
| [`the_loop.md`](the_loop.md) | **the index.** Every arrow, and which script owns it |
| [`environment.md`](environment.md) | **new.** Recreating the env — venv, toolchain, board |
| [`modelblaster_integration.md`](modelblaster_integration.md) | **new** (replaces `merlin_integration.md`). Both feedback channels |
| [`solvers.md`](solvers.md) | **new.** The registry, the two axes, why makespan is term 7 |
| [`workload_specs.md`](workload_specs.md) | **new.** The six load-bearing spec fields and their failure modes |
| [`k1_contention.md`](k1_contention.md), [`k1_cost_by_pred.md`](k1_cost_by_pred.md) | board measurements, with what is *not* established |
| [`merging_to_dev.md`](merging_to_dev.md) | what landed, and the one thing still blocked |
| [`k1_modelblaster_xpurt_closed_loop.md`](k1_modelblaster_xpurt_closed_loop.md) | the K1 closed-loop narrative |

### Examples — runnable, and tested so they cannot rot

```bash
.venv/bin/python examples/run_all.py
```

`tests/test_examples.py` runs the cheap subset and asserts every example on
disk is referenced by `run_all.py`, because a new example nobody runs rots
exactly like an old one.

| example | shows |
|---|---|
| `examples/feedback_loop/one_revolution.py` | the full cycle on measured data; refuses to fake the board steps |
| `examples/verbs/all_five_verbs.py` | each verb's chain, and what each bridge **refuses** |
| `examples/workloads/anatomy_of_a_spec.py` | the six spec fields that have caused wrong answers |
| `examples/k1_board/board_flow.py` | the board flow and every precondition checkable from the host |
| `examples/solvers/compare_solvers.py` | the registry side by side — headline result is that they **agree** |

## 4. Traps worth knowing before you touch the board

* **GCC 13.2 reorders `vsetvl`** → the binary SIGILLs with no stdout.
  **GCC 14.3 substitutes a wrong AVL on a chained `vsetvl`** → a wrong answer,
  silently. Two committed kernels shipped that way. Pass the element count to
  every width; `check_rvv_avl.py` enforces it.
* **`/proc/loadavg` floors at exactly 2.00** from two D-state kernel threads.
  Per-CPU `/proc/stat` is the only valid busy signal.
* **IME (`smt.vmadot`) is cluster 0 only.** `{"cpu_p": 8}` — the runbook's own
  recommendation — SIGILLs. Use `{"cpu_p": 4, "cpu_e": 4}` and let
  `check_schedule_feasibility.py` refuse the schedule first.
* **A digit-ending network name splits wrong** without the known-name set.
  `yolov8_nano_64x960` → `yolov8_nano_64x` + instance 960, deadline ~48 s,
  zero misses forever. `job_names.py` owns that split; ModelBlaster's copy is
  pinned to agree by test.
* **`gen_mb/profile` is a symlink** to `gen/profile_mb`. `find` does not follow
  it and will report the tree as empty.

## 5. What is *not* done

* **Flow A was not verified end to end.** Its conda env has never been
  installed in this checkout, so all four spike specs fail at env activation
  before reaching anything this branch touched. Someone with a working Flow A
  environment should run it once against `main`.
* **The zephyr submodule bump is blocked** by a 403 on
  `ucb-bar/zephyr-chipyard-sw`. Until someone with write access lands
  `feat/firesim-bitexact-profile-recalibration` (on the fork), the two
  ModelBlaster checkouts cannot converge. See
  [`merging_to_dev.md`](merging_to_dev.md).
* **`mosek.lic` is in history** on five refs. Treat it as compromised and
  re-issue; rewriting published branches is the owner's call.
* **One revolution is not a campaign.** The loop has been round once, on one
  model, with one verb. The other four verbs' chains are tested against the
  real rewriters but have not each been taken around on hardware since.
