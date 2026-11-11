# Flow C — ModelBlaster's front end, a QNN back end

PyTorch models are ingested **exactly** the way the RISC-V flow ingests
them, scheduled by the same `xpu-rt` scheduler, and read back through
ModelBlaster's own schedule ingest. Only the last stage differs: instead
of Zephyr C calling generated kernels on harts, this emits Linux C++
calling `QnnGraph_execute` on backend lanes of a QRB5165.

```
models/<id>.py ──► extract_graph ──► graph.json IR ──┐   (modelblaster, verbatim)
   (one PyTorch definition,          torch.onnx.export│
    one checkpoint)                        └──► .onnx ┤──► DLC ──► ctx binary
                                                     │
   bindings/<net>.json  ── tile map: IR ops ──► QNN graph, per backend
                                                     │
   cores/qrb5165_qnn.json ── registry: kinds, capabilities, lane policy
                                                     ▼
              dispatch_graph.json + results.csv   (modelblaster emitters)
                                                     ▼
                        scripts/run_xpurt_schedule.py   (xpu-rt, unchanged)
                                                     ▼
                   ingest_xpurt_schedule.load()        (modelblaster, verbatim)
                                                     ▼
                 flowc/emit_runtime.py ──► dispatch_table.h + runtime_main.cpp
                                                     ▼
                         deploy_and_run.sh ──► board ──► trace
```

Step-by-step reproduction, the diff against the standard modelblaster flow,
and a file map: [`docs/mlp_dronet_yolo_qnn_reproduction.md`](../../docs/mlp_dronet_yolo_qnn_reproduction.md).

## Run it

```bash
cd qnn_models/flow_c
python3 flow_c.py ir          # extract_graph on modelblaster's model zoo
python3 flow_c.py artifacts   # dispatch graphs + profile CSVs + workload spec
python3 flow_c.py schedule    # xpu-rt scheduler (greedy_periodic; --solver milp needs cvxpy)
python3 flow_c.py runtime     # schedule -> dispatch table + QNN runtime sources
python3 flow_c.py stage       # link the context binaries into the board's ctx dir
python3 flow_c.py run         # build on the board, run, capture the trace
```

`--workload` selects a spec under `workloads/` (default `dronet_mlp_yolo.json`).
`FLOWC_MB_PYTHON` overrides the interpreter used for the torch-dependent
extract stage; everything else runs on stock Python 3.

## What is reused, and what had to be new

| Stage | Source |
|---|---|
| PyTorch → IR | `modelblaster.pipeline.extract_graph`, as a subprocess in its own env |
| ONNX for the converter | `torch.onnx.export` on the same `get_model()` — one checkpoint feeds both flows |
| Core registry | `modelblaster.pipeline.core_registry`, unmodified, reading `cores/qrb5165_qnn.json` |
| `dispatch_graph.json` | `modelblaster.pipeline.emit_dispatch_graph.emit()`, fed a coarse IR |
| `results.csv` | `modelblaster.pipeline.profile_writer.write_profile()`, `clock_mhz=1.0` so a "cycle" is a µs |
| Scheduling | `scripts/run_xpurt_schedule.py` |
| Schedule → entries | `modelblaster.pipeline.ingest_xpurt_schedule.load()` — job-name splitting, slot resolution, topological order, `deps[]` **and** `time_dependency` |
| Trace plotting | `modelblaster/scripts/plot_xpurt_trace.py --clock-mhz 1`, unmodified |
| **Runtime codegen** | **new** (`flowc/emit_runtime.py`) — see below |
| **Tile map + lowering** | **new** (`flowc/bindings.py`, `flowc/lowering.py`) |

The one upstream patch: `ingest_xpurt_schedule._resolve_target()` knows only
`CPU_P` and `CPU_E`; a three-lane board also schedules onto `CPU_X`.
`flowc/mb.py::install_slot_map()` installs the N-slot version until the
six-line change lands upstream.

## Threading and tiling — why not modelblaster's

ModelBlaster's runtime parallelism is `modelblaster_pool`: a parallel-for
that splits **one op's range across harts**. That is the right primitive
when you own the kernel. Here the work inside a dispatch belongs to HVX,
the tensor accelerator, or the CPU op package, and the host cannot
subdivide it — so Flow C parallelises differently:

* **Lanes, not pools.** One `std::thread` per machine kind
  (`--lane-mode kind`) or per (kind, network) (`--lane-mode kind-network`
  when the schedule serialises two networks with very different periods
  onto one kind). Each lane is pinned with `pthread_setaffinity_np` to the
  core the registry gives that kind. Workers block inside
  `QnnGraph_execute`; the lane count *is* the concurrency.
* **Tiles are bindings.** A tile is a contiguous IR op range compiled into
  one QNN graph (`bindings/*.json`). Finer tiles buy the scheduler freedom
  — the yolov8n head can leave HTA, which it must, because its
  Reshape/Resize/Slice/Softmax/Transpose do not compose there — and cost
  one dispatch each (~0.5 ms of FastRPC on the DSP). Re-tiling costs a
  compile, which is why it is a manifest rather than a flag.
* **Instance tiling is free.** Two instances of a network may carry
  different backends; each (context, backend) pair is brought up once and
  shared.
* **Gates sleep, they don't spin.** Each entry waits on
  `clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME)` until `gate_spin_us`
  before its scheduled start, then spins. The spin budget is per lane,
  from the registry.
* **Per-lane scheduling policy.** `qnn.sched_fifo_prio` puts one lane on
  `SCHED_FIFO` without touching the others.

### The measurements behind those defaults

Same schedule, same board, 4 reps each (128 `mlp_control` instances):

| Mechanism | Makespan vs predicted | mlp exec max | mlp deadline misses |
|---|---|---|---|
| spin-yield gate | 1.03× | 0.104 ms | 0 |
| absolute-sleep gate, 200 µs spin, all lanes SCHED_OTHER | 1.00–1.03× | 2.22 ms | 3 / 128 |
| + CPU lane spin 2000 µs | 1.00× | — | — |
| + CPU lane `SCHED_FIFO 40` (shipped default) | 1.00× | **0.106 ms** | **0** |

Two things worth keeping:

1. A timer wakeup on an idle-clocked A78 costs ~1.4 ms, which is most of
   `mlp_control`'s 2 ms period. Widening *only* the CPU lane's spin window
   recovers the makespan as well as widening every lane's — the other two
   lanes should give their cores back while they wait.
2. `SCHED_FIFO` on **one** short-dispatch lane removes the tail entirely.
   The earlier hand-written runtime found process-wide `SCHED_FIFO`
   regressed wall time and disabled it; the difference is scope — an
   accelerator lane on `SCHED_FIFO` starves the FastRPC callbacks it is
   itself waiting on.

## Where the partitioning comes from

A tile boundary has three possible provenances, and the manifest says which
one each binding used (`flow_c.py ir` prints it as `range from:`):

| Binding | Range | Provenance |
|---|---|---|
| `dronet_full` | ops 0..23 | `ops: "all"` — a coarse *declaration*: whole network, one graph. The rationale (whole-DSP 0.92 ms vs 3.49 ms summed over 7 segments) is prose in the manifest, backed by the measurements file. |
| `mlp_control_full` | ops 0..6 | `ops: "all"` — 70k MACs; any cut costs a dispatch to save microseconds. |
| `yolov8n_backbone` / `_head` | ops 0..102 / 103..248 | **derived** — `ops: {from_partition: {group: N}}` reads the contiguous `hardware_target` runs out of `boards/qrb5165_v66/graphs/yolov8n_HTA_split.json`, the file `build_yolov8n_hta_split.py` wrote when it chose the cut. Re-slice upstream and the ranges move here too. |

So the coarse choices are *encoded* (there is no automatic "is this network
worth splitting" pass), and the one real cut is *derived* from the
partitioner's own output rather than transcribed.

The cut is also materialised as artifacts — `yolov8n_backbone.onnx` and
`yolov8n_head.onnx`, sliced by `slice_to_subonnx.py`, built into sub-DLCs
by `build_subdlcs.sh` — and the runtime dispatches purely on
`(context, graph)`. The op range never reaches the runtime except as trace
metadata, which means a manifest can describe a cut the artifact no longer
has. `bindings.verify_against_artifacts()` is the guard: it counts nodes in
each tile's `source_onnx` and checks the partition's `handoff_tensors` are
produced by one tile and consumed by the next.

That check surfaces a real and expected discrepancy: the head declares 146
IR ops but its ONNX holds 138 nodes. Same network, three op counts —
`yolov8n.onnx` 233 nodes (ONNX route), backbone 103 + head 138 = 241 after
slicing materialises the 8 `Split` ops at the cut, and 249 in the QNN-route
IR the partition file carries. The tile *boundary* is what agrees across
all three: group 0 ends where the backbone ONNX ends, and its three outputs
are exactly the head's three inputs. Op-level correspondence between the
routes would need `runtime/build_qnn_to_onnx_namemap.py`; tile-level is
what the runtime actually dispatches on.

The one automated signal about granularity is xpu-rt's
`granularity_advisor.py`, which fires during the solve — on the greedy
schedule it said yolov8n's largest dispatch (21.58 ms) exceeds
mlp_control's free slot (1.97 ms). Flow C prints it and does not act on it;
feeding it back into the manifest would close that loop.

## Capabilities instead of sentinels

The old coarse cost model carried `100_000.0` µs sentinels to keep the
scheduler off HTA for models it could not compose. Here the registry
carries op capabilities, the binding manifest declares which lowerings its
compiled artifact went through (`flowc/lowering.py`), and feasibility is
*derived*:

```
mlp_control_full   runs on: cpu, dsp        hta blocked by elu_s8
yolov8n_backbone   runs on: cpu, dsp, hta
yolov8n_head       runs on: cpu, dsp        hta blocked by reshape, resize,
                                                slice, softmax, transpose
dronet_full        runs on: cpu, dsp, hta   (after the declared BN fold +
                                             conv head + trailing-view drop)
```

Every line agrees with what the board actually reports at compose time.
`dronet` is the instructive case: against the raw PyTorch IR it is
infeasible everywhere, and it only becomes an HTA graph because the ONNX
was rewritten offline — so the manifest declares that rewrite and the
check runs on the result. A binding that forgets to declare a lowering
fails loudly, naming the op that blocked it.

Cells the registry excludes still need *a* number, because `xpu-rt`'s
`Operation` carries a processing time per machine and has no "forbidden"
flag. `flowc/artifacts.py` derives one (100× the largest measured peer),
tags the row `qnn-excluded`, and prints the reason. The clean fix is a
per-op machine mask in `xpu-rt/workload.py`.

## Layout

```
flow_c/
├── flow_c.py                 CLI driver (stages: ir/artifacts/schedule/runtime/stage/run)
├── cores/qrb5165_qnn.json    core registry: kinds, capabilities, lane policy
├── bindings/*.json           tile map: IR op ranges -> QNN graph per backend
├── measurements/*.json       measured per-(binding, backend) times, with provenance
├── workloads/*.json          which networks, which periods, which slots
└── flowc/
    ├── mb.py                 modelblaster import bridge + the N-slot patch
    ├── ir.py                 PyTorch -> IR (extract_graph) | graph.json adapter
    ├── bindings.py           tile map + registry capability check
    ├── lowering.py           declared converter transforms (BN fold, activation fuse, ...)
    ├── artifacts.py          dispatch graphs, profile CSVs, workload spec
    ├── schedule.py           modelblaster ingest + binding join
    └── emit_runtime.py       dispatch_table.h + runtime_main.cpp
```

## Splitting for coverage: FusedSensorNet

The monolithic `fused_full` graph composes on DSP int8 and CPU fp32 only —
HTA rejects the whole thing on the `Transpose` its `Flatten` lowers to, and
the CPU op package rejects the quantized `Reshape` in the LSTM tail. Cutting
the graph gives the scheduler back the placements the op set was hiding:

| Tile | HTA int8 | DSP int8 | CPU int8 | CPU fp32 |
|---|---|---|---|---|
| `vision_conv` (4 convs) | **1.42 ms** | 0.58 | 12.22 | — |
| `depth_conv` (2 convs) | **1.13 ms** | 0.38 | **0.014** | — |
| `vision_head` (+Flatten+FC) | ✗ Transpose | 1.30 | 14.27 | — |
| `depth_head` (+Flatten+FC) | ✗ Transpose | 0.80 | 0.037 | — |
| `tail` (cat + 3x LSTM + head) | ✗ Transpose | 3.06 | ✗ Reshape | **0.37 ms** |

The Flatten is the boundary that costs HTA, so the useful cut is before it:
two parallel conv branches plus the tail. `bindings/fused_split.json` is that
manifest — and its tail is the first tile with non-contiguous IR ops (4-5 and
8-15, with the depth branch's 6-7 in between), because parallel branches
interleave in dispatch-id order.

What it buys, measured over 3 reps of the 4-way schedule (MOSEK optimal on
66 ops, which puts the vision branch on HTA):

| | monolith | split |
|---|---|---|
| FusedSensorNet CPU work per instance | 6.39 ms | **1.48 ms** |
| mlp_control exec p50 / p90 | 0.106 / 0.138 ms | **0.058 / 0.085 ms** |
| mlp_control worst lateness | +9.32 ms | **+2.36 ms** |
| dronet exec p50 / max | 2.50 / 6.02 ms | **1.77 / 2.27 ms** |

The makespan barely moves (yolov8n bounds it either way) — the win is that
4.9 ms of contended CPU work per instance moves onto idle accelerator
silicon, and every other network on the shared cores gets faster and more
punctual. Note CPU int8 is only worth taking on the small branch: QnnCpu's
int8 conv is a reference kernel (12.2 ms for the vision branch against
0.014 ms for the 8x8 depth branch).

## Predicted vs actual: measure in the state you run in

The 4-way schedule ran 1.07-1.16x its prediction until the cost model and the
run were put in the same machine state. Four causes, separated by experiment:

| Cause | Evidence | Fix | Effect on makespan |
|---|---|---|---|
| Cells measured under a different governor | `schedutil` idles the board at 710 MHz of 2419; the host clock gates FastRPC, so accelerator cells come out 10-36% pessimistic (yolov8n backbone on HTA: 21.6 ms under schedutil, 13.9 ms under performance) | measure and run under `performance` | 1.16x → 1.07x |
| Cold first walk — page faults, cold caches, DVFS ramp | iteration 1 vs iteration 2 of the same schedule | `FLOWC_ITERATIONS=2`, report iteration 2 | 1.16x → 1.06x |
| Both together | | `flow_c.py run --tuned` | **1.16x → 1.00x**, 0.08 ms spread over 3 runs |
| CPU-lane contention | the fp32 tail is 0.35 ms alone and 3.69 ms in-situ, while every accelerator tile lands at 0.95-1.12x | not fixed — see below | residual |

Per-tile accuracy before and after, pooled over 3 runs each:

```
BEFORE  schedutil cells, schedutil run, no warm-up      ratio spread 0.70x .. 18.46x (median 1.14x)
AFTER   performance cells, performance run + warm-up    ratio spread 0.64x .. 10.44x (median 0.96x)
        of which: yolov8n backbone 0.95x  head 1.12x  vision_conv 0.96x  tail@dsp 0.95x
                  dronet 0.79x  mlp_control 1.43x  tail@cpu 10.44x  <- the one outlier
```

`--tuned` sets the governor on the board, runs with a warm-up iteration, and
restores the governor afterwards. `measurements/*.json` records the conditions
its cells were taken under; the pre-governor cells are kept alongside in
`_previous_cells_schedutil` so the two are comparable.

One structural note: the makespan is bounded by a scheduler-inserted
`time_dependency` — dronet's last instance waits for yolov8n's head because
the solver serialised them onto the DSP — so a small overrun on the head
propagates straight to the wall clock. That is the schedule working as
specified, not runtime jitter.

## Feedback: what it fixes and what it cannot

`flow_c.py feedback --tag <t>` reads a run's trace and promotes each tile's
in-situ median into the cost model, tagged with the run it came from. Use it
when a cell is wrong for a *stable* reason — measured under the wrong
governor, or under an affinity mask the runtime does not actually apply.

It does not converge for a multi-threaded CPU tile, and ViNT shows why:

```
vint_decoder @cpu   12.7 ms  standalone, unmasked, idle board
                    37.8 ms  in-situ, promoted to a cell by `feedback`
                    12.8 ms  next run — the new schedule gave it more machine
```

Measuring changed the schedule, which changed the measurement. A scalar cell
cannot express a cost that depends on concurrent load. Tiles on dedicated
silicon have no such problem: ViNT's encoders hit 1.01x of their DSP cell in
every run, and dronet/yolov8n sit inside 5-12% of theirs. The two real fixes
are a contention term in the cost model, or binding the QNN CPU backend's
thread pool so the cost stops depending on neighbours — neither of which is
a measurement change.

## Known gaps

* **A CPU-heavy network's real cost is not its isolated cell.** FusedSensorNet
  measures 1.147 ms alone under its lane's exec mask and ~6.4 ms (p90 10.2 ms)
  inside the 4-way schedule. The lane's execute mask only binds the lane
  thread; the QNN CPU op package builds its thread pool at bringup, on the
  main thread, with full-machine affinity — so it contends with every other
  lane, the real-time-gated one included. Networks on their own silicon
  (dronet on HTA, yolov8n on DSP) match their cells to within 5%.

* **Tensor handoff is name-matched and currently finds nothing.** The
  runtime caches each graph's outputs by tensor name and feeds a later
  graph's identically-named input; the yolov8n backbone→head pair does not
  match because the two sub-DLCs got different converter-assigned names.
  Run with `FLOWC_DUMP_TENSORS=1` to see them; the fix is an explicit
  `handoff` mapping in the binding manifest. Until then, timings are
  real and outputs are not.
* **yolov8n's IR comes through the graph_json door**, not `extract_graph`:
  the network on this board is the 640×640 ultralytics export behind the
  QNN converter, not modelblaster's 64×64 `yolov8_nano`. Same schema,
  different provenance.
* **`--solver milp` needs `cvxpy`**, which is not installed in this
  checkout; the shipped workload uses `greedy_periodic`.
* **Op-level correspondence** between the PyTorch IR and the compiled
  graph is structural, not name-by-name. `runtime/build_qnn_to_onnx_namemap.py`
  is the missing hop for finer tiles.
