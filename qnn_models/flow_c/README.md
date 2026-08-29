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
