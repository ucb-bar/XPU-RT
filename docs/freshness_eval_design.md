# Freshness-validity evaluation — design

Audit and design for the smallest scientifically defensible test of one claim:

> A dependent robotic control pipeline can keep meeting its local deadlines while
> producing invalid outputs, because its inputs are stale.

Written after the repository audit and updated to match what was actually built,
so the "files modified" and "commands" sections describe the real thing rather
than an intention.

Companion docs: `benchmarks/freshness_eval/README.md` (how to run it),
`gen/profile/**/_provenance.json` (per-cell timing provenance).

---

## 1. Audit summary

### What `dev` had

`origin/dev` carries the scheduler core (`scheduler.py` MOSEK MILP,
`greedy_scheduler.py`, `workload_factory.py`, `profile_loader.py` with strict
mode, `postprocessing.py`, `granularity_advisor.py`) but **none** of the
evaluation infrastructure: no scheduler registry, no `metrics.py`, no list
schedulers (EDF/HEFT/PEFT), no CP-SAT, no `plot_gantt.py`, no sweep runners, no
freshness concept. `tests/` at the repo root is empty.

Local `dev` was 1 commit ahead / 66 behind `origin/dev`, so `git pull --ff-only`
could not run. The one local commit (`c94fbe5`, marked "do not push upstream")
holds `qnn_scheduler/`, `mosek.lic`, the runtime C driver, and — critically —
`workload.py`'s `deadline_us` / `skip_allowed` / `infeasible_combinations`
fields, which `origin/dev` lacks. It is preserved as branch
`premerge-local-dev-c94fbe5`.

### What `feat/policy-sweep-eval` had

A fast-forward from the old merge-base: 85 commits, 278 files, 96% new files.
Contains the whole registry/reporting/policy/diagnostics stack. Also contains
~70 files of unrelated QNN/smolVLA/sims work and several committed *generated*
fixtures (one is a 10391-line solved schedule), none of which was taken.

`origin/dev` had meanwhile absorbed several of the branch's own commits
(`13e2316` strict profile loader, `adc371d` greedy_periodic + decomposed,
`647041c` horizon fix, `cf8df59` unified entrypoint) and added
`granularity_advisor.py`, which the branch predates and would have deleted.

### What ModelBlaster had

Branch `feat/agentic-fusion-loop`, 142 commits ahead of `main`. Schedule JSON
ingestion (`pipeline/ingest_xpurt_schedule.py`), dispatch-table codegen, a
schedule-driven multi-network harness (`harness_xpurt/`), FireSim-measured
per-dispatch costs (`benchmarks/profile_db/*.jsonl`), and a dispatch-graph
emitter (`pipeline/emit_dispatch_graph.py`) that writes exactly the shape
XPU-RT's `dispatch_deps_path` reader wants.

**No freshness, staleness, or data-age concept anywhere** — the only "stale"
machinery is `pdb_hash`, which is about build-artifact staleness and is never
written by any producer.

### The finding that most constrains the claim

**There is no dataflow between the perception and control networks, at any
layer.**

| layer | what an "edge" is |
|---|---|
| XPU-RT `edges` | `Operation.predecessors` → a MILP precedence constraint (`workload_factory.py:733-760`) |
| ModelBlaster `deps` / `time_dep_entry_id` | `k_sem` ordering edges (`ingest_xpurt_schedule.py:475,488`) |
| buffers | `buf_<model_id>_<tensor>` (`generate_skeleton.py:113-124`) — mangled by model id only: no instance, no version, no slot |

`harness_xpurt/CMakeLists.txt:191-207` *enforces* one buffers translation unit
per model so all backends share storage, and all K instances of a network share
one output buffer. Per-instance output buffers were attempted and reverted
(ModelBlaster `0e0eab8`).

So which producer instance a consumer consumed is **inferred from timestamps,
never recorded**, and that remains true on hardware. Every record carries
`producer_instance_provenance = inferred_from_schedule_timestamps`.

---

## 2. Files and classes

### Added

| path | contents |
|---|---|
| `xpu-rt/freshness.py` | `Invocation`, `FreshnessEdge`, `FreshnessRecord`, `FreshnessEvaluation`, `evaluate_freshness`, `select_producer`, `aggregate_metrics`, `freshness_edges_from_config`, `criticality_from_config`, `split_instance_name`, `analytic_age_supremum`, `analytic_age_ceiling_realized` |
| `benchmarks/freshness_eval/trace.py` | `invocations_from_fixture`, `periodic_spec`, `soft_utility` — the only module that knows the fixture schema |
| `benchmarks/freshness_eval/run.py` | `POLICIES`, `materialise`, `run_schedule`, `compute_a0`, sweep + artifact writing |
| `benchmarks/freshness_eval/plot.py` | plots 1–3 |
| `scripts/export_profile_db_to_results_csv.py` | ModelBlaster `profile_db` → IREE `results.csv` bridge |
| `data/toplevel/freshness_canon_300ms.json` | canonical workload |
| `xpu-rt/tests/test_freshness.py`, `test_profile_bridge.py`, `test_periodic_instances.py` | 59 tests |

### Modified

| path | change |
|---|---|
| `xpu-rt/workload_factory.py` | `num_instances` override applied on **every** return path (was dropped when all networks were periodic) |
| `xpu-rt/postprocessing.py` | automerge post-pass made opt-in; `automerge_enabled()` |
| `xpu-rt/profile_loader.py` | `preferred_hw` that matches no combination now raises |
| `xpu-rt/schedulers.py` | compaction made opt-in; `compaction_enabled()`; ML scheduler entries removed |

### Ported from `feat/policy-sweep-eval`

Registry cluster (`schedulers`, `metrics`, `scheduler_heft`, `scheduler_cpsat`,
`compaction`, `automerge`, `oracle`), reporting cluster (`profiling`,
`plot_gantt`, `report`, `advisor`, `fusion_advisor`, `postmortem`),
`decision_formulas`, `policies/`, `diagnostics/`, 9 test modules.
`scheduler.py`, `postprocessing.py`, `run_xpurt_schedule.py`,
`profile_loader.py`, `workload.py`, `workload_factory.py` were **three-way
merged**, not overwritten, so `dev`'s granularity-advisor work survives.

---

## 3. Data flow

```
ModelBlaster                                  XPU-RT
────────────                                  ──────
examples/<m>/int8/generated/graph.json
  │  pipeline/emit_dispatch_graph.py
  └────────────────────────────────────────►  gen/vmfb/<m>/<target>/gemmini/
                                                <m>.int8/<m>.int8_dispatch_graph.json
benchmarks/profile_db/<m>__<be>__int8.jsonl                        │
  │  scripts/export_profile_db_to_results_csv.py                   │
  │    median cycles/dispatch, cycles→ms @ assumed 1 GHz           │
  └────────────────────────────────────────►  gen/profile/<be>/<target>/<m>/     │
                                                <m>.int8/topo_0/results.csv     │
                                                + _provenance.json              │
                                                       │                        │
                    data/toplevel/freshness_canon_300ms.json ◄───────────────────┘
                      networks · periods · num_instances · criticality
                      edges: []          ← precedence, deliberately empty
                      freshness_edges: [dronet → mlp_control]
                                                       │
                          benchmarks/freshness_eval/run.py::materialise
                                       (burst B, preferred_hw, seed)
                                                       │
                                  scripts/run_xpurt_schedule.py
                                    profile_loader → workload_factory
                                    → schedulers.get_scheduler(...)
                                    → postprocessing.output_scheduled_json
                                                       │
                              schedules/scheduled_<stem>_<tag>_profiled.json
                                       (dispatch rows + metadata.pdb_hash)
                                                       │
                        benchmarks/freshness_eval/trace.py::invocations_from_fixture
                              group by job_name → per-instance [min start, max end]
                              release/deadline from the workload spec
                                                       │
                                   xpu-rt/freshness.py::evaluate_freshness
                                     × phi ∈ {A0+δ}   (post-hoc, no re-solve)
                                                       │
                     results/freshness_eval/{per_invocation,aggregate,intervals}.csv
                                       manifest.json · command.txt · git_commits.json
                                                       │
                                 benchmarks/freshness_eval/plot.py → figures/
```

---

## 4. Timing semantics

**Unit: milliseconds, everywhere in this evaluation**, declared to the evaluator
as `time_unit="ms"` and recorded in the manifest.

- Config `period` / `window_duration` / `start_time` are ms (repo convention).
- `profile_loader` fills `processing_times` in ms.
- Fixture `start_time` / `duration` are ms.
- The metrics sidecar labels the same quantity `makespan_us`. That key is
  **mislabelled** and is not used here.
- `Operation.deadline_us` is never set on this path (`run_xpurt_schedule.py` has
  no flag for it), so the known hazard of comparing `deadline_us` against
  millisecond durations in `scheduler.py:562-577` is latent, not live. Consumer
  deadlines are computed by the evaluator instead.

**Cycles → ms is a documented assumption, not a measurement.**
`mean_time = cycles / 1e6`, i.e. an assumed 1 GHz, matching ModelBlaster's
`cycles_per_ms: 1000000`. The Alveo U250 bitstreams close timing at **25–30 MHz**
(`config_build_recipes.yaml:96,112`), and the on-device trace counter is mtime at
~1 MHz. At 25 MHz a DroNet inference is 361 ms and a 10 ms control period is
impossible, so the millisecond-denominated workload exists only under the 1 GHz
assumption. Uniform, recorded everywhere, cancels in relative comparisons,
invalidates unqualified absolute claims. Raw cycles are preserved.

### Per-instance intervals

`start = min(start_time)`, `end = max(start_time + duration)` over an instance's
dispatches — the perception result is unusable until its last dispatch writes,
and the control command is not emitted until the controller's last dispatch runs.
`release = phase + instance × period`, `deadline = release + window_duration`,
both from the **spec**, not the fixture. A fixture whose instance starts before
its release is rejected.

---

## 5. Producer-consumer semantics

For consumer invocation `C_j`, `P_k(j)` is the producer instance with the
greatest `end_time` among those satisfying `producer_end_time <=
consumer_start_time`.

```
producer_sample_time  = producer_release_time     (sample_time_semantics; also
                                                   producer_start, or explicit)
input_age_at_start_j  = consumer_start_time_j - producer_sample_time_k(j)
input_age_at_output_j = consumer_end_time_j   - producer_sample_time_k(j)

deadline_valid_j  = consumer_end_time_j <= consumer_deadline_j
freshness_valid_j = input_age_at_output_j <= phi
output_valid_j    = deadline_valid_j AND freshness_valid_j
```

`invalid_reason ∈ {valid, deadline_miss, stale_input, deadline_and_stale,
no_completed_producer}`, a partition of the total.

Four decisions worth stating explicitly:

1. **`latest_completed` selects by max `end_time`**, not release order and not
   instance index. With a 13.4× backend asymmetry producers complete out of
   release order, and a consumer cannot read a result not yet written.
2. **The boundary is inclusive.** Equality is the normal case in solver output,
   so excluding it would reclassify tight schedules as `no_completed_producer`.
3. **The window is applied at output**, because the actuation command is only
   emitted then. `input_age_at_start` is kept for the alternative reading.
4. **Freshness edges are NOT precedence edges.** A controller that blocks
   waiting for perception can never consume a stale input — it can only miss its
   deadline. A precedence edge would delete the phenomenon. `freshness_edges` is
   a separate top-level key and declaring a pair under both raises.

### Why φ is anchored on A0

The uncontended input-age ceiling is **not** the producer period. With DroNet
`T=50, L=17.973` feeding control `T=10, L=0.546`, the uncontended age set is
exactly `{20.5, 30.5, 40.5, 50.5, 60.5}` ms, so `A0 = 60.546` ms (the closed-form
supremum `T_p + L_p + L_c = 68.5` is not attained because the consumer period
divides the producer period).

`φ = 50 + δ` for `δ ∈ {5,10}` would therefore sit **below** A0 and report
staleness caused by the 50 ms sampling rate rather than by contention. The sweep
uses `φ = A0 + δ`, `δ ∈ {5,10,20,30,50}` ms, and A0 is cross-checked against the
closed form in `test_freshness.py::AnalyticCrossCheck`.

---

## 6. Experiment matrix

Epoch 300 ms = 6 DroNet periods = 30 control periods. Control 10 ms and
perception 50 ms exactly as specified; only the epoch was stretched, because a
100 ms epoch with a 0–4 YOLO burst is infeasible by ~17× on measured timings
(full-res `yolov8_nano` is 418 ms on gemmini — 4.2 epochs for one instance).

| network | role | criticality | gemmini | rvv_opu | ratio |
|---|---|---|---|---|---|
| `mlp_control` | consumer, hard chain | hard | 0.546 ms | 0.546 ms | **1.00×** |
| `dronet` | producer, hard chain | hard | 17.973 ms | 241.462 ms | 13.4× |
| `yolov8_nano_64` | soft interference | soft | 67.202 ms | 1069.004 ms | 15.9× |

Gemmini load, and why the contention axis is well-posed:

```
B=0  124.2/300 =  41%       B=3  325.8/300 = 109%  oversubscribed
B=1  191.4/300 =  64%       B=4  393.0/300 = 131%
B=2  258.6/300 =  86%
```

| axis | values |
|---|---|
| contention `B` | 0, 1, 2, 3, 4 YOLO instances/epoch |
| window `φ` | `A0 + δ`, `δ ∈ {5, 10, 20, 30, 50}` ms |
| policy | `static_nominal`, `edf`, `heft`, `static_conservative`, + derived `oracle` |
| seeds | deterministic policies; seeds are a determinism control |

`static_conservative` applies **one** mechanism: reserve the fast accelerator for
the producer (`preferred_hw`, a soft cost penalty). Also forcing YOLO onto the
vector unit was tried and dropped — 1069 ms/instance means no instance finishes
in the epoch, soft utility is 0 at every B, and neither mechanism could be
attributed. That belongs to the degraded-safety candidate.

The `oracle` is a post-hoc upper bound per `(B, φ)`, not a deployable policy.

**Criticality rationale.** `dronet` is the producer in the hard-validity chain,
`mlp_control` the hard-critical consumer, `yolov8_nano_64` soft interference.
Criticality defaults to `soft` so a task cannot be silently promoted and inflate
the hard-validity denominator.

---

## 7. Implementation order

1. `eval: add producer-consumer freshness semantics` — `xpu-rt/freshness.py`
2. `eval: add freshness validity unit tests` — 7 specified cases + boundaries + analytic cross-check
3. `workload: add explicit DroNet-to-control dependency` — `freshness_edges` schema
4. `profiles: bridge ModelBlaster profile_db into the XPU-RT profile tree`
5. `fix: three silent-failure modes in the ported scheduling path`
6. `benchmark: add contention and freshness-threshold sweep`
7. `plots: add deadline-versus-freshness figures`
8. — **Decision Gate A** —
9. schedule candidates C0/C1/C2; hysteretic epoch-level selector; adaptive comparison
10. — **Decision Gate B** —
11. ModelBlaster candidate validation on FireSim (through the shared queue)

---

## 8. Commands

```bash
PY=/scratch2/agustin/miniforge3/envs/merlin-dev/bin/python
export PYTHONPATH=/scratch2/agustin/XPU-RT:/scratch2/agustin/XPU-RT/xpu-rt

# tests
$PY -m pytest xpu-rt/tests/ -q

# regenerate the profile bridge (offline; no FPGA)
$PY scripts/export_profile_db_to_results_csv.py

# sweep
$PY -m benchmarks.freshness_eval.run \
    --config data/toplevel/freshness_canon_300ms.json \
    --output-dir results/freshness_eval \
    --seeds 0 --bursts 0,1,2,3,4 --deltas 5,10,20,30,50 \
    --policies static_nominal,edf,heft,static_conservative --cell-timeout 900

# figures
$PY -m benchmarks.freshness_eval.plot \
    --input results/freshness_eval --output figures/freshness_eval
```

---

## 9. Known limitations

- **Freshness is imposed and evaluated analytically, not observed.** No
  producer→consumer dataflow exists (§1); the consumed instance is always
  inferred from timestamps, on hardware too. Follow-up: a version/sequence word
  per buffer plus a trace column — ModelBlaster `0e0eab8` reverted per-instance
  buffers for a *reporting*-path bug, not a routing one, so that is the thread to
  pull.
- **φ is workload-anchored, not application-derived.** A0 is a measured property
  of the periods and latencies, not a control-stability requirement.
- **Timings are measured cycles under an assumed 1 GHz**, not the 25–30 MHz the
  bitstreams run at (§4).
- **`mlp_control` has no backend differentiation** — identical profiles on both
  clusters, because `kernels/gemmini/` has no linear kernel and an all-linear
  model falls back to the same scalar reference. "Isolate control on a separate
  core" is therefore unavailable, and the consumer's placement is not a degree of
  freedom.
- **Deadline success is near-structural** for a 0.546 ms consumer in a 10 ms
  window on two clusters. The divergence is real but bounded above only by the
  oversubscribed `B>=3` points; it must not be presented as an open-ended gap.
- **Makespan is not a contention metric here** — pinned near 290.5 ms at every B
  by the last control release.
- **`latest_completed` is one consumption policy** among several.
- **`static_conservative` is a soft preference**, not a hard reservation.
- **Trace-driven.** These are solver schedules; nothing has run on the FPGA. The
  runtime hot-swap path is separately blocked:
  `runtime/tools/xpurt_scheduler_runner.c` assigns six
  `scheduler_runner_config_t` fields absent from the pinned merlin header, so it
  does not compile, and merlin is out of scope.
- **The spacemit board is not attached** — no USB-net interface, only the FPGA's
  FTDI JTAG/UART chain. The runtime does carry a `spacemit_x60` preset.
