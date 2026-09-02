# Workload specs

A spec in `data/toplevel/*.json` is the only place several facts live. Six of
its fields have produced wrong answers when misunderstood — what follows is
the failure mode for each, because the meaning is guessable and the failure
mode is not.

```bash
.venv/bin/python examples/workloads/anatomy_of_a_spec.py
```

## Shape

```json
{
  "_comment": "why this spec exists, and what it is measuring",
  "hardware": {
    "machines":    { "CPU_P": 4, "CPU_E": 4 },
    "profile_hw":  { "cpu_p": "rvv_x60", "cpu_e": "rvv_x60" },
    "profile":     { "gen_root": "gen_mb", "topo_tag_override": false }
  },
  "scheduler": { "enable_impls": false },
  "networks": {
    "dronet": {
      "id": 1,
      "dispatch_deps_path": "gen_mb/vmfb/.../dronet.int8_dispatch_graph.json",
      "period": 33.3,
      "window_duration": 33.3
    }
  },
  "edges": []
}
```

`networks` is a **mapping** of name → spec, not a list. Several sibling
formats in this repo use a list and the two look identical until you index one.

## `window_duration` — the deadline, and it is not the period

`trace_metrics.summarise_trace` uses `D = windows_ms.get(m, T)`. The spec's
`window_duration` **is** the deadline; it defaults to the period only when
omitted. Scoring without passing it scores against the wrong deadline, and the
score still looks fine.

`scripts/compare_candidates.py` and `xpu-rt/streaming_feedback.py` both take
`--windows-from` for exactly this.

## `period` — and the instance index that comes from the job name

Instance *k* is due at `k*T + D`. The index comes from the **job name**:
`dronet3` is instance 3.

Which is fine until a network name ends in a digit. `yolov8_nano_64x960` split
as `yolov8_nano_64x` + instance **960**, the deadline became ~48 s, and the
detector reported **zero deadline misses forever** — a structural zero that
reads exactly like a pass.

`xpu-rt/job_names.py` owns that split now, and it needs the **known network
names** to do it (longest match wins). Four independent copies of the splitter
existed before that module, each written after the previous one broke; three
now delegate to it and ModelBlaster's is pinned to agree by test.

## `dispatch_deps_path` — the graph

Points at ModelBlaster's emitted `*_dispatch_graph.json`. A rewrite (split,
fuse, unfuse) means pointing this at the **rewritten** graph — and then the
candidate and baseline are only comparable if the instance counts still match.

Not hypothetical: a refinement loop once grew `mlp_control` from 32 instances
to 91, the file landed on disk under the baseline's name, `pdb_hash` still
differed, every term still computed, and the figure reported the opposite
verdict for the DroNet ×2 rung. `compare_candidates.py` checks instance counts
before comparing any term because of it.

The directory is called `vmfb/` for historical reasons — it holds dispatch
graphs, not VMFBs. Renaming it would touch ~20 specs to change nothing.

## `gen_root` — which profile tree

`gen_mb` is ModelBlaster's measured tree; `gen` is the retired IREE one.
Mixing them compares timings from two different runtimes.

`gen_mb/profile` is a **symlink** to `gen/profile_mb`. Worth knowing because
`find` does not follow symlinks by default and will report the tree as empty.

## `topo_tag_override` — may the solver pick a core width?

`false` is load-bearing for shard-mode solves: it lets the solver read the
per-width profiles (`topo_0`, `topo_0_1`, `topo_0_1_2_3`,
`topo_0_1_2_3_4_5_6_7`) and choose per dispatch, instead of being handed one
width for the whole model.

Measured per-dispatch scaling varies **4.8× within one model** — 4.02× on a
wide-OC conv down to 0.83× on a 1×1 — so one width per model is wrong for
nearly every dispatch in it.

The selected composite target is consumed end to end by ModelBlaster; see
[`ModelBlaster/docs/xpurt_schedule_sharding.md`](../ModelBlaster/docs/xpurt_schedule_sharding.md)
for codegen, weight packing, exact-hart pools, locking, and refusal rules.

Those tags are produced by `MB_CORES` in `run_model_k1.sh`, which derives the
pool width, the affinity mask, the binary suffix and the tag together, so a
profile cannot claim a core count it did not run on.

## `scheduler.enable_impls` — may the solver pick an implementation?

With it on, every core-group combination is emitted once per legal
implementation and each dispatch records the winner as `impl`. That is how one
core runs a MAC-unit GEMM and then a vector one: `hardware_target` names
**where**, `impl` names **with what**.

The binary honours it — `ingest_xpurt_schedule.py` reads `impl` and the walker
selects its dispatch table on it, failing loudly if asked for an
implementation the build lacks. Before that it selected on `core_kind`, so a
heterogeneous schedule produced a binary that quietly ran one implementation
everywhere and reported the runtime it got.

## `hardware.machines` on the K1

`{"cpu_p": 8}` — the runbook's own recommended config — **SIGILLs under IME**.
The board has two 4-core L2 clusters; `smt.vmadot` executes on harts 0–3 and
traps on 4–7, and an 8-wide CPU_P maps `CPU_P#4..7` onto cluster 1.

Use `{"cpu_p": 4, "cpu_e": 4}` for anything with IME, and let
`scripts/check_schedule_feasibility.py` refuse the schedule before deployment.
Unlike every other finding it reports, an illegal implementation placement does
not make the run slow — it produces no output at all.

## Writing a new one

Start from the closest existing spec, keep `_comment` honest about what the
spec is measuring, and change things **in the spec** rather than on the command
line. The spec is what gets committed beside a result; a flag is not, and a
figure whose spec does not reproduce it is a figure nobody can check.

## Related

* [`solvers.md`](solvers.md) — what consumes this
* [`the_loop.md`](the_loop.md) — where it sits in the cycle
* `data/toplevel/networks_k1_multicore_shard.json` — a worked shard-mode spec
