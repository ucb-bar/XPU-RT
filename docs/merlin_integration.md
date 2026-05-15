# Merlin integration

XPU-RT and Merlin (`/scratch2/agustin/merlin`, ground-truth — not the
embedded submodule) collaborate through three artifact channels and one
MCP tool. The integration is **strictly additive**: with no feedback
artifacts present, Merlin's standalone path is byte-identical to today.

## Artifact channels

```
                 ┌──────────────── XPU-RT ─────────────────┐
                 │                                          │
profiled_manifest.json ──▶ scripts/merlin_adapter.py ──▶ schedule.json
                 │              ↓ --emit-feedback           │
                 │           xpurt_feedback.json            │
                 │                ↓                          │
                 │      ingest_xpurt_feedback (MCP)         │
                 │                ↓                          │
                 │   <merlin_dir>/breakdowns/feedback.json  │
                 │                                          │
                 ▼                                          ▼
         scheduler_runner.cc                    tools/compile_qnn.py
         (runs the schedule, emits          tools/compile.py --with-feedback
          telemetry JSON-Lines and          benchmarks/SpacemiTX60/...
          watches schedule_next.json)       (read feedback.json,
                 │                           bias granularity, inert
                 ▼                           if absent)
         telemetry JSON-Lines stream
                 │
                 ▼
         xpu-rt/streaming_feedback.py
         (windowed → ingest_xpurt_feedback every N epochs)
```

| Artifact                            | Producer                            | Consumer                                  | Optional? |
| ----------------------------------- | ----------------------------------- | ----------------------------------------- | --------- |
| `breakdowns/profiled_manifest.json` | `third_party/merlin/tools/board_roundtrip.py`   | `scripts/merlin_adapter.py`               | No        |
| `breakdowns/schedule.json`          | `scripts/merlin_adapter.py`         | `merlin compile --with-schedule`, runner  | No        |
| `breakdowns/xpurt_feedback.json`    | `scripts/merlin_adapter.py --emit-feedback` | `ingest_xpurt_feedback` MCP tool   | **Yes**   |
| `breakdowns/feedback.json`          | `ingest_xpurt_feedback` MCP tool    | `tools/compile_qnn.py`, `tools/compile.py --with-feedback`, SpaceMit scripts | **Yes**   |
| `<run>_telemetry.jsonl`             | `scheduler_runner.cc` TelemetrySink | `xpu-rt/streaming_feedback.py`            | **Yes**   |
| `schedule_next.json`                | host-side scheduler / re-solver     | `scheduler_runner.cc` ScheduleHotSwap     | **Yes**   |

The four optional artifacts form the feedback loop. If none are present,
behavior matches the pre-integration baseline.

## Feedback schema (`xpurt_feedback.json` and persisted `feedback.json`)

```json
{
  "schema_version": 1,
  "run_id": "string — stable across iterations to accumulate hints",
  "source_schedule": "schedule.json",
  "model_signals": {
    "makespan": 12345.6,
    "makespan_efficiency": 0.71,
    "deadline_met": true,
    "skip_triggered": ["dispatch_id_1", "dispatch_id_2"],
    "problem_status": "optimal",
    "fusion_applied": false
  },
  "dispatches": {
    "<dispatch_id>": {
      "current_target": "CPU_P",
      "idle_fraction": 0.42,
      "transfer_cost_ratio": 1.8,
      "deadline_slack_us": 2100,
      "hints": ["prefer_finer", "consider_fuse_with_pred"],
      "rationale": "long idle gap with cross-cluster transfer penalty 1.8x base"
    }
  }
}
```

### Hint vocabulary (closed set)

| Hint                       | Meaning                                                                |
| -------------------------- | ---------------------------------------------------------------------- |
| `prefer_coarser`           | Slack available; reduce overhead (larger tiles / fewer chunks).        |
| `prefer_finer`             | Schedule has unexploited parallelism; smaller tiles / more chunks.     |
| `consider_fuse_with_pred`  | Cross-cluster transfer dominates duration; merge with predecessor.     |
| `pin_target=<machine>`     | Keep this dispatch on `<machine>` — much faster there or stable there. |
| `consider_split_backend`   | Current backend is the slow side; let the compiler re-evaluate.        |

The vocabulary is enforced by the `ingest_xpurt_feedback` MCP tool —
unknown hints are rejected with a `ToolError`. Per-target translation
(e.g. `pin_target=qnn-hta` for QNN, `pin_target=CPU_P` for SpaceMit
schedules) lives in the consumer.

## Three modes

| Mode       | Cadence              | Driver                              | Use case                  |
| ---------- | -------------------- | ----------------------------------- | ------------------------- |
| One-shot   | once per compile     | `scripts/merlin_adapter.py`         | Manual experiments        |
| Batch loop | once per iter        | `third_party/merlin/tools/run_full_loop.py`     | Convergence (existing)    |
| Streaming  | per-epoch on-board   | `scheduler_runner.cc` + `streaming_feedback.py` | HW-in-the-loop / robotics |

All three speak the same schema. What changes is *who emits when* — the
`ingest_xpurt_feedback` MCP tool's merge semantics (set-union per
`run_id`) make repeated incremental posts safe.

### One-shot

```sh
python scripts/merlin_adapter.py schedule \
    /scratch2/agustin/merlin/eval/qrb5165/dronet \
    --solver greedy --emit-feedback
# -> .../breakdowns/xpurt_feedback.json
# Operator (or downstream tooling) calls the MCP tool to persist.
```

### Batch loop

```sh
python third_party/merlin/tools/run_full_loop.py \
    --merlin-dir eval/qrb5165/dronet \
    --remote-vmfb-dir /root/iree_run/dronet/breakdowns \
    --solver greedy --iters 3 --converge-on-stable-hints
```

`run_full_loop.py` already orchestrates profile → schedule → run → fold
→ recompile. The integration adds:
- `schedule_once` calls `merlin_adapter.py --emit-feedback`
- `ingest_feedback` step persists `breakdowns/feedback.json` before each
  on-board run (so any sidecar compile sees fresh hints)
- Optional `--converge-on-stable-hints` early-exit when the hint set
  stops changing between iterations

### Streaming (HW-in-the-loop)

```sh
python third_party/merlin/tools/run_full_loop.py \
    --merlin-dir eval/qrb5165/dronet \
    --remote-vmfb-dir /root/iree_run/dronet/breakdowns \
    --solver greedy --iters 1 --stream-epochs 64
```

When `--stream-epochs N > 0`:
- Adds `--telemetry_jsonl=<remote_path>` to the on-board scheduler_runner
- Spawns a local `ssh tail -f` to pull the JSON-Lines stream into a
  local file
- Spawns `xpu-rt/streaming_feedback.py --follow` which windows the
  stream (rolling N epochs) and posts incremental feedback to the
  ingest tool every `N/4` epochs
- Cleans both up after the on-board run completes

The runtime side (`samples/common/xpu-rt/scheduler_runner.cc`) has two
HW-in-the-loop affordances:

1. **TelemetrySink** — per-dispatch JSON-Lines emission, gated on
   `cfg->telemetry_jsonl_path` or `cfg->telemetry_fd > 0`. Inert
   otherwise (zero overhead vs the prior baseline).
2. **ScheduleHotSwap** — watches `cfg->schedule_next_path` mtime, parses
   a new schedule.json at each graph iteration boundary, and atomically
   swaps the safe-to-mutate fields (`start_time_ms`, `hardware_target`,
   `deadline_ms`, `skipped`) on the live `model.nodes`. Adding/removing
   dispatches still requires a recompile.

This split is intentional: bidirectional feedback is continuous, but the
artifacts that change online are bounded to what the runtime can apply
safely. Compile-time decisions (chunk boundaries, ukernel choice) still
go through `run_full_loop.py`'s offline iteration.

## Backend wiring

Each backend reads `<merlin_dir>/breakdowns/feedback.json` if present
via the shared `third_party/merlin/tools/feedback_overlay.py:load_feedback_overlay()`
helper. Behavior is identical to today when the file is absent.

### QNN (`third_party/merlin/tools/compile_qnn.py`)

- `pin_target=qnn-cpu|qnn-gpu|qnn-hta` → restricts the per-chunk
  `--backends` list to that single backend.
- Other hints (`consider_split_backend`, `consider_fuse_with_pred`,
  `prefer_finer`, `prefer_coarser`) are surfaced as advisory in the
  compile log and recorded in `compile_qnn_summary.json`. Acting on
  them requires a re-run of `tools/chunk_extractor.py` upstream.

### SpaceMit X60 (`third_party/merlin/tools/compile.py --with-feedback`)

- `compile.py` accepts `--with-feedback <path>` (analogous to
  `--with-schedule`). When set:
  - Loads the overlay
  - Logs hint counts and a model-level disposition (`finer` /
    `coarser` / `neutral` based on majority hint type)
  - Writes `<output_dir>/feedback_applied.json` so target-specific
    follow-up scripts can act on it
- `benchmarks/SpacemiTX60/compile_matmul_xsmt_i8_ukernel_all.sh`
  passes `--with-feedback` automatically when `MERLIN_DIR` is set and
  `<MERLIN_DIR>/breakdowns/feedback.json` exists. Without that env var
  the script behaves as before.
- IREE's tile / ukernel selection lives in target specs and pass
  config; silent flag injection from the overlay would break the
  additive-only invariant. The disposition is therefore a *signal* to
  downstream tooling, not an automatic flag rewrite.

## Verification

End-to-end smoke checklist (run from `/scratch2/agustin/XPU-RT`):

1. **One-shot, schema-only:** `python scripts/merlin_adapter.py schedule
   <merlin_dir> --solver greedy --emit-feedback` produces
   `xpurt_feedback.json` next to `schedule.json` with at least one
   `dispatches.*.hints` entry on a known-bottlenecked workload.

2. **MCP ingestion + merge semantics:** call the tool twice with the
   same `run_id` and verify `dispatches.*.hints` accumulates rather
   than replaces. Calling with a different `run_id` replaces the file
   wholesale. (See `xpu-rt/tests/test_feedback_derivation.py` for the
   unit tests that gate the derivation logic; manual smoke included
   in this doc's commit message.)

3. **QNN convergence:** `third_party/merlin/tools/run_full_loop.py --merlin-dir
   eval/qrb5165/dronet --iters 3 --converge-on-stable-hints`. Expect
   makespan to decrease across iterations and the loop to exit early
   when the hint set stabilises.

4. **SpaceMit convergence:** same with
   `--merlin-dir benchmarks/SpacemiTX60/<workload>` plus the matmul
   ukernel compile script (which auto-detects `MERLIN_DIR`).

5. **Streaming smoke:** add `--stream-epochs 64` to `run_full_loop.py`.
   Confirm `<merlin_dir>/telemetry_iter1.jsonl` accumulates events,
   `streaming_feedback.py` emits at least one POST during the run, and
   `breakdowns/feedback.json` shows merged hints from both the offline
   and streaming paths.

6. **Standalone-runtime invariant:** build `scheduler_runner` without
   `--telemetry_jsonl` and `--schedule_next` flags; `TelemetrySink::Open`
   stays inactive and `ScheduleHotSwap::active()` returns false. Zero
   overhead vs the prior baseline.

7. **Standalone-Merlin invariant:** with no `xpurt_feedback.json` or
   `feedback.json` anywhere, all integration consumers fall through to
   their pre-integration paths. (`tools/feedback_overlay.py`
   `load_feedback_overlay()` returns an empty overlay object whose
   `for_dispatch()` reports no opinion for every dispatch.)

## Hint stability and convergence

The two-tier convergence story:

- **Online (per-epoch, fast):** `streaming_feedback.py` accumulates
  hints into the same `run_id` as the windowed telemetry rolls. The
  `scheduler_runner.cc` ScheduleHotSwap allows the host to push a new
  schedule mid-run when a re-solve is warranted (e.g., a change in
  hardware target). No Merlin involvement needed.
- **Offline (per-iter, slow):** when `run_full_loop.py` detects that
  the offline hint set is stable across two iterations
  (`--converge-on-stable-hints`), it exits — the workload is no longer
  asking for a recompile. Otherwise the loop continues, picking up the
  accumulated hints into the next compile.

Compile-time decisions (chunk boundaries, ukernel choice, tile size)
require a recompile and live in the offline tier. The online tier is
limited to schedule-table mutations (target/release-time/deadline/skip)
that the runtime can apply safely between epochs.
