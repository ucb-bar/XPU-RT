# Compilation Trace & IR Dumps

Every CompGen compile produces a **single correlated JSONL trace** plus
an **IREE-style sequence of IR dumps** so every decision the compiler
made — and every IR mutation that followed — is reconstructable from
disk after the fact. Nothing in the trace is written by the LLM; the
trace is emitted by the infrastructure surrounding it.

## Where the artifacts live

Given `compile_model(..., output_dir=<out>, session_id=<sid>)`:

```text
<out>/
  trace/
    trace.jsonl            # the unified event stream
  ir_dumps/
    NNNN_<name>_<phase>.mlir   # one per pass / stage checkpoint
    index.json                 # lookup: index, name, phase, ir_hash, duration_ms, trace_event_id
    final.mlir                 # the glued module after all stages
```

A symlink (or JSON pointer file when symlinks aren't available) at
`sessions/<sid>/trace.jsonl` mirrors the per-compile trace for
cross-session analysis.

## Event schema

Each line of `trace.jsonl` is one `TraceEvent`:

| Field             | Description                                                                                                                                                         |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `event_id`        | Monotonic `evt_NNNNNNNNNN` assigned by the bus                                                                                                                      |
| `parent_event_id` | The enclosing span's `event_id` (via `contextvars`), or `""`                                                                                                        |
| `session_id`      | Partitions events across concurrent compiles                                                                                                                        |
| `ts`              | ISO-8601 UTC timestamp                                                                                                                                              |
| `kind`            | `pass_run` / `stage_run` / `ir_dump` / `decision` / `decision_site` / `oracle_advisory` / `analysis_run` / `llm_prompt` / `llm_response` / `mcp_call` / `tool_call` |
| `phase`           | `start` / `end` / `point`                                                                                                                                           |
| `elapsed_ms`      | Real wall-clock ms between paired `start`/`end`; `0` for `point`                                                                                                    |
| `payload`         | Kind-specific structured dict                                                                                                                                       |

### Per-kind payload shapes (abbreviated)

* **`pass_run`** — `{name, source, ir_hash_before, ir_hash_after, stats, duration_ms, span_id}`
* **`stage_run`** — `{stage, target, llm_phase}` on `start`; `{span_id}` on `end`
* **`ir_dump`** — `{index, name, phase_tag, path, ir_hash, duration_ms, span_id}`
* **`decision_site`** — `{site_id, kind, context, candidate_ids, oracle_recommended_id}` — declared, nothing applied yet
* **`decision`** — `{decision_type, site_id, chosen, chosen_value, source, rationale, candidates, oracle_recommended_id, llm_turn_id}` — the applied outcome
* **`oracle_advisory`** — `{oracle, binding: false, ...oracle-specific fields}` — non-binding
* **`analysis_run`** — `{analysis, target, clusters, unclustered, opportunities}`
* **`llm_prompt`** / **`llm_response`** — include `prompt_hash`, token counts, latency; the response's `event_id` becomes the `llm_turn_id` tagged on subsequent `decision` events

### Correlation rules

* Every `ir_dump` event is parented to the `pass_run` or `stage_run`
  that produced it (via `parent_event_id`).
* Every `decision` event carries the `llm_turn_id` of the last
  `llm_response` seen in its context. That makes "prompt → decision
  → IR hash delta" fully traceable.
* `source` on a `decision` event distinguishes:
  * `"agent"` — an LLM picked via `apply_decision`
  * `"fallback_oracle"` — no agent pick; the oracle's recommendation was used
  * `"override"` — an agent replaced a resolved outcome
  * `"invent"` — a novel candidate submitted with `chosen_id="invent:..."`

## IR dump cadence

For each stage in `StageRegistry.run_pipeline`, we emit four dumps:

| Phase          | When                                        |
| -------------- | ------------------------------------------- |
| `entry`        | Before `verify_contract` on the input       |
| `shared_after` | After `shared_passes` (target-agnostic)     |
| `plugin_after` | After the target plugin's `transform`       |
| `exit`         | Before returning `StageResult`              |

Plus top-level spans for `capture_frontend`, `fx_to_xdsl`,
`ukernel_annotate`, and `eqsat` that run before the stage registry.
Each writes `before` / `after` IR dumps. Finally `compile_model`
writes `final.mlir` with the glued module.

Duration fields on `index.json` are measured in real time, not
placeholders. Zero-duration entries mark checkpoints (e.g. `entry`)
that don't do work themselves.

## Opting in

Default is off for production speed. Opt in via either:

* `compile_model(..., dump_ir=True)` Python kwarg
* Environment variable: `COMPGEN_DUMP_IR=1`

The trace bus is **always on** — it's cheap JSONL with one lock. Only
the IR `.mlir` writes gate on the dump flag.

## Hooks for readers

Code that wants to read the trace live should:

```python
from compgen.trace import get_active_bus
bus = get_active_bus()
print(bus.trace_path)   # <out>/trace/trace.jsonl
```

Post-hoc analysis is plain JSONL:

```bash
jq -c 'select(.kind == "decision" and .payload.source == "agent")' trace.jsonl
```

## Weaknesses to know about

* Pass-level granularity inside a `stage.shared_passes()` method isn't
  visible — each stage is one opaque mutation step from the trace's
  point of view. If you need finer detail, emit your own `pass_run`
  spans inside the stage.
* `oracle_advisory` events fire only when the oracle module paths are
  exercised (currently from `analysis/graph_digest.py` and the
  `stage_*_plugin` decision sites). If a consumer wires an oracle call
  elsewhere, it must use `compgen.trace.OraclePublisher.emit` to keep
  the advisory visible.
* Trace-event order is bus-write order, not wall-clock order (they
  usually match — the bus lock serialises writes — but don't assume
  strict causality without inspecting `parent_event_id`).
