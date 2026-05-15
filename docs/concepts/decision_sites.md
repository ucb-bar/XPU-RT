# Decision Sites — the Agent's Write Path

## Why this exists

Every compiler has to make choices: which encoding, which tile, which
fusion, which memory tier, which kernel backend. IREE, XLA, and
traditional heuristic compilers hardwire those choices into stage
plugins. CompGen rejects that premise — heuristics can't keep up with
new hardware, new model shapes, or novel op combinations.

The **decision site** is CompGen's primary architectural inversion:
every choice the compiler used to make silently is now a first-class
object the agent can read, propose against, and override — via MCP —
before the pipeline writes anything to the IR.

## The contract

```
DecisionSite
  ├─ site_id                 # stable, e.g. "simt.encoding:matmul_0"
  ├─ kind                    # encoding | tile | fusion | granularity | kernel_choice | ...
  ├─ context                 # shapes, dtypes, is_matmul, envelope fields
  ├─ candidates              # tuple[DecisionCandidate]
  ├─ oracle_recommended_id   # NON-BINDING
  ├─ status                  # pending | resolved | overridden
  └─ outcome                 # DecisionOutcome once resolved

DecisionCandidate
  ├─ id                      # unique within the site
  ├─ value                   # applied to the IR when this candidate wins
  ├─ source                  # "oracle:fusion" | "oracle:tile" | "cost_model" | "invent" | ...
  ├─ oracle_verdict          # "recommended" | "allowed" | "discouraged"
  ├─ oracle_reason           # short justification from the oracle
  ├─ oracle_confidence       # 0..1
  ├─ cost_breakdown          # {dram_savings_us, launch_savings_us, ...}
  ├─ knowledge_brief         # markdown excerpt from memory/knowledge
  └─ evidence                # free-form: autotune history, prior wins

DecisionOutcome
  ├─ site_id
  ├─ chosen_id
  ├─ chosen_value            # backfilled from candidate if the agent didn't pass one
  ├─ source                  # agent | fallback_oracle | override | invent
  ├─ rationale
  ├─ llm_turn_id             # cross-reference to the llm_response event
  └─ decision_event_id       # the trace event id for this outcome
```

## Resolution order

When a stage plugin reaches a choice point, it calls:

```python
registry.enqueue(site)         # emits a decision_site trace event
outcome = registry.resolve(site_id)
```

`resolve()` checks three things, in order:

1. **Agent pre-apply** — has `apply_decision` already been called
   (with matching `site_id`)? If so, use that outcome, emit a
   `decision(source="agent")` event, done.
2. **Oracle fallback** — use `site.oracle_recommended_id`, emit a
   `decision(source="fallback_oracle")` event.
3. **First-candidate fallback** — if the oracle gave no
   recommendation, pick the first candidate. Same event, different
   rationale.

The agent can still **override** a resolved site post-hoc via
`override_decision`, which emits a `decision(source="override")`
event and mutates the outcome. Downstream stages that haven't run
yet will read the overridden value.

## Why the oracle is now "one voice"

Oracles (`fusion_oracle.should_fuse`, `tile_oracle.recommend_tile`,
`granularity_oracle.recommend_granularity`) still exist and are still
used — they seed the `oracle_recommended_id` on each site and they
populate candidate `oracle_verdict` / `oracle_reason` / `cost_breakdown`
fields. What changed:

* **Their output is non-binding.** A `FUSE` verdict doesn't cause a
  fusion — it just flags the candidate. The agent can pick `DONT_FUSE`
  with its own rationale.
* **They advertise themselves.** Every oracle call emits an
  `oracle_advisory` trace event with `binding: false`.
* **Their authored constants are visible.** `_LAUNCH_OVERHEAD_US`
  table and the `ratio = 1 + net_us / per_launch` formula used by
  `fusion_oracle` are cost-model outputs, not measurements. The
  trace records them so reviewers can correct them against real
  bench results later.

## How the agent drives it

Four MCP tools:

| Tool | Purpose |
|------|---------|
| `list_decisions` | Enumerate sites with status, candidates, oracle rec |
| `propose_decision` | Record a non-binding proposal (trace event only) |
| `apply_decision` | Commit a pick; usable **before** the site enqueues |
| `override_decision` | Replace an already-resolved outcome |

All four are trace-emitting — every action the agent takes lands as a
`decision` event with `source="agent"` and an `llm_turn_id` linking
back to the LLM turn that drove it.

### Typical flow

```python
# The agent lists sites (may be empty if compile hasn't started).
list_decisions(session_id=...)

# The agent reasons and pre-applies picks BEFORE load_model.
apply_decision(
    session_id=...,
    site_id="simt.encoding:matmul_0",
    chosen_id="row_major",
    rationale="Block size mismatch; row_major avoids a gather on this target.",
)

# load_model triggers compile_model, which walks the stage pipeline.
# The stage plugin enqueues matmul_0's site and calls registry.resolve(),
# which drains the pre-applied outcome → IR gets "row_major".
load_model(session_id=..., model_path=...)

# After compile, check what actually got used.
list_decisions(session_id=..., status="resolved")   # sources: agent vs fallback_oracle
```

### Novel values (invent)

`chosen_id="invent:<slug>"` lets the agent submit a value no
oracle enumerated. `chosen_value` must be supplied. Example:

```python
apply_decision(
    session_id=...,
    site_id="simt.encoding:matmul_0",
    chosen_id="invent:packed_nt_stride8",
    chosen_value="packed_nt_stride8",
    rationale="Target has no MMA; packed N-T layout + stride-8 avoids a gather."
)
```

The trace records `source="invent"`. Verification / bench still gates
whether the invented value actually ships — invention is not a free
pass.

## Plugins that have been inverted

| Plugin | File | Sites emitted |
|--------|------|---------------|
| `GpuEncodingPlugin` | `targetgen/families/simt_gpu_hal.py` | `simt.encoding:<region>` per tensor-producing op |
| `CudaEncodingPlugin` | `stages/targets/cuda_gpu.py` | `cuda.encoding:<region>` per tensor-producing op |
| `CudaTilingPlugin` | `stages/targets/cuda_gpu.py` | `cuda.tiling:<region>` per matmul / linalg op |

Other plugins still write attributes directly; they can be inverted
using the same pattern — call `get_active_registry()`, build a
`DecisionSite`, `enqueue`/`resolve`, write the outcome's
`chosen_value` to the IR.

## Invariants to hold

* **A stage must not mutate the IR between `enqueue` and `resolve`**
  for a given site. `resolve` is the only commitment point.
* **Site ids must be stable across compiles of the same model** —
  use `compgen.region_id` from the importer, not raw `id(op)`.
* **Oracle advisories never gate control flow** — they're recorded
  for observability and agent context; they must not change IR by
  themselves.
* **The registry is per-session**, held on `McpSession.decision_registry`
  and also installed into `compgen.agent.decisions._active_registry`
  via `install_registry`. Both channels exist because MCP dispatches
  run on a thread-pool and `ContextVar` values don't cross thread
  boundaries by default; the process-level fallback in
  `install_registry` handles that.

## Open questions (non-goals for this iteration)

* **Dry-run mode.** Right now sites enqueue during the real compile;
  a future pass would run a dry compile that surfaces every site for
  the agent to reason over BEFORE any IR is produced.
* **Multi-turn negotiation.** An agent today can `propose_decision`
  multiple times and then `apply_decision` once; it can't revoke a
  proposal. A future tool could support `retract_proposal`.
* **Per-site bench history.** An autotune subsystem should associate
  measured wins with `site_id + chosen_id` so the knowledge store can
  learn which agent picks beat the oracle.
