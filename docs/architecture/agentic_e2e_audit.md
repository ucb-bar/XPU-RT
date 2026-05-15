# Agentic E2E Audit (post-W7)

How much of the CompGen compile loop is *actually* driven by the
agent (Claude Code) over MCP, vs still living in Python-only paths.

The litmus test for "agentic": can a Claude Code session, talking
exclusively over MCP, *(a)* observe the current compile state, *(b)*
make every load-bearing decision, *(c)* fulfil the work, *(d)* persist
what it learned for the next session?

## What IS now agent-driven via MCP

| Concern | MCP tools (request → register → lookup → list) | Notes |
|---|---|---|
| Lifecycle (open target, load model, sessions) | `open_target`, `load_model`, `close`, `register_pack`, … | W0 |
| Model + IR inspection | `inspect_*` family (graph breaks, payload dump, recipe view) | W0 |
| Diagnosis (model compatibility, exported program) | `diagnose_model_compatibility`, `diagnose_exported_program` | W0 |
| Recovery from blackbox ops | `recovery_*` (register_blackbox, suggest_recovery, …) | W0 |
| Recipe-level transforms | `transform_*`, `apply_recipe` | W0 |
| Suggestion engines (fusion / megakernel candidates) | `suggest_*` | W0 |
| Vendor / NPU dialects | `vendor_dialect_*` | W0 |
| **Per-kernel codegen** | `request_kernel_codegen` / `register_kernel_result` / `lookup_cached_kernel` / `list_pending_kernel_requests` | W4 |
| **HW-aware dispatch decision** (granularity + best target) | `request_dispatch_decision` / `register_dispatch_decision` / `lookup_dispatch_decision` / `list_pending_dispatch_decisions` | **W6.4** |
| **Per-kernel bench** (perf + correctness) | `request_kernel_bench` / `register_bench_result` / `lookup_bench_result` / `list_pending_bench_requests` | **W7.1** |
| **Cross-session knowledge** (lessons + briefs) | `record_lesson` / `query_knowledge` / `get_context_brief` | **W7.2** |
| **End-to-end optimisation tracking** | `request_model_optimization` / `register_optimization_progress` | **W7.3** |
| **Refinement loop** (agent decides when to stop) | `request_refinement` / `register_refinement_attempt` / `lookup_refinement_history` / `list_pending_refinements` | **W8.2** |
| **Per-shape autotune trials** (gated + persisted) | `request_autotune_trial` / `register_autotune_pick` / `lookup_autotune_pick` / `list_pending_autotune_trials` | **W8.3** |
| **`compile_with_llm` opt-in MCP path** (`mcp_session=…` arg) | `mcp_optimized` field on `LLMCompileResult` | **W8.1** |
| Pattern graduation (agent-authored tools → registry) | `graduate_*` | W3 |

The W7-shipped pieces close the four prior seams:

* **Bench** — was Python-only via `bench/kernel_bench.run_microbench`;
  now `McpBenchFn` slot in the optimizer round-trips through the
  bench MCP tools, with KernelDB rehydration on first lookup so a
  bench measured in any prior session hits cache transparently.
* **Knowledge** — was Python-only via `KnowledgeStore.add` / `.query`;
  now exposed as MCP tools so the agent can record perf/correctness/
  recipe lessons + query narrowly by `target × stage × op_family × topic`.
* **Optimisation loop** — was Python-only via
  `agent.kernel_optimizer.optimize_model`; now wrapped by
  `agent.mcp_optimizer.optimize_via_mcp` which pre-wires
  `McpDispatchLLM` + `McpCodegenFn` + `McpBenchFn` so every callback
  is an MCP round-trip. Plus `request_model_optimization` is the
  single tool the agent calls to advance / monitor the loop.
* **Default codegen for `optimize_model`** — was the identity stub;
  callers using `optimize_via_mcp` get the MCP-backed codegen for
  free.

## What is still NOT fully agent-driven (post-W8)

These are not *missing*, just not first-class through MCP yet — the
agent has to drop into Python (or a pre-existing tool) for them.

| Concern | Today | Why it's deferred |
|---|---|---|
| FX import / payload IR construction | Python (`capture/`, `ir/payload/import_fx.py`); agent triggers it via `load_model` | The IR build is deterministic; no LLM judgment needed |
| Recipe → MLIR transform-script lowering | Python (`ir/recipe/lower_megakernel*.py`); deterministic | No LLM-loop value-add |
| Pattern graduation orchestration | Python (`agent/self_extension/graduate.py`); agent surfaces results via `graduate_*` | Existing surface is sufficient for the use case today |
| Plugin discovery via entry points | Python (`compgen/plugins/`) — discovery is install-time, not runtime | Not a per-compile decision |
| Automatic contract extraction from `CompiledModel` | `compile_with_llm(mcp_session=…, mcp_contracts=…)` requires the caller to supply contracts | Real model→region→contract conversion is W9+ work; the W8 opt-in path runs the optimiser when contracts are supplied externally |

## How the agentic loop runs in practice (W7)

A Claude Code session driving CompGen through MCP looks like:

```
1. open_target(profile=…)                          # lifecycle
2. load_model(model=…)                             # lifecycle → builds contracts
3. request_model_optimization(                     # W7.3
       contract_fingerprints=[fp1, fp2, …]
   )                                              → returns pending counts
4. While pending_dispatch > 0:
       list_pending_dispatch_decisions
       (analyse spec + region; emit JSON verdict)
       register_dispatch_decision
5. While pending_codegen > 0:
       list_pending_kernel_requests
       (write the kernel from the prompt's KernelFacingView)
       register_kernel_result
6. While pending_bench > 0:
       list_pending_bench_requests
       (run the bench, report perf + correct + notes)
       register_bench_result
7. record_lesson(category=…, summary=…, target=…) # W7.2
8. register_optimization_progress(summary="pass N done")
9. Loop back to (3) until every contract has cached kernel + perf
```

Steps 3-9 use only MCP tools. No Python code is invoked by the agent
between steps. KernelDB + KernelStore + KnowledgeStore writes happen
inside the register tools, so kernels generated and lessons learned
in this session are visible to every future session and to headless
`compile_with_llm` invocations.

## Post-W8 status

After W8 every load-bearing compile decision is an MCP round-trip:

* W7.1 — bench
* W7.2 — knowledge
* W7.3 — agentic optimiser glue + progress tracking
* **W8.1** — `compile_with_llm(mcp_session=(sm, sid), mcp_contracts=…)`
  runs the W7 loop on top of the standard pipeline; result lands in
  `LLMCompileResult.mcp_optimized`.
* **W8.2** — refinement loop is MCP-driven; agent decides when to
  set `done=True` for a kernel.
* **W8.3** — autotune trials route through `request_autotune_trial`
  with disk persistence to `~/.compgen/autotune/`.

The remaining Python-only work is purely deterministic infrastructure
(IR construction, MLIR lowering, plugin discovery) where there's no
LLM judgment to add. *The agentic-completeness story is done.*

What a future W9 might add:

1. Automatic contract extraction from `CompiledModel` so callers don't
   have to pass `mcp_contracts=…` explicitly.
2. Real e2e perf validation through `optimize_via_mcp` on TinyLlama
   / SmolVLA / Gemma 2B (the W6 acceptance criteria).
3. Production hardening: error recovery, observability dashboards,
   plugin ecosystem.
