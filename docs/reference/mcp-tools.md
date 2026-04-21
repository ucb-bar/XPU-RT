# MCP tools reference

Every tool the `compgen-mcp` stdio server exposes, grouped by pipeline
phase. Generated from `compgen.mcp.tools.ALL_TOOLS`; re-generate via
`compgen mcp tools`.

The client contract — name, description, input schema, phase — lives in
`python/compgen/mcp/tools/`. Each tool handler takes a `SessionManager`
plus keyword args and returns a JSON-serialisable dict.

## lifecycle (5)

Tools that open / close sessions, load models, and export bundles. You
typically call these first.

- **`open_target`** — Load a hardware-spec YAML and open a session.
- **`load_model`** — Run the deterministic pipeline + open an LLM-driven session.
- **`register_pack`** — Register a `compgen.packs` extension with the session (path or entry-point identifier). Rebuilds the device if a target is already open.
- **`compile`** — Run the full LLM-driven agentic loop on the session's model.
- **`bundle_export`** — Write the session's compiled bundle to disk.

## inspect (21)

Read-only tools: introspect recipe state, query knowledge, diff
checkpoints, diagnose model compatibility, scan vendor repos.

- **`checkpoint`** — Freeze the current Recipe IR view as a named checkpoint.
- **`diagnose_model_compatibility`** — Summarise which operators lack a registered Payload lowering, classify each, and recommend a recovery tool.
- **`diff_recipe`** — Diff the current Recipe IR against a prior checkpoint.
- **`explain_verification`** — Return the latest `apply_recipe`'s per-obligation + per-script failures with typed remediation hints + suggested next steps.
- **`get_context_brief`** — One-shot prompt-friendly brief of the most-relevant lessons for the given target × stage × topic × op_family.
- **`get_dossier`** — Return the deterministic graph-analysis dossier for this session.
- **`list_pending_bench_requests`** — List outstanding bench requests for the agent to fulfil.
- **`list_pending_dispatch_decisions`** — List outstanding dispatch decisions for the agent to fulfil.
- **`list_pending_kernel_requests`** — List all outstanding codegen requests in the session.
- **`list_phase_tools`** — List the tools + invent-slots registered in the LLM registry.
- **`lookup_bench_result`** — Cache lookup by (kernel × shape × dtype) fingerprint.
- **`lookup_cached_kernel`** — Check whether a kernel for this v3 contract is already cached.
- **`lookup_dispatch_decision`** — Check whether a dispatch decision for this region × target list is already cached.
- **`propose_vendor_spec`** — Re-run the vendor classifier with explicit vendor/package overrides.
- **`query_knowledge`** — Query the knowledge store for lessons applicable to a target × stage × op_family × topic.
- **`recovery_status`** — Report the session's accumulated recovery decisions.
- **`scan_vendor_repo`** — Scan a third-party MLIR dialect repo and propose a frozen `VendorDialectDescriptor` plus a lowering proposal.
- **`session_summary`** — Return the driver session summary (step index, hashes, counts).
- **`suggest_proposals`** — Return ranked candidate proposals for an invent slot.
- **`verify_vendor_package`** — Run the verification ladder against a scaffolded adapter package.
- **`view_recipe`** — Return a token-efficient view of the current Recipe IR.

## transform (21)

Tools that mutate session state: apply the recipe, record proposals,
fulfil codegen / dispatch / bench requests, install decomps or
translations, scaffold user-space vendor packages.

- **`apply_recipe`** — Lower the session's accumulated Recipe IR (including any agent-proposed `propose_*` ops) and apply the resulting transform scripts + kernel jobs + verification obligations.
- **`batch_propose`** — Submit a list of invent-slot proposals in one roundtrip. `atomic=True` rolls back recipe + payload on first rejection.
- **`invoke_tool`** — Invoke a registered LLM Tool by name.
- **`promote_in_session_authored_tools`** — Promote authored tools with `>= min_passes` in-session trials into the session's driver registry.
- **`propose_invent_slot`** — Submit a proposal to an invent-slot gate.
- **`record_lesson`** — Append a lesson to the hierarchical knowledge store.
- **`register_bench_result`** — Fulfil a pending bench request with the measured perf and correctness.
- **`register_blackbox`** — Mark an op target as an explicit opaque-boundary fallback.
- **`register_dispatch_decision`** — Fulfil a pending dispatch decision with a JSON verdict.
- **`register_kernel_result`** — Fulfil a pending codegen request with kernel source.
- **`register_optimization_progress`** — Agent posts a short summary of optimisation progress.
- **`request_dispatch_decision`** — Queue a HW-aware dispatch decision for the agent to fulfil.
- **`request_kernel_bench`** — Queue a kernel-bench request for the agent.
- **`request_kernel_codegen`** — Register a kernel-codegen request.
- **`request_model_optimization`** — Kick off / advance the agent optimisation loop.
- **`resolve_unsupported_op`** — Aggregator: pick `auto|decomp|translation|blackbox` and apply recovery for one unsupported op.
- **`scaffold_vendor_package`** — Render a pip-installable user-space adapter package from a reviewed `VendorDialectDescriptor`.
- **`step_proposal`** — Translate a typed LLM action proposal into an env step.
- **`synthesize_decomp`** — Install an ATen allow-list decomposition for the given op target.
- **`synthesize_translation`** — Wire a Payload-level external-call translation for the op target.
- **`verify_proposal`** — Run named gates directly on a proposal.

## job (1)

Long-running asynchronous tool calls.

- **`poll_job`** — Poll the status of a long-running asynchronous tool call.

## Related

- [MCP Setup](../getting-started/mcp-setup.md) — how to wire the server into Claude Code.
- [CLI Reference](cli.md) — see the "MCP subcommands" section.
- Source: `python/compgen/mcp/tools/`.
