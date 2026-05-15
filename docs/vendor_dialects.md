# Vendor MLIR Dialect Integration

XPU-RT supports integrating third-party MLIR dialects (e.g. NVIDIA
CUDA Tile IR, Qualcomm Hexagon MLIR) as backends through a **user-space
adapter package**. A dedicated integration agent, driven by Claude Code
through XPU-RT's existing MCP server, explores a vendor repo and
scaffolds the adapter.

## Architecture

```
┌─ XPU-RT core (general) ───────────────────────────────────────────────┐
│                                                                        │
│  extensions/vendor_dialect/            protocol + scanner + scaffold   │
│  agent/vendor_integration/             LLM-driven explore/propose loop │
│  mcp/tools/vendor_dialect.py           4 MCP tools on existing server  │
│  kernels/providers/claude_kernel.py    generic Claude-backed provider  │
│  api.py :: compile_with_vendor         end-to-end entry point          │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
             │ (entry point `xpu_rt.vendor_dialects`)
             ▼
┌─ User-space (per-vendor, scaffolded) ─────────────────────────────────┐
│                                                                        │
│  xpu_rt_cuda_tile/                                                    │
│    adapter.py, lowering.py, kernels.py, bundle.py, templates/          │
│  xpu_rt_hexagon/                                                      │
│    adapter.py, lowering.py, bundle.py                                  │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
```

## Workflow

Four MCP tools, called in order:

| Tool                       | What it does                                                |
|----------------------------|-------------------------------------------------------------|
| `scan_vendor_repo`         | Deterministic scan + LLM classification → descriptor YAML   |
| `propose_vendor_spec`      | Re-run with explicit vendor/package overrides               |
| `scaffold_vendor_package`  | Render pip-installable adapter from approved descriptor     |
| `verify_vendor_package`    | Structural → matmul diff → workload diff verification ladder|

The agent loop is:

1. **Scan** — `xpu_rt.extensions.vendor_dialect.scanner.scan_repo` walks
   the repo and collects README / CMake / TableGen ops / CLI tools /
   Python bindings. Deterministic, no LLM.
2. **Classify** — `xpu_rt.agent.vendor_integration.explore.explore_vendor_repo`
   feeds the scan into a prompt and parses a JSON classification. Falls
   back to a conservative deterministic classifier when no LLM is
   available.
3. **Review gate** — the descriptor YAML is returned to the user;
   editing is expected before scaffolding.
4. **Scaffold** — `xpu_rt.extensions.vendor_dialect.scaffold.scaffold_package`
   renders the user-space package from a Jinja2 template pack. The
   package declares a `xpu_rt.vendor_dialects` entry point so that
   `pip install -e .` makes the adapter visible process-wide.
5. **Verify** — `xpu_rt.extensions.vendor_dialect.verify.verify_package`
   runs the verification ladder declared in the descriptor.

## Kernel authoring

When a vendor has no direct linalg/tosa/stablehlo ingress (e.g. CUDA
Tile IR), the adapter attaches a
`xpu_rt.kernels.providers.claude_kernel.ClaudeKernelProvider`. That
provider takes a dialect-specific `PromptPack`, invokes
`autocomp.common.llm_utils.LLMClient` (reused per the repo rule against
duplicating LLM plumbing), and runs a bounded retry loop with a
caller-supplied structural gate.

## Key types

| Type                        | Location                                                        |
|-----------------------------|-----------------------------------------------------------------|
| `VendorDialectDescriptor`   | `xpu_rt.extensions.vendor_dialect.descriptor`                  |
| `VendorDialectAdapter`      | `xpu_rt.extensions.vendor_dialect.adapter`                     |
| `ScanResult`                | `xpu_rt.extensions.vendor_dialect.scanner`                     |
| `ScaffoldResult`            | `xpu_rt.extensions.vendor_dialect.scaffold`                    |
| `VerificationReport`        | `xpu_rt.extensions.vendor_dialect.verify`                      |
| `ExploreResult`             | `xpu_rt.agent.vendor_integration.explore`                      |
| `ProposedAdapter`           | `xpu_rt.agent.vendor_integration.propose_adapter`              |
| `ClaudeKernelProvider`      | `xpu_rt.kernels.providers.claude_kernel`                       |

## Authoring an adapter by hand

The scaffold pack is a starting point, not a straitjacket. A
hand-written adapter just needs to subclass `VendorDialectAdapter`
and implement `lower_payload` + `emit_artifact`, then register itself
via `register_adapter()` or a `xpu_rt.vendor_dialects` entry point.
