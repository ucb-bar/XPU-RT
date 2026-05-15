# Extension points — reference

XPU-RT's user-extensible surfaces. One-line summaries here; see the linked
pages for the full contract.

## Discovery paths

| Path                                            | When to use                                                   |
|-------------------------------------------------|---------------------------------------------------------------|
| Entry-point plugins in an installed package     | Durable, versioned, shareable                                  |
| `~/.xpu_rt/extensions/*.py`                    | Drop-in files; no `pip install` needed; experimentation-friendly |
| Runtime `register()` calls                      | Tests, demos, in-process composition                            |

Trigger discovery: `xpu-rt ext list` — or implicitly on MCP server startup.

## Entry-point groups

Declared in `xpu_rt.plugins.KNOWN_GROUPS`. All groups are pre-registered in
XPU-RT's own `pyproject.toml` so `pip show xpu-rt` advertises them.

| Group name                          | Object contract                                                    | Registry / loader                             |
|-------------------------------------|--------------------------------------------------------------------|-----------------------------------------------|
| `xpu_rt.kernels.providers`         | `KernelProvider` protocol                                          | `xpu_rt.plugins.discover_all`                |
| `xpu_rt.transforms.decompositions` | Callable `(operands, meta, node_name) -> DecompResult`             | `xpu_rt.plugins.discover_all`                |
| `xpu_rt.kernels.fusion_rules`      | Callable `(producer_v3, consumer_v3) -> bool \| FusionVerdict`     | `xpu_rt.plugins.discover_all`                |
| `xpu_rt.targets.backends`          | `TargetBackendProtocol`                                            | `xpu_rt.plugins.discover_all`                |
| `xpu_rt.kernels.contracts`         | Callable returning a `KernelContractV3`                             | `xpu_rt.plugins.discover_all`                |
| `xpu_rt.vendor_dialects`           | Factory returning a `VendorDialectAdapter` subclass                | `xpu_rt.extensions.vendor_dialect.registry`  |

## User-space `~/.xpu_rt/extensions/`

- Default root: `~/.xpu_rt/extensions/`. Override with `XPU_RT_EXTENSIONS_DIR`.
- Disable entirely: `XPU_RT_DISABLE_LOCAL_EXTENSIONS=1`.
- Each `*.py` file may define `def register(registry): ...` or module-level
  `TOOL` / `TOOLS` / `SLOT` / `SLOTS` constants.
- Idempotent: `_state.json` tracks what's already been loaded.

## In-tree kernel providers

Not every provider ships as an entry-point package. Core XPU-RT ships
three in-tree implementations under `xpu_rt.kernels.providers`:

| Provider | Target | Invocation | Guide |
|----------|--------|------------|-------|
| `AutocompProvider` | GPU (Triton / CUDA) | In-process Python | — |
| `ExoProvider` | Accelerators (Gemmini, custom) | In-process Python | — |
| `KernelBlasterProvider` | CUDA | Subprocess (local shell or Docker) | [KernelBlaster Provider](../guides/kernelblaster.md) |

The agent loop registers them alongside any entry-point providers;
`xpu_rt.kernels.registry.ProviderRegistry` dispatches contracts in
registration order.

## Related docs

- [Authoring an extension](../getting-started/extension-authoring.md) — walkthrough
- [Architecture: extension points](../architecture/extension-points.md) — full protocols + examples
- [Architecture: target backend model](../architecture/target-backend-model.md) — for `xpu_rt.targets.backends`
- [Vendor dialects overview](../vendor_dialects.md) — for `xpu_rt.vendor_dialects`
- [KernelBlaster provider](../guides/kernelblaster.md) — subprocess-based kernel search
