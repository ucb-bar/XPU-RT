# Extension points — reference

CompGen's user-extensible surfaces. One-line summaries here; see the linked
pages for the full contract.

## Discovery paths

| Path                                            | When to use                                                   |
|-------------------------------------------------|---------------------------------------------------------------|
| Entry-point plugins in an installed package     | Durable, versioned, shareable                                  |
| `~/.compgen/extensions/*.py`                    | Drop-in files; no `pip install` needed; experimentation-friendly |
| Runtime `register()` calls                      | Tests, demos, in-process composition                            |

Trigger discovery: `compgen ext list` — or implicitly on MCP server startup.

## Entry-point groups

Declared in `compgen.plugins.KNOWN_GROUPS`. All groups are pre-registered in
CompGen's own `pyproject.toml` so `pip show compgen` advertises them.

| Group name                          | Object contract                                                    | Registry / loader                             |
|-------------------------------------|--------------------------------------------------------------------|-----------------------------------------------|
| `compgen.kernels.providers`         | `KernelProvider` protocol                                          | `compgen.plugins.discover_all`                |
| `compgen.transforms.decompositions` | Callable `(operands, meta, node_name) -> DecompResult`             | `compgen.plugins.discover_all`                |
| `compgen.kernels.fusion_rules`      | Callable `(producer_v3, consumer_v3) -> bool \| FusionVerdict`     | `compgen.plugins.discover_all`                |
| `compgen.targets.backends`          | `TargetBackendProtocol`                                            | `compgen.plugins.discover_all`                |
| `compgen.kernels.contracts`         | Callable returning a `KernelContractV3`                             | `compgen.plugins.discover_all`                |
| `compgen.vendor_dialects`           | Factory returning a `VendorDialectAdapter` subclass                | `compgen.extensions.vendor_dialect.registry`  |

## User-space `~/.compgen/extensions/`

- Default root: `~/.compgen/extensions/`. Override with `COMPGEN_EXTENSIONS_DIR`.
- Disable entirely: `COMPGEN_DISABLE_LOCAL_EXTENSIONS=1`.
- Each `*.py` file may define `def register(registry): ...` or module-level
  `TOOL` / `TOOLS` / `SLOT` / `SLOTS` constants.
- Idempotent: `_state.json` tracks what's already been loaded.

## In-tree kernel providers

Not every provider ships as an entry-point package. Core CompGen ships
three in-tree implementations under `compgen.kernels.providers`:

| Provider | Target | Invocation | Guide |
|----------|--------|------------|-------|
| `AutocompProvider` | GPU (Triton / CUDA) | In-process Python | — |
| `ExoProvider` | Accelerators (Gemmini, custom) | In-process Python | — |
| `KernelBlasterProvider` | CUDA | Subprocess (local shell or Docker) | [KernelBlaster Provider](../guides/kernelblaster.md) |

The agent loop registers them alongside any entry-point providers;
`compgen.kernels.registry.ProviderRegistry` dispatches contracts in
registration order.

## Related docs

- [Authoring an extension](../getting-started/extension-authoring.md) — walkthrough
- [Architecture: extension points](../architecture/extension-points.md) — full protocols + examples
- [Architecture: target backend model](../architecture/target-backend-model.md) — for `compgen.targets.backends`
- [Vendor dialects overview](../vendor_dialects.md) — for `compgen.vendor_dialects`
- [KernelBlaster provider](../guides/kernelblaster.md) — subprocess-based kernel search
