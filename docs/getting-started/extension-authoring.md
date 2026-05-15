# Authoring a XPU-RT extension

You can extend XPU-RT without forking the repo. The running MCP server picks
up extensions from three discovery paths, in this order:

1. **Entry-point plugins** declared in a `pip install`-ed package's
   `pyproject.toml`. This is the durable, shareable path.
2. **Vendor dialect adapters** under the `xpu_rt.vendor_dialects` entry-point
   group (same mechanism, purpose-built for MLIR dialect integrations).
3. **Drop-in Python files** at `~/.xpu_rt/extensions/*.py` (override with
   the `XPU_RT_EXTENSIONS_DIR` env var). No `pip install` needed — handy for
   experimentation.

`xpu-rt ext list` shows what every path produced. `xpu-rt ext doctor`
re-runs the validators and surfaces failures.

## Scaffolding

```bash
xpu-rt ext new <kind> <name> [--out DIR]
```

`kind` is one of `quantization`, `target`, `provider`, `dialect`, `runtime`.
The command writes a pip-installable starter under `DIR/<name>/` with:

- `pyproject.toml` declaring the right entry-point group,
- a starter module with the plugin object,
- a manifest recording what pack kind it is.

After scaffolding:

```bash
cd <name>
pip install -e .
xpu-rt ext list             # your plugin now shows up
```

## Entry-point groups

XPU-RT declares the following groups (see `xpu_rt.plugins.KNOWN_GROUPS`):

| Group                                | Purpose                                      | Object contract                                                 |
|--------------------------------------|----------------------------------------------|-----------------------------------------------------------------|
| `xpu_rt.kernels.providers`          | Custom kernel-search providers               | `KernelProvider` protocol: `name`, `accepts_contract`, `search`, `export_knowledge` |
| `xpu_rt.transforms.decompositions`  | FX-level op decompositions                   | Callable `(operands, meta, node_name) -> DecompResult`          |
| `xpu_rt.kernels.fusion_rules`       | Fusion-rule predicates                       | Callable `(producer_v3, consumer_v3) -> bool \| FusionVerdict`  |
| `xpu_rt.targets.backends`           | Target-backend implementations               | `TargetBackendProtocol` (see `xpu_rt.targets`)                  |
| `xpu_rt.kernels.contracts`          | `KernelContractV3` factories                 | Callable returning a contract                                    |
| `xpu_rt.vendor_dialects`            | Vendor MLIR dialect adapters                 | Factory returning a `VendorDialectAdapter` subclass              |

Example entry-point block in your `pyproject.toml`:

```toml
[project.entry-points."xpu_rt.kernels.providers"]
my_chip = "my_chip_pkg.provider:MyChipProvider"
```

## Drop-in Python files (`~/.xpu_rt/extensions/`)

Each `.py` file under the user-space directory gets a chance to register
tools or invent-slots against the live LLM registry. A file may:

1. Define `def register(registry): ...` — called with the live
   `xpu_rt.llm.registry.Registry`.
2. *Or* declare module-level `TOOL` / `TOOLS` and `SLOT` / `SLOTS` constants
   (of `Tool` / `InventSlot`), which will be auto-registered.

Loading is idempotent — a state file at
`~/.xpu_rt/extensions/_state.json` records which modules have been loaded,
and subsequent process starts skip them unless the registry is cleared.

Set `XPU_RT_EXTENSIONS_DIR` to point elsewhere (useful for tests). Set
`XPU_RT_DISABLE_LOCAL_EXTENSIONS=1` to skip the loader entirely.

## Verifying

```bash
xpu-rt ext list         # show discovered plugins by group
xpu-rt ext doctor       # re-run validators, report failures
xpu-rt mcp doctor       # end-to-end smoke including discovery
```

Once verified, restart the MCP server (Claude Code, or `xpu-rt mcp serve`)
and the extension participates in the tool surface on next connect.

## Upstreaming

If a local extension earns its keep, `xpu-rt contrib draft --slot <name>`
turns the most common accepted invocations into a regression test and
produces a patch suitable for an upstream PR. See `xpu-rt contrib --help`.
