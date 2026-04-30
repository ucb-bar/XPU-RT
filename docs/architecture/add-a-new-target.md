# Add a new target — the 50-LoC cookbook

You want to plug a new accelerator (or new arch / new vendor /
new class) into CompGen. This is the recipe.

The target hierarchy + Protocols + registry are documented in
[`target-hierarchy.md`](./target-hierarchy.md). Read that for the
concept; come back here for the steps.

## Decide which level you're at

Before writing any code, classify your target:

- **New arch under existing vendor** (e.g. NVIDIA ships sm_130
  tomorrow): you only need the four arch-specific files; the
  vendor-common code (CUDA driver, NVRTC) is reused.
- **New vendor under existing class** (e.g. Tenstorrent under GPU
  class): write the vendor-common scaffold + at least one arch
  leaf.
- **New class entirely** (e.g. a dataflow accelerator that doesn't
  match GPU / CPU / TPU): write `targets/{class}/contracts.py`
  with new Protocols + at least one vendor + arch.

This doc covers the most common case — new arch under existing
vendor.

## Two paths

### Path A — Through MCP, no source fork

Best when you're prototyping, an experimental dialect, or a
deployment-specific tuning.

```python
# Inside an agent's session — no edits to CompGen source:
compgen_register_target(
    target_class="gpu",
    vendor="tenstorrent",
    arch="gridx",
    rationale="Experimental Tenstorrent path for testing",
    body_emitter_module="my_pkg.adapters.GridxBodyEmitter",
    runtime_module="my_pkg.adapters.GridxRuntime",
    probe_module="my_pkg.adapters.GridxProbe",
    cost_model_module="my_pkg.adapters.GridxCostModel",
    metadata={
        "supports_clusters": False,
        "default_tile_shape": [64, 32, 32],
    },
)
```

Your `my_pkg.adapters.*` classes implement the four Protocols.
That's it — `compgen.compile_to_megakernel(model, target="gpu.tenstorrent.gridx")`
now routes through your code.

The dotted-module-path approach makes this MCP-stdio safe (no live
callable transport over JSON). Adapter classes need a zero-arg
constructor.

### Path B — In-tree, ship a PR upstream

Best when your target is going to be a long-lived part of CompGen.

#### Step 1: Copy the template

```bash
cp -r python/compgen/targets/_template python/compgen/targets/gpu/nvidia/sm_130
```

The template ships:

- `__init__.py` — registers the package on import.
- `probe.py` — hardware detection.
- `body_emitter.py` — per-op kernel sources.
- `runtime.py` — JIT compile + dispatch.
- `cost.py` — TFLOPS / overhead numbers.

#### Step 2: Rename + fill in

In each file:

1. Rename the class (`TemplateProbe` → `Sm130Probe` etc.).
2. Fill in the methods. Each one has a docstring describing what
   it should return — see the corresponding leaf
   (`gpu/nvidia/blackwell/`, `cpu/x86/`) for working examples.

#### Step 3: Register on import

Edit `__init__.py` to call `register_target`:

```python
from compgen.targets.gpu.nvidia.sm_130.body_emitter import Sm130BodyEmitter
from compgen.targets.gpu.nvidia.sm_130.cost import Sm130CostModel
from compgen.targets.gpu.nvidia.sm_130.probe import Sm130Probe
from compgen.targets.gpu.nvidia.sm_130.runtime import Sm130Runtime
from compgen.targets.registry import register_target


def _register_sm130() -> None:
    register_target(
        target_class="gpu",
        vendor="nvidia",
        arch="sm_130",
        probe=Sm130Probe(),
        body_emitter=Sm130BodyEmitter(),
        runtime=Sm130Runtime(),
        cost_model=Sm130CostModel(),
        rationale="...",
        metadata={...},
    )

_register_sm130()
```

#### Step 4: Wire into `compgen.targets.__init__`

Add your package path to `_register_in_tree()` so it auto-registers
on import:

```python
in_tree_modules = (
    "compgen.targets.gpu.nvidia",
    ...
    "compgen.targets.gpu.nvidia.sm_130",  # ← your new arch
)
```

#### Step 5: Tests

Copy `tests/targets/test_cpu_x86_stub.py` as your reference. It
covers:

- Probe satisfies the class-level Protocol (`isinstance` check).
- Body emitter produces compilable source.
- Runtime can JIT compile + dispatch a real op (round-trip
  validated against numpy / torch).
- The package registers on import.
- Audit metadata is pinned (so future changes don't drift).

The CPU x86 stub's full end-to-end JIT test is the gold standard:
emit a body, compile via the system toolchain, dispatch via
ctypes, validate output bit-exact-ish against numpy. If your
target follows the same shape, the abstraction holds.

### Path C — Third-party PyPI package

Best when you want the user to `pip install` your target rather
than upstreaming.

Declare an entry point in your package's `pyproject.toml`:

```toml
[project.entry-points."compgen.targets"]
my_target = "my_pkg.compgen_integration:register"
```

Where `register` is a zero-arg function that calls
`compgen.targets.registry.register_target(...)`.

A user who installs your wheel + calls `discover_entry_points()`
gets your target registered automatically. Use the same Protocol
shape as in-tree targets — the registry doesn't distinguish.

## What "50 LoC" actually means

Here's the LoC budget once your scaffold is in place:

| File | LoC | What's there |
|---|---:|---|
| `__init__.py` | 25 | imports + register_target call |
| `probe.py` | 40 | 6 method stubs, `is_available` non-trivial |
| `body_emitter.py` | 80-300 | depends on # supported ops + complexity |
| `runtime.py` | 50-150 | depends on JIT toolchain quirks |
| `cost.py` | 25 | per-arch TFLOPS table + 4 method shells |

The "50 LoC" claim assumes you're inheriting most of the heavy
lifting from the vendor-common layer. A fresh class-level target
(no parent vendor) is 200-500 LoC.

## Acceptance — when is your target "done"

The same checklist the in-tree leaves go through:

1. **Protocol satisfaction** — `isinstance(YourBodyEmitter(),
   GpuBodyEmitter)` is True (via `@runtime_checkable`).
2. **JIT round-trip** — emit a body, compile it through your
   target's toolchain, dispatch on real input, validate output
   against a reference (numpy / torch / nvidia eager / etc).
3. **Audit query** — `compgen_describe_target(target_id="...")`
   returns useful metadata.
4. **Registry navigation** — your target shows up in
   `registry().tree()` as a peer of existing in-tree targets.
5. **No leakage into universal modules** — your target adds zero
   code outside `targets/{class}/{vendor}/{arch}/`. The matcher,
   autotune, cost predictor, dispatch path don't import from
   your package.

If all five hold, the target is real architecture under the
unified hierarchy — just like the in-tree NVIDIA Blackwell or
CPU x86 packages.
