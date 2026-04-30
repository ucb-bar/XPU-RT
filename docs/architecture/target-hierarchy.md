# Target hierarchy — where backend code lives + how users extend it

**Status**: scaffold in place (Waves 1.10-1.18) · NVIDIA migration partial (Wave 1.14a/b done, c/d in progress) · CPU stub validated (Wave 1.15)

This document is the contract for adding a new target to CompGen.
Every backend-specific assumption — instruction set, JIT toolchain,
launch primitive, perf table — has exactly one home in the
hierarchy. The universal modules (matcher, autotune, cost predictor,
runtime dispatch) consume Protocols; vendor-specific code stays
inside vendor-specific packages.

## The four levels

```
targets/
  ├─ gpu/                      ← class       (any GPU)
  │   ├─ contracts.py            class-level Protocols
  │   │
  │   ├─ nvidia/               ← vendor     (any NVIDIA GPU)
  │   │   ├─ __init__.py         registers vendor-common entry
  │   │   ├─ common/             code shared across all NVIDIA arches
  │   │   │
  │   │   ├─ blackwell/        ← arch       (sm_100 / sm_120)
  │   │   │   ├─ __init__.py     registers leaf
  │   │   │   ├─ probe.py        Blackwell-specific detection
  │   │   │   ├─ body_emitter.py cuBLASDx GEMM, mma.sync
  │   │   │   ├─ cu13_nvrtc.py   the cu13 JIT path
  │   │   │   ├─ cluster_launch.py (Wave 1.6)
  │   │   │   ├─ cost.py         per-arch TFLOPS table
  │   │   │   └─ runtime.py      Blackwell-specific launcher tweaks
  │   │   │
  │   │   ├─ hopper/           ← arch       (sm_90)
  │   │   │   └─ ...
  │   │   │
  │   │   └─ ampere/           ← arch       (sm_80 / sm_86)
  │   │       └─ ...
  │   │
  │   ├─ amd/                  ← vendor     (any AMD GPU; placeholder)
  │   │   └─ ...
  │   │
  │   └─ intel/                ← vendor     (any Intel XPU; placeholder)
  │       └─ ...
  │
  ├─ cpu/                      ← class       (any CPU)
  │   ├─ contracts.py
  │   ├─ x86/                  ← vendor
  │   │   ├─ probe.py
  │   │   ├─ body_emitter.py
  │   │   ├─ runtime.py
  │   │   └─ cost.py
  │   └─ arm/                  ← vendor (placeholder)
  │
  ├─ tpu/                      ← class
  │   └─ contracts.py
  │
  ├─ registry.py               in-process registry of all targets
  │
  └─ custom/                   MCP-registered user targets at session scope
```

## What goes where

The rule of thumb: **specialization goes to the deepest level it
applies to.** A symbol used by every NVIDIA arch lives under
`gpu/nvidia/common/`. A symbol that only matters on Blackwell
(`tcgen05.mma`, `cu13` NVRTC, cluster-launch) lives under
`gpu/nvidia/blackwell/`.

| Concept | Where |
|---|---|
| Universal API + IR (event tensors, megakernel graph) | `python/compgen/runtime/` (outside `targets/`) |
| Pattern matchers (Diamond, FFN, MHA) | `python/compgen/runtime/lowering/` |
| Roofline math | `python/compgen/kernels/cost/roofline.py` |
| Class-level Protocols (every GPU implements) | `targets/{class}/contracts.py` |
| Cross-arch vendor code (CUDA driver wrappers, NVRTC base) | `targets/{class}/{vendor}/common/` |
| Arch-specific leaves (cuBLASDx Blackwell tile, sm_100 only) | `targets/{class}/{vendor}/{arch}/` |
| Per-arch TFLOPS tables | `targets/{class}/{vendor}/{arch}/cost.py` |
| Per-arch JIT specifics (cu13 NVRTC) | `targets/{class}/{vendor}/{arch}/` |
| User-supplied custom targets at session scope | registered into the registry; backed by user's own pkg |

## Each leaf's standard layout

Every concrete target package (whether at the vendor or arch level)
provides the same five files:

```python
targets/{class}/{vendor}/{arch}/
  __init__.py          # registers package with the registry on import
  probe.py             # GpuProbe / CpuProbe / TpuProbe impl
  body_emitter.py      # GpuBodyEmitter / CpuBodyEmitter impl
  runtime.py           # GpuRuntime / CpuRuntime impl
  cost.py              # GpuCostModel / CpuCostModel impl
```

Plus optional files for arch-specific specializations
(`cu13_nvrtc.py`, `cluster_launch.py`, etc.).

## Registry — discovery + extensibility surface

The registry (`compgen.targets.registry`) backs both in-tree targets
and MCP-registered user targets:

```python
>>> from compgen.targets.registry import registry
>>> reg = registry()
>>> reg.classes()
('cpu', 'gpu')
>>> reg.vendors('gpu')
('amd', 'intel', 'nvidia')
>>> reg.arches('gpu', 'nvidia')
('ampere', 'blackwell', 'hopper')
>>> reg.tree()
{'gpu': {'amd': [], 'intel': [], 'nvidia': ['ampere', 'blackwell', 'hopper']},
 'cpu': {'arm': [], 'x86': []}}
>>> pkg = reg.get('gpu.nvidia.blackwell')
>>> pkg.body_emitter
<NvidiaBlackwellBodyEmitter ...>
```

In-tree packages register themselves on `import compgen.targets`.
The registry's vendor-common fallback means an arch-leaf inherits
the vendor's adapters when it doesn't override them.

## Adding a new target — three paths

### Path A — MCP-driven (no source fork)

The agent registers a target at session scope via the MCP tool:

```python
# Inside the agent's session:
compgen_register_target(
    target_class="gpu",
    vendor="tenstorrent",
    arch="gridx",
    rationale="experimental Tenstorrent path",
    body_emitter_module="my_pkg.adapters.GridxBodyEmitter",
    runtime_module="my_pkg.adapters.GridxRuntime",
    probe_module="my_pkg.adapters.GridxProbe",
    cost_model_module="my_pkg.adapters.GridxCostModel",
)
# After this call, `compile_to_megakernel(model, target="gpu.tenstorrent.gridx")`
# routes through the user's adapters.
```

The user's package only needs to:

1. Provide four classes that satisfy the class-level Protocols.
2. Be installable (so the dotted-module-path imports resolve).

No edits to CompGen source. Useful for experimental targets, custom
MLIR dialects (cuda-tile per #099), or per-deployment tuning.

### Path B — In-tree (upstream a new target)

Copy `targets/_template/` (Wave 1.17 — placeholder for the
add-a-new-target template) into `targets/{class}/{vendor}/{arch}/`,
fill in the four adapter classes, register in `__init__.py`, and
ship a PR. The target shows up in `registry().tree()` automatically
because `_register_in_tree()` imports every leaf at session start.

### Path C — Third-party entry-point

A third-party package declares an entry point in its `pyproject.toml`:

```toml
[project.entry-points."compgen.targets"]
my_target = "my_pkg.compgen_integration:register"
```

The `register` callable invokes `compgen_register_target(...)`.
Calling `compgen.targets.registry.discover_entry_points()` picks
up every such third-party registration. Useful for vendors who
ship their own CompGen support.

## Migration notes

Existing NVIDIA code is being migrated from `runtime/native/cuda.py`
and `runtime/lowering/fx_to_megakernel.py` into the appropriate
leaves under `targets/gpu/nvidia/`. Each migration step:

1. Copies the symbol to its new home.
2. Replaces the original with a re-export shim
   (`from new.location import symbol`).
3. Tests pin both old and new import paths to the same callable.

Backward compatibility holds for one round; downstream callers
should migrate to the new locations at their convenience. The
inventory + per-symbol destination map is in
`docs/architecture/target-hierarchy-inventory.md`.

## Universal-vs-target decision tree

When adding new code, ask:

1. **Does this depend on any specific arch?**
   No → universal module (e.g. `runtime/lowering/fx_to_megakernel.py`).
   Yes → target package.
2. **Does this depend on a specific vendor's instruction set / JIT?**
   No → class level (`targets/{class}/contracts.py` if it's a
   Protocol; class-level extensions otherwise).
   Yes → vendor level (`targets/{class}/{vendor}/common/`).
3. **Does this depend on a specific arch within the vendor?**
   No → vendor-common (e.g. CUDA driver wrappers).
   Yes → arch leaf (e.g. cuBLASDx SM<1000>, mma.sync on Blackwell only).

Apply at every change. The architectural invariant: minimum
target-specific code at every level, with the heaviest code at
the leaves.

## Cross-references

- **Inventory + migration plan**:
  `docs/architecture/target-hierarchy-inventory.md`.
- **Wave 1.10's Protocols**:
  `python/compgen/targets/{gpu,cpu,tpu}/contracts.py`.
- **Wave 1.13's MCP tools**:
  `python/compgen/mcp/tools/targets.py`.
- **Wave 1.18's registry**:
  `python/compgen/targets/registry.py`.
- **CPU x86 stub** (the load-bearing test that the abstraction
  holds): `python/compgen/targets/cpu/x86/`.
