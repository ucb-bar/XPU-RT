# Target Pack — User-Space Expansion Model

This page consolidates how a third party adds a new hardware target to
XPU-RT without forking the repo. It pulls together the four
entry-point groups, the contract surfaces each one carries, and the
end-to-end flow from `pip install` to `bundle/generated_kernels/`.

> **TL;DR.** A target pack is a pip-installable Python package that
> declares four entry points. XPU-RT discovers and consumes them at
> import time. The shape is identical regardless of the target's
> archetype (Triton-friendly GPU, accel-native, ukernel-runtime CPU,
> hybrid). To get a working skeleton:
> ```bash
> xpu-rt scaffold-pack --kind target_pack --name my_target --out ./packs
> ```

## What a target pack is

A target pack is a Python distribution that satisfies the
*compiler-generator-for-this-target* contract by plugging into four
extension points. Nothing about a pack is target-specific to XPU-RT
— packs live outside the XPU-RT tree, under independent version
control, and ship to PyPI on their own schedule.

The reference pack today is `radiance-xpu-rt-pack` (chipyard / Muon
GPU). It demonstrates that the architecture is generic; a second
reference pack against a different archetype (CPU scalar-C, NPU,
RVV, …) follows the same shape with no XPU-RT-side changes.

## The four entry-point groups

Each group is declared in the pack's `pyproject.toml`:

```toml
[project.entry-points."xpu_rt.packs"]
my_target = "my_target"

[project.entry-points."xpu_rt.targets.backends"]
my_target = "my_target.backend:MyTargetBackend"

[project.entry-points."xpu_rt.kernels.providers"]
my_target = "my_target.kernels:MyTargetProvider"

[project.entry-points."xpu_rt.mcp.tools"]
my_target = "my_target.mcp:MY_TARGET_TOOLS"
```

Each row maps to a XPU-RT surface:

| Entry-point group | What it provides | Protocol / shape | Where XPU-RT consumes it |
|---|---|---|---|
| `xpu_rt.packs` | Manifest discovery — `manifest.yaml` describes the pack's owned/sealed surfaces. | YAML manifest + Python module exposing `PACK_ROOT`. | `xpu_rt.packs.load_pack` walks installed entry points. |
| `xpu_rt.targets.backends` | Stage-pipeline lowering specific to this target. | `TargetBackendProtocol` (`supports_target`, `get_options`, `get_compilation_stages`, `compile_stage`, `validate`). | `StageRegistry` registers the backend and routes IR through its `compile_stage` per stage in the pipeline. |
| `xpu_rt.kernels.providers` | Codegen — turns a `KernelContract` into kernel source. | `KernelProvider` Protocol (`accepts_contract`, `search`, `export_knowledge`, `name`). | `xpu_rt.kernels.codegen_fallback.run_provider_fallback` walks providers, asks each whether it accepts the contract, calls `search` on the first that does, writes the result to `bundle/generated_kernels/<provider>/<op>.<ext>`. |
| `xpu_rt.mcp.tools` | Pack-owned MCP verbs (e.g. `my_target_compile_and_run`). | List of tool dicts (`name`, `description`, `input_schema`, `handler`, `phase`). | `xpu_rt.mcp.tools.get_all_tools()` discovers and merges into `ALL_TOOLS`; `xpu-rt mcp tools` lists them. |

A pack does not need to populate every group. Minimum viable target
pack: `xpu_rt.packs` + `xpu_rt.kernels.providers` (with a
HardwareSpec YAML so `xpu_rt.api.device(spec_path)` works). The
`xpu_rt.targets.backends` and `xpu_rt.mcp.tools` groups are
optional refinements.

## Contract surfaces

The four groups talk to XPU-RT through three documented contracts.
A pack author should treat these as stable; XPU-RT treats them as
the bidirectional surface that providers + backends co-evolve with.

### `KernelContract` (input to a Provider's `search`)

Defined in `xpu_rt.kernels.provider`. Carries everything XPU-RT
extracted from the IR for one kernel-eligible region:

```python
KernelContract(
    region_id="region_0",        # stable identifier within this compile
    op_family="add",             # cleaned op name: aten_add → add, aten_relu_default → relu
    input_shapes=((4,), (4,)),   # concrete tuple of int tuples
    output_shapes=((4,),),
    dtypes=("f32",),             # mlir dtype names: f32, f16, bf16, i32, …
    target_name="my_target",
    hardware_key="...",          # device.name from the target profile
    objective="latency",         # "latency" | "throughput" | "power" | …
    constraints={...},           # caller-supplied bounds (e.g. block sizes)
    provider_hints={...},        # accumulated hints across compile_model calls
)
```

A Provider's `accepts_contract(contract) -> bool` should be cheap and
side-effect-free. XPU-RT calls it for every contract in walk order
and dispatches to the first provider that returns True. If no
provider accepts, the kernel slot ends up `skipped` in the bundle
manifest.

### `ProviderResult` (output of `search`)

```python
ProviderResult(
    found=True,                  # False = "I accepted but couldn't generate"
    kernel_code="// real source\n",  # written to disk verbatim
    language="cpp",              # → bundle file extension via _LANGUAGE_EXTENSIONS
    correct=True,                # set after on-target validation, if available
    latency_us=...,              # optional measurement
    speedup=...,
    knowledge_exports=[...],     # what the provider learned (persisted to memory)
    contract_feedback=[...],     # suggestions to evolve the contract
    metadata={...},
)
```

Per-region provenance: when a Provider's `search` returns
`found=True` with non-empty `kernel_code`, XPU-RT rewrites that
op's `xpu_rt.codegen_backend` annotation from `"fallback"` to the
provider's `name`. Multiple providers can win different regions of
the same module; each region's annotation reflects its actual
emitter.

### `HardwareSpec` (declarative target description)

YAML loaded by `xpu_rt.targetgen.load_hardware_spec`. Schema in
`xpu_rt.targetgen.hardware_spec`. Sections used downstream:

- `platform` — vendor, family, chip name, host arch, deployment model
- `execution_model` — `simt_gpu` / `simd_vector` / `decoupled_matrix` / …
- `isa` — base ISA + extensions
- `engine_geometry` — vector length, warp size, tile shapes
- `memory_model` — address spaces, DMA, alignment
- `numeric_contract` — supported dtypes, denormal/NaN handling
- `verification_surface` — `simulator_command` template (with
  `{elf}`/`{config}`/`{chipyard_root}`/`{sim_backend}`/`{extra_make_args}`
  substitution), optional `build_command`, golden model, max ULP error

The HardwareSpec is what `xpu_rt.api.device(spec_path)` consumes to
build the `CompGenDevice` handle that `compile_model` needs.

## End-to-end flow

```
[pack-side]                          [xpu-rt]
─────────────────────────────────────────────────────────────────
my_target/specs/my_target.yaml ─────► device(spec_path)
                                        │
                                        ▼
PyTorch nn.Module ─────────────────► compile_model(model, dev)
                                        │
                                        │  capture (torch.export)
                                        ▼
                                     Payload IR (xDSL)
                                        │
                                        │  stage pipeline
                                        │  (uses MyTargetBackend if registered)
                                        ▼
                                     post-pipeline IR
                                        │
                                        │  build_kernel_contracts()
                                        │  → list[KernelContract]
                                        ▼
                                     run_provider_fallback()
                                        │
                                        │  for each contract:
                                        │    asks every Provider
                                        │    .accepts_contract()
                                        ▼
my_target.kernels:                   first acceptor wins
MyTargetProvider.search(contract) ── (calls into pack)
   returns ProviderResult            │
   with kernel_code                  ▼
                                     bundle_emit
                                        │
                                        ▼
                                     bundle/generated_kernels/
                                       my_target/aten_add.cpp
                                       my_target/aten_relu_default.cpp
                                       index.json
```

Pack-side build / sim runs separately, typically through pack-owned
MCP tools that wrap `xpu_rt.mcp.tools.embedded.simulator_run`.
Since REQ-013, `simulator_run(execute=True)` supports build-less
flows where the spec's `simulator_command` does its own build (e.g.
`make -C sims/vcs run-binary BINARY={elf}`).

## Archetypes

Different targets pin different defaults. The Provider is the same
Protocol regardless; what changes is how `search` materialises the
kernel.

| Archetype | What `search` emits | Reference |
|---|---|---|
| Triton-friendly GPU | Triton `@triton.jit` Python source | `xpu_rt.kernels.providers.triton_templates` (in tree as a reference, not a pack) |
| Accel-native | Vendor C/C++ targeting custom intrinsics + a runtime ABI like `libxpu_rt` | `radiance-xpu-rt-pack` (chipyard / Muon) |
| Ukernel-runtime CPU | Scalar/SIMD C linked against `libxpu_rt_static.a` | (no in-tree reference yet — write it as a pack) |
| Hybrid | Mix of the above per `op_family` | combine archetypes via multiple providers in one pack, or via multiple packs |

The codegen-fallback dispatcher's per-region provenance means a
single compile can mix providers — Provider A handles `matmul`,
Provider B handles `softmax`, in the same pack or across packs.

## Runtime ABI

Native code emitted by a Provider links against
`libxpu_rt_static.a`, the C11 HAL that ships in
`runtime/native/libxpu_rt/`. Headers in
`runtime/native/libxpu_rt/include/xpu_rt/` are the stable C
ABI: semaphores, command buffers, event tensors, and a `cpu_sync`
driver.

The runtime is built per (target ABI, toolchain) pair. Today XPU-RT
ships a host build (`build/libxpu_rt_static.a`) and a
`riscv64-zephyr-elf` build (`build-riscv/`). For an arbitrary target,
drop a CMake toolchain file next to the existing one and rebuild:

```bash
cd runtime/native/libxpu_rt
cmake -B build-<triple> -DCMAKE_TOOLCHAIN_FILE=toolchains/<triple>.cmake
cmake --build build-<triple>
# → build-<triple>/libxpu_rt_static.a
```

## How to scaffold

```bash
xpu-rt scaffold-pack --kind target_pack --name my_target --out ./packs
cd ./packs/my_target
pip install -e .
pytest tests/ -v   # 5 smoke tests should pass immediately
```

What the scaffolder produces:

```
my_target/
├── pyproject.toml          # all four entry points wired
├── README.md               # what to fill in, how to verify
├── src/my_target/
│   ├── __init__.py         # exposes PACK_ROOT
│   ├── manifest.yaml
│   ├── backend.py          # MyTargetBackend  (skeleton, supports_target=False)
│   ├── kernels.py          # MyTargetProvider (skeleton, accepts_contract=False)
│   ├── mcp.py              # MY_TARGET_TOOLS  (empty list)
│   └── specs/my_target.yaml  # HardwareSpec stub
└── tests/
    └── test_pack_smoke.py  # imports + entry-point discoverability + Protocol shape
```

The skeleton declines every contract until `MyTargetProvider.search`
is filled in. That's deliberate: the pack is installable and
discoverable from day zero, then the user incrementally populates it.

## What you fill in, in what order

1. **`specs/my_target.yaml`** — describe the target. Without this,
   `xpu_rt.api.device(spec_path)` has nothing to load.
2. **`MyTargetProvider.accepts_contract`** — narrow the predicate to
   the op_family / shape / dtype combos you support.
3. **`MyTargetProvider.search`** — render real source for the
   contract. Start with one op_family at the smallest shape; verify
   the bundle file appears and contains expected text.
4. **`MyTargetBackend.compile_stage`** — only needed if your target
   needs stage-pipeline transforms beyond what the auto-generated
   stages do. Most packs leave this as a pass-through.
5. **`MY_TARGET_TOOLS`** — pack-owned MCP verbs for compile +
   build + sim + verify. Reuse `xpu_rt.mcp.tools.embedded.simulator_run`
   for the sim leg.
6. **CMake toolchain** for `libxpu_rt_static.a` if your runtime
   needs the native HAL.

## Verification

Once the pack is installed and the Provider accepts at least one
contract:

```python
from xpu_rt.api import device, compile_model
import torch, torch.nn as nn

dev = device("./packs/my_target/src/my_target/specs/my_target.yaml")
cm = compile_model(
    nn.Sequential(...).eval(),
    dev,
    sample_inputs=(torch.randn(4),),
    verify=False,
)
# bundle_dir = cm.pipeline_result.all_artifacts["bundle_dir"]
# bundle_dir/generated_kernels/my_target/<op>.<ext> appears.
```

The bundle's `manifest.json::extended_artifacts.generated_kernels`
status flips from `skipped` to `ok`, and per-op
`xpu_rt.codegen_backend` annotations in `payload.mlir` show the
provider name in place of `"fallback"`.

## See also

- [`extension-points.md`](extension-points.md) — protocol-level
  reference for all five extension points (this page is target-pack-
  focused).
- [`target-backend-model.md`](target-backend-model.md) — how
  `TargetBackendProtocol` integrates with the stage pipeline.
- [`runtime-model.md`](runtime-model.md) — what
  `libxpu_rt_static.a` exposes and what the Provider's emitted
  code links against.
