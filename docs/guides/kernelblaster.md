# KernelBlaster kernel provider

[KernelBlaster](https://github.com/NVlabs/KernelBlaster) is NVlabs'
memory-augmented RL loop for CUDA kernel optimisation. CompGen ships a
first-class `KernelBlasterProvider` that implements the same
`KernelProvider` protocol as the Autocomp provider, so KernelBlaster
participates in `ProviderRegistry.search()` exactly like any other
backend.

Unlike Autocomp, KernelBlaster is not a Python library — it ships as a
Docker image + shell script and assumes an NVIDIA GPU. CompGen's
adapter orchestrates the subprocess call, stages KB's expected input
tree, and parses its output database back into CompGen types.

## Installation modes

The adapter supports two modes, auto-detected from the environment.

### Local mode — source checkout

KernelBlaster ships as a git submodule at `third_party/kernelblaster`.
`./scripts/bootstrap.sh` initialises it. To enable the provider:

```bash
# Already present after bootstrap; re-run if needed:
git submodule update --init third_party/kernelblaster

# Point the adapter at it for the current shell:
export COMPGEN_KERNELBLASTER_ROOT=$PWD/third_party/kernelblaster
```

The adapter builds an overlay workdir that symlinks KB's `scripts/`,
`src/`, `utils/`, etc. into a temp directory and stages the caller's
`init.cu` / `driver.cpp` under a matching `data/<dataset>/<level>/NNN_<name>/`
entry, so the user's KB checkout is never mutated. It then runs
`bash <workdir>/scripts/run_single_kernelblaster.sh …`.

Suitable for development machines with KB's Python/CUDA dependencies
already installed.  KB itself needs `pip install -e third_party/kernelblaster`
or its equivalent to make its Python package importable — CompGen does
not do this automatically; run the step KB's own README prescribes.

### Docker mode — pre-built image

```bash
cd third_party/kernelblaster
docker build . -t kernelblaster:latest
export COMPGEN_KERNELBLASTER_IMAGE=kernelblaster:latest
```

The adapter runs `docker run --rm --gpus all -v <workdir>:/workspace
<image> bash scripts/run_single_kernelblaster.sh …`.

### Force a mode

Override the auto-detect via `COMPGEN_KERNELBLASTER_MODE=local` or `=docker`.

## Required environment

Both modes require:

| Variable | Purpose |
|----------|---------|
| `OPENAI_API_KEY` | KB uses OpenAI for candidate mutations |

Tunables (sensible defaults baked into the adapter):

| Variable | Default | Purpose |
|----------|---------|---------|
| `COMPGEN_KERNELBLASTER_MODEL` / `MODEL` | `gpt-5-mini-2025-08-07` | OpenAI model used by KB |
| `COMPGEN_KERNELBLASTER_GPU_TYPE` / `GPU_TYPE` | `H100` | GPU target tag |
| `COMPGEN_KERNELBLASTER_DATASET` / `DATASET` | `kernelbench-cuda` | KB's dataset label |
| `COMPGEN_KERNELBLASTER_PRECISION` / `PRECISION` | `fp16` | KB's precision label |
| `COMPGEN_KERNELBLASTER_EXPERIMENT` / `EXPERIMENT_NAME` | `compgen_run` | Output subdirectory |

The adapter forwards `HF_TOKEN`, `HUGGINGFACE_TOKEN`, and
`WANDB_API_KEY` when set.

## Contract inputs

KernelBlaster needs the CUDA kernel to optimise plus a C++ validation
harness. Pass both through `contract.constraints.kernelblaster`:

```python
from compgen.kernels.provider import KernelContract, SearchBudget
from compgen.kernels.providers.kernelblaster import KernelBlasterProvider

contract = KernelContract(
    region_id="matmul_0",
    op_family="matmul",
    input_shapes=((128, 128), (128, 128)),
    output_shapes=((128, 128),),
    dtypes=("fp16",),
    target_name="cuda",
    hardware_key="H100",
    constraints={
        "kernelblaster": {
            "init_cu": open("path/to/init.cu").read(),
            "driver_cpp": open("path/to/driver.cpp").read(),
            # Optional overrides — each falls back to the env default
            "dataset": "kernelbench-cuda",
            "level": "level1",
            "problem_id": 1,
            "problem_name": "my_matmul",   # NNN_<name> dir under data/
            "precision": "fp16",
            # Any extra files to stage alongside init.cu / driver.cpp
            # (keys are paths relative to the problem dir):
            "extra_files": {"reference.h": "..."},
        }
    },
)

provider = KernelBlasterProvider()
result = provider.search(contract, SearchBudget(max_iterations=20))
```

The adapter lays out KB's expected tree in a temp overlay directory:

```
<tmp>/scripts   -> third_party/kernelblaster/scripts        (symlink)
<tmp>/src       -> third_party/kernelblaster/src            (symlink)
<tmp>/utils     -> third_party/kernelblaster/utils          (symlink)
<tmp>/data/<dataset>/<level>/NNN_<problem_name>/init.cu      (staged)
<tmp>/data/<dataset>/<level>/NNN_<problem_name>/driver.cpp   (staged)
```

KB's `run_single_kernelblaster.sh` computes `ROOT_DIR` from its own
path, so invoking it via the overlay puts `ROOT_DIR=<tmp>` — and KB's
subsequent `data/` + `out/` lookups resolve against the overlay, not
the user's checkout. Output lands at:

```
<tmp>/out/<dataset>/<precision>/<experiment>/final_rl_cuda_perf.cu
<tmp>/out/<dataset>/<precision>/<experiment>/optimization_database.json
```

## How it participates in the registry

The agent loop registers KernelBlaster alongside Autocomp and Exo in
`compgen/agent/env/core.py`:

```python
self._provider_registry.register(AutocompProvider())
self._provider_registry.register(ExoProvider())
self._provider_registry.register(KernelBlasterProvider())
```

Registry search tries providers in registration order. The first that
both `accepts_contract` and returns `found=True` wins. KB's
`accepts_contract` filters for CUDA targets *and* the presence of
`init_cu`/`driver_cpp` — contracts without KB payloads pass cleanly
through to the next provider.

## Graceful degradation

If the host isn't provisioned (no source tree, no docker, no API key),
the provider returns `ProviderResult(found=False)` with the specific
reason in `metadata["reason"]` — it never crashes the pipeline. Check
pre-flight with:

```python
from compgen.kernels.kernelblaster_adapter import KernelBlasterAdapter

ok, reason = KernelBlasterAdapter().is_available()
print("KernelBlaster available:", ok, reason or "")
```

## Knowledge + contract feedback

On success the adapter parses KB's `optimization_database.json` and
surfaces:

- `ProviderResult.knowledge_exports` — lessons KB learned (tiling,
  async-copy strategy, etc.). `ProviderRegistry.ingest_knowledge()`
  persists these into `CompilerMemory`.
- `ProviderResult.contract_feedback` — KB-suggested contract
  modifications (e.g. layout, dtype) with measured gain.
  `ProviderRegistry.evolve_contract()` rolls them into future contracts.

The provider accumulates exports across calls; drain them via
`provider.export_knowledge()`.

## Testing

```bash
uv run --no-sync pytest tests/kernels/providers/test_kernelblaster_provider.py
```

The test suite mocks the subprocess runner + fakes KB's `out/` tree, so
CI doesn't need Docker, a GPU, or an OpenAI API key. To exercise the
real KB:

```bash
export OPENAI_API_KEY=sk-...
export COMPGEN_KERNELBLASTER_ROOT=$PWD/third_party/kernelblaster
uv run --no-sync python -c "
from compgen.kernels.providers.kernelblaster import KernelBlasterProvider
from compgen.kernels.provider import KernelContract, SearchBudget
contract = KernelContract(
    region_id='smoke',
    op_family='matmul',
    target_name='cuda', hardware_key='H100',
    constraints={'kernelblaster': {
        'init_cu': open('my_kernel.cu').read(),
        'driver_cpp': open('my_driver.cpp').read(),
    }},
)
print(KernelBlasterProvider().search(contract, SearchBudget(max_iterations=5)))
"
```

## Related

- [Extension Points](../reference/extension-points.md) — how providers
  participate in CompGen.
- [Architecture → Extension Points](../architecture/extension-points.md) —
  full `KernelProvider` protocol + alternative backends.
- Source: `python/compgen/kernels/kernelblaster_adapter.py` +
  `python/compgen/kernels/providers/kernelblaster.py`.
- Tests: `tests/kernels/providers/test_kernelblaster_provider.py`.
