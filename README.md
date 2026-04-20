# RuntimeCompGen

CompGen is an LLM-driven compiler generator for heterogeneous hardware targets.

It is not a monolithic compiler. The goal is to generate the target-specific recipe around compilation: transforms, kernel decisions, planning artifacts, runtime packaging, and verification outputs.

## What You Can Do Today

- install the repo and its submodules
- inspect the public CLI surface with `--help` and `--version`
- run a real demo path through capture, IR conversion, planning, bundling, and benchmarking
- use the top-level Python API for target generation and scripted experiments

## Quickstart

```bash
git clone --recurse-submodules https://github.com/compgen-project/compgen.git
cd compgen
./scripts/bootstrap.sh
uv run python -m compgen.cli --help
uv run python -m compgen.cli --llm-backend claude-cli llm show
uv run python scripts/e2e_demo.py
```

The demo is the current best end-to-end path. Most CLI subcommands are still documented contract surfaces rather than fully implemented workflows.

## LLM Backend Selection

CompGen now has a canonical project-level LLM selection path through the top-level CLI. You can select Gemini, OpenAI, Anthropic, Claude CLI, or Codex CLI without manual Python instantiation:

```bash
uv run python -m compgen.cli --llm-backend gemini llm show
uv run python -m compgen.cli --llm-backend claude-cli --llm-model sonnet llm smoke
uv run python -m compgen.cli --llm-backend codex-cli --llm-model gpt-5.4-mini llm smoke
```

Global options:

- `--llm-backend`: `gemini`, `openai`, `anthropic`, `claude-cli`, or `codex-cli`
- `--llm-model`: override the backend default model or alias
- `--llm-record-dir`: choose where request/response logs are written
- `--llm-no-record`: disable request/response recording

The same selection is mirrored into `COMPGEN_LLM_BACKEND` / `COMPGEN_LLM_MODEL` for downstream code paths inside the same process.

## Documentation

- [Docs Home](docs/index.md)
- [Installation](docs/getting-started/installation.md)
- [Quickstart](docs/getting-started/quickstart.md)
- [What Works Today](docs/getting-started/what-works-today.md)
- [Use the Demo](docs/guides/use-the-demo.md)
- [Bring Up a Target](docs/guides/bring-up-a-target.md)
- [CLI Reference](docs/reference/cli.md)
- [Python API](docs/reference/python-api.md)

## Public Examples

- Target profiles: [`examples/target_profiles/`](examples/target_profiles/)
- Hardware-spec example for `compgen.device(...)`: [`examples/hardware_specs/gpu_simt_demo.yaml`](examples/hardware_specs/gpu_simt_demo.yaml)
- Demo model and script: [`examples/models/`](examples/models/) and [`scripts/e2e_demo.py`](scripts/e2e_demo.py)

## Compile a model end-to-end

`compgen.pipeline.compile_and_diff` takes an `nn.Module`, example inputs,
and a `CompGenOptions` preset; it captures the graph, runs the staged
xDSL pipeline, and returns a differential report against the eager
reference.

```python
import torch
import torch.nn as nn
from compgen.options import cuda_a100_defaults
from compgen.pipeline import compile_and_diff

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(64, 64)
    def forward(self, x):
        return torch.relu(self.fc(x))

model = Block().eval()
x = torch.randn(1, 4, 64)

report = compile_and_diff(
    model, (x,),
    options=cuda_a100_defaults(),
    fixture_name="my_block",
    eager_reference=model(x).detach(),
    run_compiled_executor=True,
)
print("passed:", report.passed)
print("opaque rate:", report.opaque_rate)
print("compiled diff:", report.compiled_diff_max_abs)
```

## What's in the box

- Staged xDSL pipeline covering structural, quantization, layout,
  distributed, control-flow, and runtime-side passes.
- Custom dialects `compgen.quant`, `compgen.tensor_ext`,
  `compgen.linalg_ext`, `compgen.event`, `compgen.collective`, plus
  FP8 + HMX tile primitives on `compgen.accel`.
- `CompGenOptions` presets (`cuda_a100`, `cuda_h100`, `npu_fp8`), an
  LRU pipeline cache, a differential test harness, a CPU reference
  executor, a Triton kernel emitter, an autotuner, and a benchmark
  harness.
- Real-workload fixtures under `tests/_fixtures/` (SmolVLA, Gemma,
  TinyLlama, Qwen-MoE, VLA-decoder) used by the pipeline probes.

## License

Apache License 2.0. See [LICENSE](LICENSE).
