# XPU-RT

XPU-RT is an LLM-driven compiler generator for heterogeneous hardware targets.
It does not replace your compiler — it generates the target-specific *recipe*
around one: the transforms, kernel decisions, placement/scheduling plans,
runtime packaging, and verification outputs that turn a PyTorch program into a
verified deployment bundle for a given hardware profile.

The primary way to drive XPU-RT is through Claude Code via its MCP server.
Every pipeline stage is exposed as an MCP tool, so the LLM can inspect, propose,
and verify compilation decisions interactively.

## Install

```bash
pip install xpu-rt
```

Installs the compiler generator + the MCP server (`xpu-rt-mcp`). For the
optional extras, see [docs/getting-started/installation.md](docs/getting-started/installation.md).

## Wire up Claude Code

```bash
xpu-rt mcp install          # merges into ~/.claude.json (backup on edit)
xpu-rt mcp doctor           # verifies tools load and discovery works
```

Then restart Claude Code and the `xpu-rt` server appears in the tool picker.
Prefer to paste the config yourself? `xpu-rt mcp print-config` emits the
snippet to stdout. Project-scoped `.mcp.json` works too: `xpu-rt mcp install --project`.

## Extend it in user space

When you need something XPU-RT doesn't ship — a new kernel provider, a new
target backend, a custom vendor MLIR dialect adapter — scaffold it locally and
the running MCP server picks it up on next restart:

```bash
xpu-rt ext new provider my_chip       # scaffolds a pip-installable pack
cd my_chip && pip install -e .
xpu-rt ext list                       # verify discovery
```

Drop-in Python tools at `~/.xpu_rt/extensions/*.py` are discovered without any
`pip install` step — useful for one-off experimentation. See
[docs/getting-started/extension-authoring.md](docs/getting-started/extension-authoring.md).

## Python API

```python
import torch, torch.nn as nn
from xpu_rt.options import cuda_a100_defaults
from xpu_rt.pipeline import compile_and_diff

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(64, 64)
    def forward(self, x):
        return torch.relu(self.fc(x))

model, x = Block().eval(), torch.randn(1, 4, 64)
report = compile_and_diff(
    model, (x,),
    options=cuda_a100_defaults(),
    fixture_name="my_block",
    eager_reference=model(x).detach(),
    run_compiled_executor=True,
)
print("passed:", report.passed, "opaque rate:", report.opaque_rate)
```

## What's in the box

- Staged xDSL pipeline covering structural, quantization, layout, distributed,
  control-flow, and runtime-side passes.
- Custom dialects `xpu_rt.quant`, `xpu_rt.tensor_ext`, `xpu_rt.linalg_ext`,
  `xpu_rt.event`, `xpu_rt.collective`, plus FP8 + HMX tile primitives on
  `xpu_rt.accel`.
- `CompGenOptions` presets (`cuda_a100`, `cuda_h100`, `npu_fp8`), an LRU
  pipeline cache, differential test harness, Triton kernel emitter, autotuner,
  benchmark harness.
- Real-workload fixtures under `tests/_fixtures/` (SmolVLA, Gemma, TinyLlama,
  Qwen-MoE, VLA-decoder) used by the pipeline probes.
- MCP server (`xpu-rt-mcp`) exposing every stage as a first-class tool that
  Claude Code (or any MCP client) can drive.

## Documentation

- [Docs Home](docs/index.md)
- [Installation](docs/getting-started/installation.md)
- [MCP Setup](docs/getting-started/mcp-setup.md)
- [Extension Authoring](docs/getting-started/extension-authoring.md)
- [Quickstart](docs/getting-started/quickstart.md)
- [CLI Reference](docs/reference/cli.md)
- [Python API](docs/reference/python-api.md)
- [Extension Points](docs/reference/extension-points.md)

## From source (contributors)

```bash
git clone --recurse-submodules https://github.com/xpu-rt-project/xpu_rt.git
cd xpu-rt && ./scripts/bootstrap.sh
```

See [AGENT.md](AGENT.md) for the repository-local operating manual.

## License

Apache License 2.0. See [LICENSE](LICENSE).
