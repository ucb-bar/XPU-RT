# Quickstart

The shortest truthful path through CompGen today.

## 1. Install

```bash
pip install compgen
```

See [Installation](installation.md) for extras and the from-source path.

## 2. Wire the MCP server into Claude Code (optional but recommended)

```bash
compgen mcp install            # merges into ~/.claude.json (with backup)
compgen mcp doctor             # verifies tools import and discovery works
```

Restart Claude Code — the `compgen` MCP server now appears in the tool
picker. See [MCP Setup](mcp-setup.md) for project-scoped configs and
troubleshooting.

## 3. Compile a model from Python

```python
import torch, torch.nn as nn
from compgen.options import cuda_a100_defaults
from compgen.pipeline import compile_and_diff

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

## 4. Run the end-to-end demo

Clone the repo (or the demo script) to get the most complete vertical slice:

```bash
git clone --recurse-submodules https://github.com/compgen-project/compgen.git
cd compgen && ./scripts/bootstrap.sh
uv run python scripts/e2e_demo.py
```

The demo exercises capture → Payload IR → kernel contracts → equality
saturation → execution plan → bundle → local benchmark, writes a bundle
directory, and prints its path at the end.

Expected artifacts in the bundle:

- `payload.mlir`
- `execution_plan.yaml`
- `golden_inputs.pt`
- `golden_outputs.pt`
- `manifest.json`

See [Inspect Artifacts](../guides/inspect-artifacts.md) for the details.

## 5. Scaffold an extension

```bash
compgen ext new provider my_chip      # pip-installable starter
compgen ext list                      # verify discovery
```

See [Extension Authoring](extension-authoring.md).

## 6. Where to go next

- [What Works Today](what-works-today.md) — the current implementation boundary.
- [Use the Demo](../guides/use-the-demo.md) — walkthrough of each stage.
- [Bring Up a Target](../guides/bring-up-a-target.md) — experiment with hardware specs.
- [CLI Reference](../reference/cli.md) — every subcommand.
