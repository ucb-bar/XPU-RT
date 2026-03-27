# Quickstart

This is the shortest truthful path through the repo today.

## 1. Bootstrap the Environment

```bash
./scripts/bootstrap.sh
```

## 2. Inspect the CLI Surface

```bash
uv run python -m compgen.cli --help
uv run python -m compgen.cli --version
```

This confirms the package is installed and shows the public command surface.

## 3. Run the Demo

```bash
uv run python scripts/e2e_demo.py
```

The demo currently exercises the most useful runnable path in the repo:

- capture a small PyTorch model with `torch.export`
- convert it to Payload IR
- load an example target profile
- build kernel contracts and strategy decisions
- run equality saturation
- generate an execution plan
- build a temporary bundle
- benchmark locally

## 4. Know What the Demo Produces

The demo writes a temporary bundle directory and prints its location at the end of the run. Inside that bundle you should expect at least:

- `payload.mlir`
- `execution_plan.yaml`
- `golden_inputs.pt`
- `golden_outputs.pt`
- `manifest.json`

See [Inspect Artifacts](../guides/inspect-artifacts.md) for the artifact details.

## 5. Choose Your Next Step

- Read [What Works Today](what-works-today.md) if you want the implementation boundary.
- Read [Use the Demo](../guides/use-the-demo.md) if you want to understand each stage.
- Read [Bring Up a Target](../guides/bring-up-a-target.md) if you want to experiment with hardware specs or target profiles.
