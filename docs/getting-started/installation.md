# Installation

CompGen is a Python 3.11+ project managed with `uv`.

## Prerequisites

- Python 3.11 or newer
- `uv`
- `git`

## Supported Install Path

Clone the repo with submodules, then run the bootstrap script:

```bash
git clone --recurse-submodules https://github.com/compgen-project/compgen.git
cd compgen
./scripts/bootstrap.sh
```

The bootstrap script initializes submodules, creates `.venv/`, installs the project, installs the editable `autocomp` dependency, and runs lightweight smoke checks.

## Smoke Checks

Use `uv run` so you do not need to activate the virtual environment manually:

```bash
uv run python -m compgen.cli --help
uv run pytest tests/test_version.py
```

If you want a real runnable path after install, continue to the [Quickstart](quickstart.md).

## Optional Workflows

- Docs build: `uv sync --extra docs`
- Solver-related work: `uv sync --extra solve`
- LLM client work: `uv sync --extra llm`

## Notes

- GPU support is optional. The demo and most tests can still run on CPU-only machines.
- The CLI command surface exists, but most pipeline commands are still stubbed. The runnable path today is the demo and Python API, not the full CLI pipeline.
