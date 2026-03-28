# CLI Reference

The `compgen` CLI defines the public command surface for the project.

## Current Status

`--help` and `--version` are implemented. Most command bodies currently print their expected contract and then raise `NotImplementedError`.

Use this page to understand the intended CLI shape, not as a promise that every command is runnable today.

## Global Usage

```bash
uv run python -m compgen.cli --help
uv run python -m compgen.cli --version
uv run python -m compgen.cli --llm-backend claude-cli llm show
```

## Global LLM Options

The top-level CLI now exposes project-level LLM selection:

- `--llm-backend {gemini,openai,anthropic,claude-cli,codex-cli}`
- `--llm-model MODEL`
- `--llm-record-dir DIR`
- `--llm-no-record`

These options apply to the whole command invocation and are mirrored into the process environment for downstream code.

## Commands

| Command | Purpose | Current state |
|--------|---------|---------------|
| `init-target PROFILE` | Validate a target profile | Contract only |
| `analyze MODEL --inputs SPEC --target PROFILE` | Capture model and build analysis artifacts | Contract only |
| `generate --target PROFILE --analysis-dir DIR` | Run generation pipeline | Contract only |
| `llm show` | Inspect the resolved LLM backend selection | Implemented |
| `llm smoke` | Run a direct smoke test against the selected backend | Implemented |
| `verify BUNDLE_PATH` | Run verification ladder | Contract only |
| `run BUNDLE_PATH` | Execute a bundle locally | Contract only |
| `promote BUNDLE_PATH` | Promote a verified bundle | Contract only |
| `scaffold-target HARDWARE_SPEC` | Generate a target package | Contract only |

## Benchmark Harness

The benchmark harness is a separate implemented CLI:

```bash
env PYTHONPATH=python python -m benchmarks.cli --help
```

Implemented benchmark commands:

- `check-baselines`
- `list-suites`
- `list-suite-workloads SUITE_ID`
- `probe-suite SUITE_ID`
- `run-case CASE_ID`
- `run-study STUDY_ID`
- `run-suite SUITE_ID`
- `run-suite-workload SUITE_ID WORKLOAD_ID`
- `aggregate RESULTS_DIR`
- `export-suite-results RESULTS_DIR`
- `plot RESULTS_DIR`

The suite runner supports:

- recognized benchmark suites: `torchbench`, `huggingface`, `timm`, `mlperf`, `sol_execbench`, `heterobench`
- pack-backed integrations through `pack_integrations`
- normalized cross-suite JSON exports derived from canonical `RunRecord` files
- explicit workspace-driven suite commands through `suite_configs` and `pack_configs`

Use the [Benchmark Suites guide](../guides/use-benchmark-suites.md) for workspace YAML examples and runnable command lines.

## Recommended Use Today

- Use `--help` to discover the shape of the eventual public CLI.
- Use the [Quickstart](../getting-started/quickstart.md) and [Python API](python-api.md) for runnable workflows.
- Use the [Benchmark Suites guide](../guides/use-benchmark-suites.md) for the separate benchmark harness and suite adapters.

## Notes

- The CLI is still useful as a contract and design boundary.
- The docs intentionally call out stub status instead of documenting the commands as if they already worked end to end.
