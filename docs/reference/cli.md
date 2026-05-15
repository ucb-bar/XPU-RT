# CLI Reference

The `compgen` console script is installed with `pip install compgen`. It
exposes the public command surface — discovery, MCP wire-up, extension
scaffolding, and the pipeline subcommands.

## Status

`compgen --help`, `--version`, the `mcp` subgroup, the `ext` subgroup, the
`llm` subgroup, `contrib`, and `scaffold-pack` are implemented. The
pipeline subcommands (`init-target`, `analyze`, `generate`, `verify`, `run`,
`promote`, `scaffold-target`) define the intended contract — their bodies
print their expected shape and fall back to partial implementations.

Use this page as the intended CLI shape, not as a promise that every
pipeline command is runnable end-to-end.

## Global usage

```bash
compgen --help
compgen --version
compgen --llm-backend claude-cli llm show
```

Without an active venv the console script resolves via the entry point
installed by `pip`. If you work out of a source checkout, `uv run compgen
--help` works too.

## Global LLM options

The top-level command exposes project-level LLM selection:

- `--llm-backend {gemini,openai,anthropic,claude-cli,codex-cli}`
- `--llm-model MODEL`
- `--llm-record-dir DIR`
- `--llm-no-record`

These apply to the entire invocation and are mirrored into
`COMPGEN_LLM_BACKEND` / `COMPGEN_LLM_MODEL` for downstream code.

## MCP subcommands — `compgen mcp ...`

| Command | Purpose |
|--------|--------|
| `compgen mcp serve` | Run the `compgen-mcp` stdio server in the current process |
| `compgen mcp tools` | List every MCP tool this server exposes |
| `compgen mcp print-config` | Emit the canonical `.mcp.json` snippet to stdout |
| `compgen mcp install [--project] [--target PATH] [--force] [--dry-run]` | Merge the server entry into `~/.claude.json` (or `./.mcp.json` with `--project`), with a timestamped backup |
| `compgen mcp doctor` | Import tool tree, list discovered extensions, check the MCP SDK, verify `compgen-mcp` is on `PATH` |

See [MCP Setup](../getting-started/mcp-setup.md) and
[MCP Tools Reference](mcp-tools.md).

## Extension subcommands — `compgen ext ...`

| Command | Purpose |
|--------|--------|
| `compgen ext list` | Show every discovered extension across entry-point plugins, vendor dialects, and `~/.compgen/extensions/` |
| `compgen ext new <kind> <name> [--out DIR]` | Scaffold a pip-installable extension pack (`quantization`, `target`, `provider`, `dialect`, or `runtime`) |
| `compgen ext doctor` | Re-run every discovery validator and report failures |

See [Extension Authoring](../getting-started/extension-authoring.md) and
[Extension Points](extension-points.md).

## LLM subcommands — `compgen llm ...`

| Command | Purpose | State |
|--------|--------|-------|
| `compgen llm show` | Show resolved backend + environment | Implemented |
| `compgen llm smoke [--prompt ...] [--structured]` | One-shot smoke call against the selected backend | Implemented |

## Pipeline subcommands

| Command | Purpose | Current state |
|--------|--------|---------------|
| `compgen init-target PROFILE` | Validate a target profile | Partial — loads + prints profile |
| `compgen analyze MODEL --inputs SPEC --target PROFILE` | Capture model + build analysis artifacts | Partial — runs Stage 0/1/2 best-effort |
| `compgen generate --target PROFILE --analysis-dir DIR` | Run LLM generation pipeline | Partial — writes transform + plan stubs |
| `compgen verify BUNDLE_PATH` | Run verification ladder | Partial — structural/functional/performance/formal levels |
| `compgen run BUNDLE_PATH` | Execute a bundle locally | Partial — loads golden inputs, reports verification status |
| `compgen promote BUNDLE_PATH` | Promote a verified bundle to the recipe library | Implemented via `RecipePromoter` |
| `compgen scaffold-target HARDWARE_SPEC` | Generate a target package | Implemented — emits a target enablement package |
| `compgen scaffold-pack --kind ... --name ...` | Scaffold a pip-installable extension pack | Implemented — identical to `compgen ext new` |

## Contrib — drafting upstream contributions

| Command | Purpose |
|--------|--------|
| `compgen contrib list` | List local extensions + invocation counts |
| `compgen contrib status` | Show which extensions are eligible for upstream drafting |
| `compgen contrib draft --slot NAME` | Draft a patch + regression test from a local extension |

## Benchmark harness (separate)

The benchmark harness is a distinct implemented CLI:

```bash
env PYTHONPATH=python python -m benchmarks.cli --help
```

Implemented benchmark commands:

- `check-baselines`, `list-suites`, `list-suite-workloads SUITE_ID`
- `probe-suite SUITE_ID`
- `run-case CASE_ID`, `run-study STUDY_ID`
- `run-suite SUITE_ID`, `run-suite-workload SUITE_ID WORKLOAD_ID`
- `aggregate RESULTS_DIR`, `export-suite-results RESULTS_DIR`, `plot RESULTS_DIR`

See the [Benchmark Suites guide](../guides/use-benchmark-suites.md) for
workspace YAML examples and runnable command lines.

## Recommended use today

- `compgen mcp install` + Claude Code for the full interactive surface.
- `compgen ext new ...` + `pip install -e` for extending CompGen in user space.
- The Python API (`compgen.api`, `compgen.pipeline.compile_and_diff`) for
  scripted runs.
- The demo script and benchmark harness for end-to-end validation.
