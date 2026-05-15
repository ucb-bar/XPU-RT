# What Works Today

This page is intentionally strict about current state.

## Implemented and Runnable

| Surface | Status | Notes |
|--------|--------|-------|
| `pip install xpu-rt` | Implemented | PyPI install; ships the CLI + the `xpu-rt-mcp` server |
| `xpu-rt --help` / `xpu-rt --version` | Implemented | Discovers the command surface without activating a venv |
| `xpu-rt mcp install` / `doctor` / `print-config` | Implemented | Wires the MCP server into Claude Code configs |
| `xpu-rt ext list` / `new` / `doctor` | Implemented | Scaffolds and inspects user extensions |
| `./scripts/bootstrap.sh` (contributors) | Implemented | Source install: submodules, `.venv/`, editable autocomp |
| `uv run python scripts/e2e_demo.py` | Runnable demo | Best public vertical slice through the current system |
| `xpu_rt.device()` | Implemented | Consumes a targetgen-style hardware spec YAML |
| `xpu_rt.compile_model()` | Implemented | Returns a `CompiledModel` that benchmarks with the local executor |
| `python -m benchmarks.cli list-suites` | Implemented | Probes the recognized benchmark suites and pack integrations |
| `python -m benchmarks.cli run-suite-workload ...` | Runnable with configured inputs | Runs one benchmark-suite workload and emits normalized result JSONs |
| Example target profiles in `examples/target_profiles/` | Available | Used by lower-level profile-centric flows and the demo |

## Implemented but More Advanced

| Surface | Status | Notes |
|--------|--------|-------|
| `python/xpu_rt/targetgen/generate.py` | Implemented | Generates target artifacts from a hardware spec |
| Hardware-spec exemplars | Available | Public example now lives at `examples/hardware_specs/gpu_simt_demo.yaml` |
| Bundle creation | Implemented | Used by the demo to emit `manifest.json` plus artifacts |

## Declared but Not Yet a Full User Workflow

| Surface | Status | Notes |
|--------|--------|-------|
| `xpu-rt init-target` | Contract only | Prints expected behavior, then raises `NotImplementedError` |
| `xpu-rt analyze` | Contract only | CLI shape is defined, end-to-end command not implemented |
| `xpu-rt generate` | Contract only | Same |
| `xpu-rt verify` | Contract only | Same |
| `xpu-rt run` | Contract only | Same |
| `xpu-rt promote` | Contract only | Same |
| `xpu-rt scaffold-target` | Contract only | CLI contract exists; use Python APIs for current experimentation |

## Practical Guidance

- If you need a real first run, use the demo.
- If you need a scriptable entrypoint, use the Python API.
- If you need benchmark coverage or cross-suite result exports, use the benchmark harness guide and `python -m benchmarks.cli`.
- If you need the eventual CLI shape, use the CLI reference, but treat command execution semantics as planned unless noted otherwise.
