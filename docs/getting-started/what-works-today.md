# What Works Today

This page is intentionally strict about current state.

## Implemented and Runnable

| Surface | Status | Notes |
|--------|--------|-------|
| `./scripts/bootstrap.sh` | Implemented | Installs the repo and runs smoke checks |
| `uv run python -m compgen.cli --help` | Implemented | Good for discovering the command surface |
| `uv run python scripts/e2e_demo.py` | Runnable demo | Best public path through the current system |
| `compgen.device()` | Implemented | Consumes a targetgen-style hardware spec YAML |
| `compgen.compile_model()` | Implemented | Returns a `CompiledModel` that benchmarks with the local executor |
| `python -m benchmarks.cli list-suites` | Implemented | Probes the recognized benchmark suites and pack integrations |
| `python -m benchmarks.cli run-suite-workload ...` | Runnable with configured inputs | Runs one benchmark-suite workload and emits normalized result JSONs |
| Example target profiles in `examples/target_profiles/` | Available | Used by lower-level profile-centric flows and the demo |

## Implemented but More Advanced

| Surface | Status | Notes |
|--------|--------|-------|
| `python/compgen/targetgen/generate.py` | Implemented | Generates target artifacts from a hardware spec |
| Hardware-spec exemplars | Available | Public example now lives at `examples/hardware_specs/gpu_simt_demo.yaml` |
| Bundle creation | Implemented | Used by the demo to emit `manifest.json` plus artifacts |

## Declared but Not Yet a Full User Workflow

| Surface | Status | Notes |
|--------|--------|-------|
| `compgen init-target` | Contract only | Prints expected behavior, then raises `NotImplementedError` |
| `compgen analyze` | Contract only | CLI shape is defined, end-to-end command not implemented |
| `compgen generate` | Contract only | Same |
| `compgen verify` | Contract only | Same |
| `compgen run` | Contract only | Same |
| `compgen promote` | Contract only | Same |
| `compgen scaffold-target` | Contract only | CLI contract exists; use Python APIs for current experimentation |

## Practical Guidance

- If you need a real first run, use the demo.
- If you need a scriptable entrypoint, use the Python API.
- If you need benchmark coverage or cross-suite result exports, use the benchmark harness guide and `python -m benchmarks.cli`.
- If you need the eventual CLI shape, use the CLI reference, but treat command execution semantics as planned unless noted otherwise.
