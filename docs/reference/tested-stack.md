# Tested Stack

These are the versions and components the repo documents as validated or expected.

## Core Versions

| Component | Version or floor |
|----------|-------------------|
| Python | 3.11+ |
| `torch` | 2.4+ |
| `xdsl` | 0.24+ |
| `click` | 8.1+ |
| `pyyaml` | 6.0+ |
| `jsonschema` | 4.20+ |
| `jinja2` | 3.1+ |
| `rich` | 13.0+ |
| `structlog` | 24.1+ |

## Optional Components

| Area | Package group |
|------|---------------|
| Docs | `--extra docs` |
| Solvers | `--extra solve` |
| LLM clients | `--extra llm` |
| IREE integration | `--extra iree` |

## Platform Expectations

- Linux is the primary development platform.
- GPU and CUDA are optional for many flows, but required for some hardware-specific paths and benchmarks.
- `autocomp` is expected as an editable submodule install after bootstrap.
