# CompGen

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
uv run python scripts/e2e_demo.py
```

The demo is the current best end-to-end path. Most CLI subcommands are still documented contract surfaces rather than fully implemented workflows.

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

## Internal Documentation

Roadmap, status, thesis, and detailed design material moved to `tmp/agentic_documentation/` so the main docs tree stays user-facing.

## License

Apache License 2.0. See [LICENSE](LICENSE).
