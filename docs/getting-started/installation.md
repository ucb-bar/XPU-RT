# Installation

CompGen is distributed on PyPI and targets Python 3.11+.

## From PyPI (recommended)

```bash
pip install compgen
```

That includes everything needed to run the MCP server (`compgen-mcp`) and
exercise the in-process compile path. Verify the install:

```bash
compgen --version
compgen mcp doctor
```

## Extras

Extras layer on optional functionality. Combine multiple extras with commas.

| Extra        | Adds                                                 |
|--------------|------------------------------------------------------|
| `[compile]`  | Alias for torch + xDSL (currently in base; kept for forward compatibility) |
| `[kernels]`  | Kernel-search backends (autocomp, once published to PyPI) |
| `[llm]`      | Gemini / OpenAI / Anthropic SDKs                      |
| `[solve]`    | CP-SAT, Z3, SciPy for the solver stack               |
| `[ray]`      | Ray distributed control plane                         |
| `[quantization]` | torchao for quantization flows                    |
| `[iree]`     | IREE compiler + runtime adapters                      |
| `[benchmarks]` | Matplotlib + plotting for benchmark scripts         |
| `[docs]`     | MkDocs toolchain for building the docs site           |
| `[demo]`     | `transformers` + `accelerate` for the end-to-end demo |
| `[dev]`      | pytest, ruff, mypy, pre-commit                        |

Example:

```bash
pip install "compgen[llm,solve,ray]"
```

## Next steps

- [Wire the MCP server into Claude Code](mcp-setup.md)
- [Run the quickstart](quickstart.md)
- [Author an extension](extension-authoring.md)

## From source (contributors)

Cloning is only needed if you intend to develop against CompGen itself, or if
you want the `compgen-autocomp` kernel-search dependency before it lands on
PyPI.

```bash
git clone --recurse-submodules https://github.com/compgen-project/compgen.git
cd compgen
./scripts/bootstrap.sh
```

`bootstrap.sh` initialises submodules, creates `.venv/`, installs the project
via `uv`, installs the editable `autocomp` dependency, and runs lightweight
smoke checks.

## Notes

- GPU support is optional. The demo and most tests run on CPU-only machines.
- The CLI command surface exists, but some pipeline commands are still stubs.
  The runnable paths today are the MCP tools, the Python API, and the demo.
