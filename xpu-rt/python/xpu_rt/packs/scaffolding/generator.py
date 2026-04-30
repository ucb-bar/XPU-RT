"""Generator that writes a pip-installable extension-pack skeleton on disk."""

from __future__ import annotations

import keyword
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]

# (pack_kind, manifest_kind_label, template_src_in_repo, scheme_filename)
_KIND_SPEC: dict[str, dict[str, str]] = {
    "quantization": {
        "manifest_kind": "KernelPack",
        "template_src": "python/compgen/quantization/methods/_template.py",
        "scheme_basename": "scheme.py",
        "surface": "quantization_methods",
    },
    "target": {
        "manifest_kind": "TargetPack",
        "template_src": "python/compgen/targets/backends/_template.py",
        "scheme_basename": "backend.py",
        "surface": "target_backend",
    },
    "provider": {
        "manifest_kind": "KernelPack",
        "template_src": "python/compgen/kernels/providers/_template.py",
        "scheme_basename": "provider.py",
        "surface": "kernel_providers",
    },
    "dialect": {
        "manifest_kind": "DialectPack",
        "template_src": "python/compgen/extensions/dialects/_template.py",
        "scheme_basename": "dialect.py",
        "surface": "mlir_dialects",
    },
    "runtime": {
        "manifest_kind": "RuntimePack",
        "template_src": "python/compgen/runtime/adapters/_template.py",
        "scheme_basename": "adapter.py",
        "surface": "runtime_adapters",
    },
    # Multi-surface "full target pack" — generates a pyproject with all
    # four CompGen entry-point groups + Backend/Provider/MCP modules + a
    # HardwareSpec stub + a smoke test. Mirrors the radiance-muon shape.
    "target_pack": {
        "manifest_kind": "TargetPack",
        "template_src": "",  # multi-file emit; ``scheme_basename`` unused
        "scheme_basename": "backend.py",
        "surface": "target_backend",
    },
}

SUPPORTED_KINDS: tuple[str, ...] = tuple(_KIND_SPEC)


@dataclass(frozen=True)
class ScaffoldResult:
    """Paths produced by :func:`scaffold_pack`."""

    pack_root: Path
    package_root: Path
    manifest_path: Path
    pyproject_path: Path
    scheme_path: Path
    readme_path: Path


def _validate_name(name: str) -> str:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
        raise ValueError(
            f"pack name {name!r} is not a valid python identifier (letters/digits/underscore, no leading digit)"
        )
    if keyword.iskeyword(name):
        raise ValueError(f"pack name {name!r} is a Python reserved word")
    return name


_PYPROJECT_TEMPLATE = """\
[project]
name = "{name}"
version = "0.1.0"
description = "CompGen extension pack: {name} ({kind})"
requires-python = ">=3.11"
dependencies = [
    "compgen",
]

[project.entry-points."compgen.packs"]
{name} = "{name}"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/{name}"]
"""

_MANIFEST_TEMPLATE = """\
name: {name}
version: "0.1.0"
kinds: ["{manifest_kind}"]
owned_surfaces:
  - {surface}
sealed_surfaces: []
generation_apertures:
  - {surface}
integration_mode: readonly
benchmark_suite: pack_integrations
benchmark_targets: []
reference_runner: ""
source_root: src/{name}
workspace_keys: []
third_party_names: []
expected_files:
  - README.md
  - src/{name}/__init__.py
  - src/{name}/{scheme_basename}
available_profilers: []
llvm_fork_key: ""
entry_module: ""
metadata:
  scaffold_kind: "{kind}"
"""

_PKG_INIT_TEMPLATE = '''\
"""CompGen extension pack: {name} ({kind}).

Discovery: CompGen resolves the ``compgen.packs`` entry point to this package
directory, which contains ``manifest.yaml``. See ``compgen.packs.load_pack``.
"""

from __future__ import annotations

from pathlib import Path

PACK_ROOT: Path = Path(__file__).resolve().parent
"""Directory containing manifest.yaml (used by CompGen to locate the pack)."""

__all__ = ["PACK_ROOT"]
'''

_README_TEMPLATE = """\
# {name}

CompGen extension pack scaffolded via `compgen scaffold-pack --kind {kind}`.

## Install

```bash
pip install -e .
```

After install, CompGen's pack discovery picks this up automatically via the
`compgen.packs` entry point declared in `pyproject.toml`.

## Layout

```
{name}/
├── pyproject.toml          # declares compgen.packs entry point
└── src/{name}/
    ├── __init__.py         # exposes PACK_ROOT
    ├── manifest.yaml       # CompGen pack manifest
    └── {scheme_basename}  # extension implementation (edit this)
```

## Extend

Edit `src/{name}/{scheme_basename}` to implement your {kind}. It was seeded
from the CompGen in-tree template for this extension point. See
`EXTENSION_POINTS.md` in the CompGen repo for the full protocol.

## Verify

```python
from compgen.packs import load_pack
loaded = load_pack("{name}")  # resolves entry point
print(loaded.manifest.name, loaded.manifest.kinds)
```
"""


def scaffold_pack(
    *,
    kind: str,
    name: str,
    out_dir: str | Path,
    repo_root: str | Path | None = None,
    overwrite: bool = False,
) -> ScaffoldResult:
    """Write a pip-installable extension-pack skeleton under ``out_dir``.

    Args:
        kind: One of :data:`SUPPORTED_KINDS`.
        name: Python-identifier name for the pack (becomes both the pip
            distribution name and the importable package).
        out_dir: Directory in which to create the pack root (``out_dir/<name>``).
        repo_root: CompGen repo root (used to locate ``_template.py``). Falls
            back to the installed compgen package location.
        overwrite: If True, replace an existing pack directory.

    Returns:
        Paths to each artifact written.
    """

    if kind not in _KIND_SPEC:
        raise ValueError(f"unknown pack kind {kind!r}; pick from {sorted(_KIND_SPEC)!r}")
    pkg_name = _validate_name(name)
    spec = _KIND_SPEC[kind]

    out_dir = Path(out_dir).expanduser().resolve()
    pack_root = out_dir / pkg_name
    if pack_root.exists():
        if not overwrite:
            raise FileExistsError(f"pack directory already exists: {pack_root}")
        shutil.rmtree(pack_root)

    # Full target packs need multi-surface emission (Backend + Provider
    # + MCP + HardwareSpec + tests) wired through four entry-point
    # groups. Hand off to the dedicated emitter.
    if kind == "target_pack":
        return _scaffold_target_pack(pkg_name=pkg_name, pack_root=pack_root)

    package_root = pack_root / "src" / pkg_name
    package_root.mkdir(parents=True, exist_ok=True)

    # pyproject.toml
    pyproject_path = pack_root / "pyproject.toml"
    pyproject_path.write_text(_PYPROJECT_TEMPLATE.format(name=pkg_name, kind=kind))

    # README
    readme_path = pack_root / "README.md"
    readme_path.write_text(_README_TEMPLATE.format(name=pkg_name, kind=kind, scheme_basename=spec["scheme_basename"]))

    # Python package __init__.py
    (package_root / "__init__.py").write_text(_PKG_INIT_TEMPLATE.format(name=pkg_name, kind=kind))

    # manifest.yaml (lives inside the package so pip install includes it)
    manifest_path = package_root / "manifest.yaml"
    manifest_path.write_text(
        _MANIFEST_TEMPLATE.format(
            name=pkg_name,
            kind=kind,
            manifest_kind=spec["manifest_kind"],
            surface=spec["surface"],
            scheme_basename=spec["scheme_basename"],
        )
    )

    # Seed the extension file from the repo template.
    scheme_path = package_root / spec["scheme_basename"]
    template_src = _resolve_template_source(spec["template_src"], repo_root=repo_root)
    if template_src is None:
        scheme_path.write_text(_fallback_scheme(kind=kind))
    else:
        scheme_path.write_text(template_src.read_text())

    return ScaffoldResult(
        pack_root=pack_root,
        package_root=package_root,
        manifest_path=manifest_path,
        pyproject_path=pyproject_path,
        scheme_path=scheme_path,
        readme_path=readme_path,
    )


def _resolve_template_source(rel_path: str, *, repo_root: str | Path | None) -> Path | None:
    """Locate a repo template file, searching explicit repo root then install."""

    if repo_root is not None:
        candidate = Path(repo_root).expanduser().resolve() / rel_path
        if candidate.exists():
            return candidate
    # Repo-relative (Path(__file__).parents[3] == repo root when running from src layout)
    candidate = _REPO_ROOT / rel_path
    if candidate.exists():
        return candidate
    return None


def _fallback_scheme(*, kind: str) -> str:
    return f'"""Fallback scaffold for {kind} extension. Replace with your implementation."""\n'


# ---------------------------------------------------------------------------
# target_pack scaffolding (multi-surface) — see :func:`_scaffold_target_pack`
# ---------------------------------------------------------------------------


_TARGET_PACK_PYPROJECT = """\
[project]
name = "{name}"
version = "0.1.0"
description = "CompGen target pack: {name}"
requires-python = ">=3.11"
dependencies = [
    "compgen",
]

[project.optional-dependencies]
dev = ["pytest>=7.0"]

# Pack discovery — manifest.yaml lookup.
[project.entry-points."compgen.packs"]
{name} = "{name}"

# Stage-pipeline backend.
[project.entry-points."compgen.targets.backends"]
{name} = "{name}.backend:{class_name}Backend"

# Kernel provider — codegen-fallback dispatcher hands contracts to this.
[project.entry-points."compgen.kernels.providers"]
{name} = "{name}.kernels:{class_name}Provider"

# Pack-owned MCP tools.
[project.entry-points."compgen.mcp.tools"]
{name} = "{name}.mcp:{upper_name}_TOOLS"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/{name}"]
include = [
    "src/{name}/specs/*.yaml",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
"""


_TARGET_PACK_README = """\
# {name}

CompGen target pack scaffolded via `compgen scaffold-pack --kind target_pack`.

## Install

```bash
pip install -e .
```

After install, four entry-point groups expose this pack to CompGen:

- `compgen.packs` — manifest discovery
- `compgen.targets.backends` — `{class_name}Backend`
- `compgen.kernels.providers` — `{class_name}Provider`
- `compgen.mcp.tools` — `{upper_name}_TOOLS`

## What to fill in

| File | What it does |
|---|---|
| `src/{name}/specs/{name}.yaml` | HardwareSpec describing the target — platform, ISA, memory model, verification surface. |
| `src/{name}/backend.py` | `{class_name}Backend` — `TargetBackendProtocol` impl. Drives stage-pipeline lowering. |
| `src/{name}/kernels.py` | `{class_name}Provider` — `KernelProvider` impl. `accepts_contract()` + `search()` decide which contracts your target handles + emit kernel source. |
| `src/{name}/mcp.py` | Pack-owned MCP verbs (e.g. `{name}_compile_and_run`). |
| `tests/test_pack_smoke.py` | Smoke test: pack importable, entry points reachable, Provider instantiable. |

## Verify

```bash
pytest tests/ -v
```

Then exercise via CompGen:

```python
from compgen.api import device, compile_model
import torch, torch.nn as nn

class M(nn.Module):
    def forward(self, a, b): return a + b

dev = device("src/{name}/specs/{name}.yaml")
cm = compile_model(M().eval(), dev, sample_inputs=(torch.randn(4), torch.randn(4)), verify=False)
```

If your `{class_name}Provider.accepts_contract` returns True for the
extracted op_family, CompGen writes the emitted source into
`<bundle>/generated_kernels/{name}/`.
"""


_TARGET_PACK_INIT = '''\
"""CompGen target pack: {name}."""

from __future__ import annotations

from pathlib import Path

PACK_ROOT: Path = Path(__file__).resolve().parent

__all__ = ["PACK_ROOT"]
'''


_TARGET_PACK_MANIFEST = """\
name: {name}
version: "0.1.0"
kinds: ["TargetPack"]
owned_surfaces:
  - target_backend
  - kernel_providers
sealed_surfaces: []
generation_apertures:
  - kernel_providers
integration_mode: readonly
benchmark_suite: pack_integrations
benchmark_targets: []
reference_runner: ""
source_root: src/{name}
workspace_keys: []
third_party_names: []
expected_files:
  - README.md
  - src/{name}/__init__.py
  - src/{name}/backend.py
  - src/{name}/kernels.py
  - src/{name}/mcp.py
  - src/{name}/specs/{name}.yaml
available_profilers: []
llvm_fork_key: ""
entry_module: ""
metadata:
  scaffold_kind: "target_pack"
"""


_TARGET_PACK_BACKEND = '''\
"""Stage-pipeline backend for the {name} target."""

from __future__ import annotations

from typing import Any


class {class_name}Backend:
    """Skeleton ``TargetBackendProtocol`` implementation.

    Fill in stage-pipeline lowering for {name}. Until you do, every method
    returns a benign default so import-time entry-point validation passes.
    """

    name: str = "{name}"

    def supports_target(self, target: Any) -> bool:
        # TODO: return True iff `target` is a HardwareSpec/TargetProfile this
        # backend can lower for. Inspect target.platform.family or .isa.base_isa.
        return False

    def get_options(self) -> dict[str, Any]:
        return {{}}

    def get_compilation_stages(self) -> list[str]:
        # Default chain — adjust to your target. The codegen stage's
        # KernelProvider dispatch is independent of this list.
        return ["encoding", "dispatch", "tiling", "codegen", "bundle"]

    def compile_stage(self, stage_name: str, module: Any, options: Any) -> Any:
        # TODO: implement per-stage lowering. Returning the module unchanged
        # is a no-op pass-through; replace with real transforms.
        return module

    def validate(self, module: Any) -> bool:
        return True


__all__ = ["{class_name}Backend"]
'''


_TARGET_PACK_PROVIDER = '''\
"""Kernel provider for the {name} target.

CompGen's codegen-fallback dispatcher (``compgen.kernels.codegen_fallback``)
walks registered providers and asks each whether it accepts a given
:class:`~compgen.kernels.provider.KernelContract`. The first provider that
returns True is asked to emit a kernel via :meth:`search`.
"""

from __future__ import annotations

from compgen.kernels.provider import (
    KernelContract,
    KnowledgeExport,
    ProviderResult,
    SearchBudget,
)


class {class_name}Provider:
    """Skeleton kernel provider — declines every contract by default.

    To make this provider useful:

    1. Narrow ``accepts_contract`` to the op_family / shape / dtype combos
       this target supports. ``contract.op_family`` is the cleaned-up
       op name (``"add"``, ``"mul"``, ``"matmul"``, ``"relu"``, …).
    2. Implement ``search`` to render real source for the contract — the
       returned ``ProviderResult.kernel_code`` is what CompGen writes to
       ``<bundle>/generated_kernels/{name}/<op>.<ext>``.
    """

    name: str = "{name}"

    def accepts_contract(self, contract: KernelContract) -> bool:
        # TODO: return True for op_families this target handles.
        # Examples (uncomment + extend):
        # if contract.op_family in {{"add", "mul", "sub", "relu"}}:
        #     return contract.dtypes == ("f32",)
        return False

    def search(self, contract: KernelContract, budget: SearchBudget) -> ProviderResult:  # noqa: ARG002
        # TODO: render real source for `contract`. The contract carries
        # op_family, input_shapes, output_shapes, dtypes, target_name,
        # hardware_key, and any caller-supplied provider_hints.
        return ProviderResult(
            found=False,
            kernel_code="",
            language="",
            metadata={{"reason": "not_implemented"}},
        )

    def export_knowledge(self) -> list[KnowledgeExport]:
        # Optional: return whatever the provider learned during search
        # (successful schedules, hardware quirks, failure modes) so
        # CompGen's compiler-memory can persist it across sessions.
        return []


__all__ = ["{class_name}Provider"]
'''


_TARGET_PACK_MCP = '''\
"""Pack-owned MCP tools for {name}.

Each entry in ``{upper_name}_TOOLS`` is a tool dict with the same shape
as in-tree CompGen tools (see ``compgen.mcp.tools.lifecycle``). The
``compgen.mcp.tools`` entry-point group surfaces these into
``compgen.mcp.tools.ALL_TOOLS`` automatically; ``compgen mcp tools``
will list them with a ``[pack: {name}]`` annotation.
"""

from __future__ import annotations

from typing import Any


def _example_handler(*, message: str = "hello") -> dict[str, Any]:
    """Stub handler. Replace this whole tool with whatever your pack exposes."""
    return {{"ok": True, "echo": message}}


# Add tool dicts here. Empty by default — keeping the entry point loaded
# without exposing any verbs is fine; the validator accepts an empty list.
{upper_name}_TOOLS: list[dict[str, Any]] = [
    # Example — uncomment + adapt:
    # {{
    #     "name": "{name}_compile_and_run",
    #     "description": "Compile a model for {name} and run on RTL/sim/HW.",
    #     "phase": "job",
    #     "handler": _example_handler,
    #     "input_schema": {{"type": "object", "properties": {{"message": {{"type": "string"}}}}}},
    # }},
]


__all__ = ["{upper_name}_TOOLS"]
'''


_TARGET_PACK_HW_SPEC = """\
# HardwareSpec for {name}. Fields below are the minimal shape CompGen's
# ``compgen.targetgen.load.load_hardware_spec`` accepts. Fill in real
# values for your target — see the schema in
# ``python/compgen/targetgen/hardware_spec.py``.

name: {name}
schema_version: "2.0"

platform:
  vendor: ""
  family: ""
  chip_name: ""
  host_arch: "x86_64"
  toolchain: ""
  deployment_model: "linux_userspace"

execution_model:
  model: simd_vector  # one of: simt_gpu, simd_vector, decoupled_matrix, …

isa:
  base_isa: "unknown"
  extensions: []

native_ops:
  families: []

engine_geometry:
  vector_length_bits: 0
  max_warp_size: 0

memory_model:
  address_spaces: []

numeric_contract:
  supported_dtypes:
    - name: f32
      native: true

runtime_contract:
  calling_convention: c_abi

verification_surface:
  has_simulator: false
  # If your target has a Chipyard / RTL flow:
  #   simulator_command: "make -C {{chipyard_root}}/sims/vcs run-binary CONFIG={{config}} {{extra_make_args}} BINARY={{elf}}"
  #   build_command: ""  # empty = simulator_command does its own build
"""


_TARGET_PACK_TESTS_INIT = ""


_TARGET_PACK_SMOKE_TEST = '''\
"""Smoke test: pack imports + entry points reachable + Provider instantiable.

Run with::

    pip install -e .
    pytest tests/ -v
"""

from __future__ import annotations

import importlib.metadata as im


def test_pack_module_imports():
    import {name}  # noqa: F401
    from {name} import PACK_ROOT
    assert (PACK_ROOT / "manifest.yaml").exists()


def test_backend_class_resolves():
    from {name}.backend import {class_name}Backend
    assert {class_name}Backend().name == "{name}"


def test_provider_class_resolves_and_validates():
    from compgen.kernels.provider import KernelProvider
    from {name}.kernels import {class_name}Provider
    p = {class_name}Provider()
    # Stub provider declines every contract until the user implements it.
    assert p.name == "{name}"
    assert isinstance(p, KernelProvider)


def test_mcp_tools_list_is_valid():
    from {name}.mcp import {upper_name}_TOOLS
    assert isinstance({upper_name}_TOOLS, list)
    for tool in {upper_name}_TOOLS:
        for key in ("name", "description", "input_schema", "handler", "phase"):
            assert key in tool, f"tool {{tool!r}} missing key {{key!r}}"


def test_entry_points_declared_under_compgen_groups():
    """All four CompGen entry points point at this pack's modules."""
    expected = {{
        "compgen.packs": "{name}",
        "compgen.targets.backends": "{name}.backend:{class_name}Backend",
        "compgen.kernels.providers": "{name}.kernels:{class_name}Provider",
        "compgen.mcp.tools": "{name}.mcp:{upper_name}_TOOLS",
    }}
    for group, value in expected.items():
        eps = im.entry_points(group=group)
        names = {{ep.name: ep.value for ep in eps}}
        assert "{name}" in names, f"no {{group!r}} entry for pack"
        assert names["{name}"] == value, (
            f"{{group!r}} entry-point value mismatch: got {{names['{name}']!r}}, expected {{value!r}}"
        )
'''


def _scaffold_target_pack(*, pkg_name: str, pack_root: Path) -> ScaffoldResult:
    """Emit a full target-pack skeleton — radiance-shaped."""
    class_name = "".join(part.capitalize() for part in pkg_name.split("_"))
    upper_name = pkg_name.upper()

    package_root = pack_root / "src" / pkg_name
    package_root.mkdir(parents=True, exist_ok=True)
    (package_root / "specs").mkdir(exist_ok=True)
    tests_root = pack_root / "tests"
    tests_root.mkdir(exist_ok=True)

    fmt = {
        "name": pkg_name,
        "class_name": class_name,
        "upper_name": upper_name,
    }

    pyproject_path = pack_root / "pyproject.toml"
    pyproject_path.write_text(_TARGET_PACK_PYPROJECT.format(**fmt))

    readme_path = pack_root / "README.md"
    readme_path.write_text(_TARGET_PACK_README.format(**fmt))

    (package_root / "__init__.py").write_text(_TARGET_PACK_INIT.format(**fmt))
    manifest_path = package_root / "manifest.yaml"
    manifest_path.write_text(_TARGET_PACK_MANIFEST.format(**fmt))

    backend_path = package_root / "backend.py"
    backend_path.write_text(_TARGET_PACK_BACKEND.format(**fmt))
    (package_root / "kernels.py").write_text(_TARGET_PACK_PROVIDER.format(**fmt))
    (package_root / "mcp.py").write_text(_TARGET_PACK_MCP.format(**fmt))
    (package_root / "specs" / f"{pkg_name}.yaml").write_text(_TARGET_PACK_HW_SPEC.format(**fmt))

    (tests_root / "__init__.py").write_text(_TARGET_PACK_TESTS_INIT)
    (tests_root / "test_pack_smoke.py").write_text(_TARGET_PACK_SMOKE_TEST.format(**fmt))

    return ScaffoldResult(
        pack_root=pack_root,
        package_root=package_root,
        manifest_path=manifest_path,
        pyproject_path=pyproject_path,
        scheme_path=backend_path,
        readme_path=readme_path,
    )
