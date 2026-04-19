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
        raise ValueError(
            f"unknown pack kind {kind!r}; pick from {sorted(_KIND_SPEC)!r}"
        )
    pkg_name = _validate_name(name)
    spec = _KIND_SPEC[kind]

    out_dir = Path(out_dir).expanduser().resolve()
    pack_root = out_dir / pkg_name
    if pack_root.exists():
        if not overwrite:
            raise FileExistsError(f"pack directory already exists: {pack_root}")
        shutil.rmtree(pack_root)

    package_root = pack_root / "src" / pkg_name
    package_root.mkdir(parents=True, exist_ok=True)

    # pyproject.toml
    pyproject_path = pack_root / "pyproject.toml"
    pyproject_path.write_text(
        _PYPROJECT_TEMPLATE.format(name=pkg_name, kind=kind)
    )

    # README
    readme_path = pack_root / "README.md"
    readme_path.write_text(
        _README_TEMPLATE.format(
            name=pkg_name, kind=kind, scheme_basename=spec["scheme_basename"]
        )
    )

    # Python package __init__.py
    (package_root / "__init__.py").write_text(
        _PKG_INIT_TEMPLATE.format(name=pkg_name, kind=kind)
    )

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


def _resolve_template_source(
    rel_path: str, *, repo_root: str | Path | None
) -> Path | None:
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
    return (
        f'"""Fallback scaffold for {kind} extension. '
        f"Replace with your implementation.\"\"\"\n"
    )
