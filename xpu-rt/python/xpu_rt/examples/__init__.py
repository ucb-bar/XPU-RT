"""Packaged demo models and hardware specs for CompGen.

These are installable artefacts that ship with the ``compgen`` wheel so
downstream users — who have no access to the source tree — can exercise
the end-to-end compilation flow by name.

Demos live in sub-packages (e.g. ``compgen.examples.saturn_opu_convnet``)
exposing :func:`build_model` and :func:`default_inputs`. Hardware specs
live under ``compgen.examples.hardware_specs`` as YAML resources.

Use :func:`resolve_demo_module` to obtain the dotted module name for a
demo (pass to tools that load by dotted path), and :func:`resolve_spec_path`
for the filesystem path of a shipped spec (pass to tools that read a
YAML path). Both raise ``ValueError`` on unknown names.
"""

from __future__ import annotations

from importlib import resources
from importlib.util import find_spec
from pathlib import Path

_DEMO_PACKAGE = "compgen.examples"
_SPEC_PACKAGE = "compgen.examples.hardware_specs"
_TARGET_PROFILE_PACKAGE = "compgen.examples.target_profiles"


def list_demos() -> list[str]:
    """Return names of shipped demo models (sub-packages under ``compgen.examples``)."""
    pkg = resources.files(_DEMO_PACKAGE)
    out: list[str] = []
    for entry in pkg.iterdir():
        if not entry.is_dir():
            continue
        name = entry.name
        if name.startswith("_") or name == "hardware_specs":
            continue
        if find_spec(f"{_DEMO_PACKAGE}.{name}.model") is not None:
            out.append(name)
    return sorted(out)


def list_specs() -> list[str]:
    """Return names (without ``.yaml``) of shipped hardware specs (schema v2.0)."""
    pkg = resources.files(_SPEC_PACKAGE)
    return sorted(entry.name[:-5] for entry in pkg.iterdir() if entry.is_file() and entry.name.endswith(".yaml"))


def list_target_profiles() -> list[str]:
    """Return names (without ``.yaml``) of shipped declarative target profiles."""
    pkg = resources.files(_TARGET_PROFILE_PACKAGE)
    return sorted(entry.name[:-5] for entry in pkg.iterdir() if entry.is_file() and entry.name.endswith(".yaml"))


def resolve_demo_module(name: str) -> str:
    """Return the dotted module path for a shipped demo's ``model`` module."""
    if find_spec(f"{_DEMO_PACKAGE}.{name}.model") is None:
        raise ValueError(f"unknown demo '{name}'; available: {', '.join(list_demos()) or '(none)'}")
    return f"{_DEMO_PACKAGE}.{name}.model"


def resolve_spec_path(name: str) -> Path:
    """Return the filesystem path of a shipped hardware spec YAML."""
    filename = name if name.endswith(".yaml") else f"{name}.yaml"
    resource = resources.files(_SPEC_PACKAGE).joinpath(filename)
    if not resource.is_file():
        raise ValueError(f"unknown spec '{name}'; available: {', '.join(list_specs()) or '(none)'}")
    return Path(str(resource))


def resolve_target_profile_path(name: str) -> Path:
    """Return the filesystem path of a shipped declarative target profile YAML."""
    filename = name if name.endswith(".yaml") else f"{name}.yaml"
    resource = resources.files(_TARGET_PROFILE_PACKAGE).joinpath(filename)
    if not resource.is_file():
        raise ValueError(f"unknown target profile '{name}'; available: {', '.join(list_target_profiles()) or '(none)'}")
    return Path(str(resource))


__all__ = [
    "list_demos",
    "list_specs",
    "list_target_profiles",
    "resolve_demo_module",
    "resolve_spec_path",
    "resolve_target_profile_path",
]
