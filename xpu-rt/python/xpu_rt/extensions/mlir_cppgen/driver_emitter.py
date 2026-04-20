"""Emit the compgen-opt driver source file.

Generates ``compgen-opt.cpp`` — the main entry point for the generated
MLIR optimizer tool.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from compgen.extensions.mlir_cppgen.introspect import DialectInfo

_TEMPLATE_DIR = Path(__file__).parent / "templates"


def _make_env() -> Environment:
    return Environment(
        loader=FileSystemLoader(str(_TEMPLATE_DIR)),
        keep_trailing_newline=True,
        trim_blocks=True,
        lstrip_blocks=True,
    )


@dataclass
class DriverDialectEntry:
    """Simplified dialect info for the driver template."""

    prefix: str
    cpp_namespace: str
    dialect_class: str
    has_attrs: bool
    has_ops: bool


def emit_opt_driver(
    dialects: list[DialectInfo],
    *,
    dialects_with_passes: set[str] | None = None,
) -> str:
    """Generate compgen-opt.cpp content."""
    env = _make_env()
    tmpl = env.get_template("opt_driver.j2")
    dialects_with_passes = dialects_with_passes or set()

    entries = []
    for d in dialects:
        entries.append(
            DriverDialectEntry(
                prefix=d.prefix,
                cpp_namespace=d.cpp_namespace,
                dialect_class=f"{d.prefix}Dialect",
                has_attrs=bool(d.attrs),
                has_ops=bool(d.ops),
            )
        )

    pass_prefixes = sorted(dialects_with_passes)
    return tmpl.render(dialects=entries, pass_prefixes=pass_prefixes)


def write_opt_driver(
    dialects: list[DialectInfo],
    opt_dir: Path,
    *,
    dialects_with_passes: set[str] | None = None,
) -> Path:
    """Write compgen-opt.cpp to the given directory.

    Args:
        dialects: All dialects to register.
        opt_dir: The ``compgen-opt/`` directory.
        dialects_with_passes: Set of dialect prefixes that have passes.

    Returns:
        Path to the written file.
    """
    opt_dir.mkdir(parents=True, exist_ok=True)
    path = opt_dir / "compgen-opt.cpp"
    path.write_text(emit_opt_driver(dialects, dialects_with_passes=dialects_with_passes or set()))
    return path


__all__ = [
    "DriverDialectEntry",
    "emit_opt_driver",
    "write_opt_driver",
]
