"""Emit TableGen (.td) files from introspected DialectInfo.

Generates:
  - {Prefix}Dialect.td — dialect definition + base op class
  - {Prefix}Attrs.td — custom attribute definitions
  - {Prefix}Ops.td — operation definitions
"""

from __future__ import annotations

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


def emit_dialect_td(info: DialectInfo) -> str:
    """Generate {Prefix}Dialect.td content."""
    env = _make_env()
    tmpl = env.get_template("dialect_td.j2")
    return tmpl.render(
        prefix=info.prefix,
        dialect_name=info.name,
        cpp_namespace=info.cpp_namespace,
        has_attrs=bool(info.attrs),
    )


def emit_attrs_td(info: DialectInfo, dep_includes: list[str] | None = None) -> str:
    """Generate {Prefix}Attrs.td content."""
    if not info.attrs:
        return ""
    env = _make_env()
    tmpl = env.get_template("attrs_td.j2")
    return tmpl.render(
        prefix=info.prefix,
        attrs=info.attrs,
        dep_includes=dep_includes or [],
    )


def emit_ops_td(info: DialectInfo, dep_includes: list[str] | None = None) -> str:
    """Generate {Prefix}Ops.td content."""
    if not info.ops:
        return ""
    env = _make_env()
    tmpl = env.get_template("ops_td.j2")
    return tmpl.render(
        prefix=info.prefix,
        dialect_name=info.name,
        ops=info.ops,
        has_attrs=bool(info.attrs),
        dep_includes=dep_includes or [],
    )


def write_tablegen_files(
    info: DialectInfo,
    include_dir: Path,
    *,
    dep_includes: list[str] | None = None,
) -> list[Path]:
    """Write all TableGen files for a dialect to the include directory.

    Args:
        info: Introspected dialect info.
        include_dir: The ``include/{Prefix}/`` directory.
        dep_includes: Additional TableGen include paths for dependencies.

    Returns:
        List of written file paths.
    """
    include_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    # Dialect.td
    dialect_path = include_dir / f"{info.prefix}Dialect.td"
    dialect_path.write_text(emit_dialect_td(info))
    written.append(dialect_path)

    # Attrs.td
    if info.attrs:
        attrs_path = include_dir / f"{info.prefix}Attrs.td"
        attrs_path.write_text(emit_attrs_td(info, dep_includes))
        written.append(attrs_path)

    # Ops.td
    if info.ops:
        ops_path = include_dir / f"{info.prefix}Ops.td"
        ops_path.write_text(emit_ops_td(info, dep_includes))
        written.append(ops_path)

    return written


__all__ = [
    "emit_attrs_td",
    "emit_dialect_td",
    "emit_ops_td",
    "write_tablegen_files",
]
