"""Emit C++ header and source files from introspected DialectInfo.

Generates:
  - {Prefix}Dialect.h/cpp — dialect initialization
  - {Prefix}Attrs.h/cpp — custom attribute classes
  - {Prefix}Ops.h/cpp — operation classes + verifiers
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


def _dialect_class_name(info: DialectInfo) -> str:
    """Build the C++ dialect class name (e.g., 'LayoutDialect')."""
    return f"{info.prefix}Dialect"


def emit_dialect_h(info: DialectInfo) -> str:
    """Generate {Prefix}Dialect.h content."""
    env = _make_env()
    tmpl = env.get_template("dialect_h.j2")
    return tmpl.render(prefix=info.prefix)


def emit_dialect_cpp(
    info: DialectInfo,
    dep_headers: list[str] | None = None,
) -> str:
    """Generate {Prefix}Dialect.cpp content."""
    env = _make_env()
    tmpl = env.get_template("dialect_cpp.j2")
    return tmpl.render(
        prefix=info.prefix,
        cpp_namespace=info.cpp_namespace,
        dialect_class=_dialect_class_name(info),
        has_attrs=bool(info.attrs),
    )


def emit_attrs_h(
    info: DialectInfo,
    dep_headers: list[str] | None = None,
) -> str:
    """Generate {Prefix}Attrs.h content."""
    if not info.attrs:
        return ""
    env = _make_env()
    tmpl = env.get_template("attrs_h.j2")
    return tmpl.render(
        prefix=info.prefix,
        dep_headers=dep_headers or [],
    )


def emit_attrs_cpp(info: DialectInfo) -> str:
    """Generate {Prefix}Attrs.cpp content."""
    if not info.attrs:
        return ""
    env = _make_env()
    tmpl = env.get_template("attrs_cpp.j2")
    return tmpl.render(
        prefix=info.prefix,
        cpp_namespace=info.cpp_namespace,
    )


def emit_ops_h(
    info: DialectInfo,
    dep_headers: list[str] | None = None,
) -> str:
    """Generate {Prefix}Ops.h content."""
    if not info.ops:
        return ""
    env = _make_env()
    tmpl = env.get_template("ops_h.j2")
    return tmpl.render(
        prefix=info.prefix,
        has_attrs=bool(info.attrs),
        dep_headers=dep_headers or [],
    )


def emit_ops_cpp(info: DialectInfo) -> str:
    """Generate {Prefix}Ops.cpp content."""
    if not info.ops:
        return ""
    env = _make_env()
    tmpl = env.get_template("ops_cpp.j2")

    ops_with_verifier = [op for op in info.ops if op.verifier is not None]

    return tmpl.render(
        prefix=info.prefix,
        cpp_namespace=info.cpp_namespace,
        ops_with_verifier=ops_with_verifier,
    )


def write_header_files(
    info: DialectInfo,
    include_dir: Path,
    *,
    dep_headers: list[str] | None = None,
) -> list[Path]:
    """Write all C++ header files for a dialect.

    Args:
        info: Introspected dialect info.
        include_dir: The ``include/{Prefix}/`` directory.
        dep_headers: Additional #include paths for dependency headers.

    Returns:
        List of written file paths.
    """
    include_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    # Dialect.h
    path = include_dir / f"{info.prefix}Dialect.h"
    path.write_text(emit_dialect_h(info))
    written.append(path)

    # Attrs.h
    if info.attrs:
        path = include_dir / f"{info.prefix}Attrs.h"
        path.write_text(emit_attrs_h(info, dep_headers))
        written.append(path)

    # Ops.h
    if info.ops:
        path = include_dir / f"{info.prefix}Ops.h"
        path.write_text(emit_ops_h(info, dep_headers))
        written.append(path)

    return written


def write_source_files(
    info: DialectInfo,
    lib_dir: Path,
) -> list[Path]:
    """Write all C++ source files for a dialect.

    Args:
        info: Introspected dialect info.
        lib_dir: The ``lib/{Prefix}/`` directory.

    Returns:
        List of written file paths.
    """
    lib_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    # Dialect.cpp
    path = lib_dir / f"{info.prefix}Dialect.cpp"
    path.write_text(emit_dialect_cpp(info))
    written.append(path)

    # Attrs.cpp
    if info.attrs:
        path = lib_dir / f"{info.prefix}Attrs.cpp"
        path.write_text(emit_attrs_cpp(info))
        written.append(path)

    # Ops.cpp
    if info.ops:
        path = lib_dir / f"{info.prefix}Ops.cpp"
        path.write_text(emit_ops_cpp(info))
        written.append(path)

    return written


__all__ = [
    "emit_attrs_cpp",
    "emit_attrs_h",
    "emit_dialect_cpp",
    "emit_dialect_h",
    "emit_ops_cpp",
    "emit_ops_h",
    "write_header_files",
    "write_source_files",
]
