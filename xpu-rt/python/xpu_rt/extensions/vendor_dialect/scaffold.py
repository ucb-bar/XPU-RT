"""Scaffold a user-space vendor-adapter package from a descriptor.

Rendering is deterministic and side-effect-bounded: given a descriptor
and an output directory, the scaffold engine writes a self-contained
pip-installable package under ``<out_dir>/<package_name>/``. The
templates are Jinja2 files under
:data:`TEMPLATE_PACK_ROOT`; they are copied verbatim when no template
variables are present, rendered otherwise.

The scaffolded package:

* Declares a ``compgen.vendor_dialects`` entry point so
  :func:`compgen.extensions.vendor_dialect.registry.get_adapter` finds it
  after ``pip install -e .``.
* Carries a frozen ``descriptor.yaml`` alongside the Python modules so
  the adapter, kernel provider, and bundle step always have access to
  the spec that produced them.
* Ships a smoke test that imports the package and inspects the
  descriptor — enough to catch scaffold regressions without requiring
  the vendor toolchain.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog
from jinja2 import Environment, FileSystemLoader, StrictUndefined, select_autoescape

from compgen.extensions.vendor_dialect.descriptor import VendorDialectDescriptor

log = structlog.get_logger()


TEMPLATE_PACK_ROOT = Path(__file__).resolve().parent / "templates" / "user_package"


@dataclass
class ScaffoldResult:
    """Return value of :func:`scaffold_package`."""

    package_dir: Path
    files_written: list[Path] = field(default_factory=list)
    descriptor_path: Path = field(default_factory=lambda: Path())


def scaffold_package(
    descriptor: VendorDialectDescriptor,
    out_dir: str | Path,
    *,
    overwrite: bool = False,
    template_pack: str | Path = TEMPLATE_PACK_ROOT,
) -> ScaffoldResult:
    """Render a user-space adapter package.

    Args:
        descriptor: The reviewed :class:`VendorDialectDescriptor`. Its
            ``package_name`` is the name of the subdirectory that will
            hold the Python package inside ``out_dir``.
        out_dir: Parent directory for the generated package. The
            package itself is written to ``out_dir/<package_name>/``.
        overwrite: If ``True``, wipe the target package directory before
            rendering. Defaults to ``False`` to protect user edits.
        template_pack: Root of the template pack. Defaults to the bundled
            pack; tests and custom users may override.

    Returns:
        :class:`ScaffoldResult` listing written files.

    Raises:
        FileExistsError: If the target package directory exists and
            ``overwrite`` is False.
        FileNotFoundError: If ``template_pack`` does not exist.
    """
    tpack = Path(template_pack)
    if not tpack.is_dir():
        raise FileNotFoundError(f"template pack not found: {tpack}")

    out = Path(out_dir).expanduser().resolve()
    pkg_root = out / descriptor.package_name
    if pkg_root.exists():
        if not overwrite:
            raise FileExistsError(f"{pkg_root} already exists; pass overwrite=True to replace")
        shutil.rmtree(pkg_root)
    pkg_root.mkdir(parents=True, exist_ok=False)

    env = Environment(
        loader=FileSystemLoader(str(tpack)),
        undefined=StrictUndefined,
        autoescape=select_autoescape(enabled_extensions=[]),
        keep_trailing_newline=True,
    )
    ctx = _render_context(descriptor)

    written: list[Path] = []
    for tmpl_path in sorted(tpack.rglob("*")):
        if not tmpl_path.is_file():
            continue
        rel = tmpl_path.relative_to(tpack)
        rel_str = str(rel)
        # Rewrite ``compgen_pkg`` placeholder dir to the real package name.
        target_rel = rel_str.replace("compgen_pkg", descriptor.package_name)
        # Strip the ``.j2`` suffix used to mark rendered files.
        if target_rel.endswith(".j2"):
            target_rel = target_rel[: -len(".j2")]
            rendered = env.get_template(rel_str).render(**ctx)
            dest = pkg_root / target_rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(rendered)
        else:
            dest = pkg_root / target_rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(tmpl_path, dest)
        written.append(dest)

    # Always emit the frozen descriptor next to the adapter module.
    desc_dest = pkg_root / descriptor.package_name / "descriptor.yaml"
    desc_dest.parent.mkdir(parents=True, exist_ok=True)
    desc_dest.write_text(descriptor.to_yaml())
    written.append(desc_dest)

    log.info(
        "vendor_scaffold.done",
        package=descriptor.package_name,
        out=str(pkg_root),
        files=len(written),
    )
    return ScaffoldResult(
        package_dir=pkg_root,
        files_written=written,
        descriptor_path=desc_dest,
    )


# --------------------------------------------------------------------------- #
# Internals
# --------------------------------------------------------------------------- #


def _render_context(descriptor: VendorDialectDescriptor) -> dict[str, Any]:
    """Shape the descriptor into a Jinja-friendly context dict."""
    adapter_cls = "".join(part.capitalize() for part in descriptor.name.split("_")) + "Adapter"
    return {
        "descriptor": descriptor,
        "package_name": descriptor.package_name,
        "vendor_name": descriptor.name,
        "adapter_class_name": adapter_cls,
        "target": descriptor.target,
        "repo_path": descriptor.repo_path,
        "input_ir": list(descriptor.input_ir),
        "output_format": descriptor.output_format or descriptor.bundle.output_format,
        "kernel_authoring_required": descriptor.kernel_authoring_required,
        "lowering_mode": descriptor.lowering.mode,
        "op_families": list(descriptor.lowering.op_families),
        "template_ops": list(descriptor.lowering.template_ops),
        "bundle_steps": list(descriptor.bundle.steps),
        "runtime_entry": descriptor.bundle.runtime_entry,
        "cli_tools": list(descriptor.compile_entry.cli_tools),
        "python_module": descriptor.compile_entry.python_module,
        "license": descriptor.license,
        "dependencies": list(descriptor.dependencies),
    }


__all__ = ["ScaffoldResult", "TEMPLATE_PACK_ROOT", "scaffold_package"]
