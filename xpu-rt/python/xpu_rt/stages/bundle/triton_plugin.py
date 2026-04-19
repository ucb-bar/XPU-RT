"""Bundle-stage plugin: triton_friendly targets get buildable Triton .py kernels.

Parallel to :mod:`baremetal_plugin`. Walks the post-recipe payload
IR, annotates every Triton-eligible op with
``compgen.library_dispatch="triton"``, then runs the existing
:func:`compgen.runtime.triton_emitter.emit_triton_kernels` which
writes ``kernels/compgen_*.py`` plus an ``emission_manifest.json``.

The annotation step is a no-op when ops already carry the attribute
(e.g. set by an earlier stage plugin), so re-running is idempotent.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import structlog
from xdsl.dialects.builtin import ModuleOp, StringAttr
from xdsl.ir import Operation

from compgen.runtime.triton_emitter import emit_triton_kernels

log = structlog.get_logger()


_TRITON_ELIGIBLE_OP_NAMES: frozenset[str] = frozenset({
    "linalg.matmul",
    "linalg.batch_matmul",
    "linalg.softmax",
    "compgen.linalg_ext.softmax",
})


@dataclass(frozen=True)
class TritonBundleResult:
    output_dir: Path
    kernel_files: list[Path]
    manifest_path: Path
    kernels_emitted: int
    skipped: int


def _ensure_dispatch_attr(module: ModuleOp) -> int:
    """Set ``compgen.library_dispatch="triton"`` on every eligible op.

    Returns the number of ops newly annotated.
    """
    added = 0
    for op in module.walk():
        if op.name not in _TRITON_ELIGIBLE_OP_NAMES:
            continue
        if "compgen.library_dispatch" in op.attributes:
            continue
        op.attributes["compgen.library_dispatch"] = StringAttr("triton")
        added += 1
    return added


def write_triton_bundle(
    module: ModuleOp,
    output_dir: Path,
) -> TritonBundleResult:
    """Emit Triton .py kernel files for ``module`` under ``output_dir``.

    Safe to call on modules with no Triton-eligible ops — the result
    will just have ``kernels_emitted = 0``.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _ensure_dispatch_attr(module)
    report = emit_triton_kernels(module, out_dir=output_dir)

    kernels_dir = output_dir / "kernels"
    kernel_files = (
        sorted(kernels_dir.glob("*.py")) if kernels_dir.exists() else []
    )
    return TritonBundleResult(
        output_dir=output_dir,
        kernel_files=kernel_files,
        manifest_path=output_dir / "emission_manifest.json",
        kernels_emitted=report.kernels_emitted,
        skipped=report.skipped_no_dispatch_tag + report.skipped_unsupported_op,
    )


__all__ = [
    "TritonBundleResult",
    "write_triton_bundle",
]
