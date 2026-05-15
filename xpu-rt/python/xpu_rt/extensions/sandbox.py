"""Path-enforcement sandbox for extensions.

Every write requested by an extension passes through
:func:`validate_sandboxed_path`. The function resolves the path
against the extension's ``allowed_write_root`` and raises
:class:`ExtensionSandboxViolation` when:

- the path escapes ``allowed_write_root`` via ``..`` traversal;
- the path lands under a forbidden compiler-owned root
  (``python/compgen/``, ``configs/``, manifests, run ledgers);
- the path targets a contract file or canonical IR artifact
  (``payload.mlir`` etc.).

The check is purely structural — no filesystem effects. Callers are
responsible for never opening a path that fails validation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

from compgen.extensions.errors import ExtensionSandboxViolation

# Forbidden path *segments*. A write whose resolved path contains any of
# these as a segment (after normalization) is rejected — covers the case
# where the extension's ``allowed_write_root`` itself sits next to a
# forbidden tree.
_FORBIDDEN_PATH_SEGMENTS: Final[tuple[str, ...]] = (
    "python/compgen",
    "configs/targets",
    "configs/models",
    "results/audit",
    ".compgen",
)

# Forbidden filename suffixes / exact names. Treated as fatal regardless
# of directory — these are the canonical compiler artifacts an extension
# must never overwrite.
_FORBIDDEN_NAMES: Final[frozenset[str]] = frozenset(
    {
        "payload.mlir",
        "execution_plan.yaml",
        "memory_plan.yaml",
        "compgen_extension.yaml",
        "kernel_contract.yaml",
        "run_manifest.json",
        "manifest.json",
    }
)


def _resolve(path: Path) -> Path:
    # ``Path.resolve(strict=False)`` collapses ``..`` segments without
    # touching the filesystem.
    return path.expanduser().resolve(strict=False)


def validate_sandboxed_path(
    target_path: str | Path,
    *,
    allowed_write_root: str | Path,
) -> Path:
    """Resolve ``target_path`` and raise ``ExtensionSandboxViolation``
    if it escapes ``allowed_write_root`` or hits a forbidden segment.

    Returns the resolved :class:`Path` on success so callers can use
    it directly.
    """

    target = _resolve(Path(target_path))
    root = _resolve(Path(allowed_write_root))

    try:
        target.relative_to(root)
    except ValueError as exc:
        raise ExtensionSandboxViolation(
            path=str(target),
            allowed_root=str(root),
            reason="escapes_allowed_root",
        ) from exc

    target_str = target.as_posix()
    for segment in _FORBIDDEN_PATH_SEGMENTS:
        if f"/{segment}/" in target_str or target_str.endswith(f"/{segment}"):
            raise ExtensionSandboxViolation(
                path=str(target),
                allowed_root=str(root),
                reason=f"forbidden_segment:{segment}",
            )

    if target.name in _FORBIDDEN_NAMES:
        raise ExtensionSandboxViolation(
            path=str(target),
            allowed_root=str(root),
            reason=f"forbidden_filename:{target.name}",
        )

    return target


def is_under_sandbox(target_path: str | Path, allowed_write_root: str | Path) -> bool:
    """Non-raising version of :func:`validate_sandboxed_path`."""

    try:
        validate_sandboxed_path(target_path, allowed_write_root=allowed_write_root)
    except ExtensionSandboxViolation:
        return False
    return True
