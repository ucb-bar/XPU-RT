"""Stable content-addressed hashing for capture/lower artifacts.

Two surfaces:

- :func:`sha256_file` — sha256 of a file's raw bytes. mtime / mode are
  ignored.
- :func:`sha256_tree` — content-addressed digest of a directory tree.
  Entries are sorted by POSIX-relative path; for each file we feed
  ``f"{relpath}\\0{sha256(file)}\\n"`` into the hasher. Directory
  structure is captured via the relpaths themselves.

A symlink that resolves outside the tree's root raises
:class:`SymlinkEscapeError` so the validator (R004) can flag it. This
matters because Section 16 runs are auditable artifacts: a symlink
pointing at ``/etc/passwd`` would otherwise let a tampered run pass a
naive hash check by reading attacker-chosen content.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

_CHUNK = 1 << 20  # 1 MiB


class SymlinkEscapeError(ValueError):
    """A symlink resolved outside the tree root."""


def sha256_file(path: Path) -> str:
    """Return lowercase-hex sha256 of ``path``'s bytes.

    Raises ``FileNotFoundError`` if the path does not exist or
    ``IsADirectoryError`` if it is a directory.
    """
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(_CHUNK)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _resolve_inside(root: Path, candidate: Path) -> Path:
    """Resolve ``candidate`` and confirm it stays inside ``root``."""
    root_resolved = root.resolve(strict=True)
    candidate_resolved = candidate.resolve(strict=True)
    try:
        candidate_resolved.relative_to(root_resolved)
    except ValueError as exc:
        raise SymlinkEscapeError(
            f"path {candidate} resolves to {candidate_resolved}, outside root {root_resolved}"
        ) from exc
    return candidate_resolved


def sha256_tree(root: Path) -> str:
    """Return a stable content-addressed sha256 over a directory tree.

    The tree is walked deterministically (sorted relpaths). Each file
    contributes ``f"{relpath}\\0{file_sha}\\n"`` to the hasher. Empty
    directories are ignored — the hash is over file content, not the
    on-disk inode structure.

    Symlinks are followed only inside ``root``; outside-root symlinks
    raise :class:`SymlinkEscapeError`.
    """
    if not root.exists():
        raise FileNotFoundError(f"tree root does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"tree root is not a directory: {root}")

    root_resolved = root.resolve(strict=True)
    files: list[tuple[str, Path]] = []
    for p in root.rglob("*"):
        # rglob returns symlinks as-is; we resolve below.
        if p.is_symlink():
            target = _resolve_inside(root_resolved, p)
            if target.is_dir():
                # Symlinked directories: walk them via the resolved path
                # but record relpaths under the original tree.
                for sub in target.rglob("*"):
                    if sub.is_file():
                        rel = p.relative_to(root) / sub.relative_to(target)
                        files.append((rel.as_posix(), sub))
                continue
            if target.is_file():
                rel = p.relative_to(root)
                files.append((rel.as_posix(), target))
                continue
            # Anything else (broken symlink etc.) is a contract violation.
            raise SymlinkEscapeError(f"symlink {p} resolves to non-file/dir target {target}")
        if p.is_file():
            rel = p.relative_to(root)
            files.append((rel.as_posix(), p))

    files.sort(key=lambda t: t[0])
    h = hashlib.sha256()
    for entry_rel, fpath in files:
        file_hash = sha256_file(fpath)
        h.update(f"{entry_rel}\0{file_hash}\n".encode())
    return h.hexdigest()


def canonical_serialize(obj: Any) -> str:
    """JSON-serialize ``obj`` deterministically for hashing/comparison.

    Used internally by tests; also exported for tasks that need to hash
    a manifest body. Sorts keys; uses ``,`` and ``:`` separators (no
    whitespace); does not append a trailing newline.
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
