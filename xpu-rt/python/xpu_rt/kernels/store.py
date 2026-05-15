"""On-disk kernel store under the user folder.

Layout::

    ~/.compgen/kernels/
        manifest.json                      # index: fingerprint → entry
        <target>/<fingerprint>.<lang>      # one file per kernel

Each entry in ``manifest.json`` records:

    {
      "fingerprint":         "abcd1234ef567890",
      "target":              "openq_5165rb",
      "language":            "triton",
      "path":                "openq_5165rb/abcd1234ef567890.triton",
      "op_name":             "linalg.matmul",
      "archetype":           "compute_tiled",
      "granularity":         "normal",
      "perf_us":             87.3,
      "correctness_passed":  true,
      "created_at":          "2026-04-20T08:31:42Z",
      "size_bytes":          1842
    }

The MCP tool ``register_kernel_result`` writes through the store on every
fulfillment; ``lookup_cached_kernel`` and session-open both rehydrate
from it. So a kernel generated in one Claude Code session lives forever
under the user's home directory and any future session — including
``compile_with_llm`` headless runs — gets a cache hit.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Default location
# ---------------------------------------------------------------------------


def default_store_root() -> Path:
    """``~/.compgen/kernels``, overridable via ``COMPGEN_KERNEL_STORE``."""
    override = os.environ.get("COMPGEN_KERNEL_STORE")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".compgen" / "kernels"


# ---------------------------------------------------------------------------
# Manifest entry
# ---------------------------------------------------------------------------


_LANG_EXT = {
    "triton": "triton.py",
    "python": "py",
    "cuda": "cu",
    "c": "c",
    "cpp": "cpp",
    "mlir": "mlir",
    "asm": "s",
    "unknown": "txt",
}


def _ext_for(language: str) -> str:
    return _LANG_EXT.get(language, "txt")


def _safe_target(target: str) -> str:
    """Sanitise target name for filesystem use."""
    if not target:
        return "_unknown_target"
    out = "".join(c if c.isalnum() or c in ("-", "_", ".") else "_" for c in target)
    return out or "_unknown_target"


@dataclass
class StoredKernel:
    fingerprint: str
    target: str
    language: str
    path: str  # relative to store root
    op_name: str = ""
    archetype: str = ""
    granularity: str = ""
    perf_us: float | None = None
    correctness_passed: bool = False
    created_at: str = ""
    size_bytes: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


@dataclass
class KernelStore:
    """File-backed kernel cache rooted at a user-folder.

    Thread/process safety: writes are O_TRUNC + atomic-rename through a
    sibling ``.tmp`` file; manifest writes use the same pattern. Single
    writer assumed. Multiple readers are safe.
    """

    root: Path = field(default_factory=default_store_root)

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._manifest_path = self.root / "manifest.json"
        if not self._manifest_path.exists():
            self._write_manifest({})

    # ----- public API used by the MCP tools -----

    def put(
        self,
        fingerprint: str,
        kernel_code: str,
        *,
        target: str = "",
        language: str = "unknown",
        op_name: str = "",
        archetype: str = "",
        granularity: str = "",
        perf_us: float | None = None,
        correctness_passed: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> StoredKernel:
        """Persist a kernel; updates the manifest. Idempotent — overwrites
        on identical fingerprint."""
        target_dir = self.root / _safe_target(target or "_unknown_target")
        target_dir.mkdir(parents=True, exist_ok=True)

        ext = _ext_for(language)
        rel_path = f"{_safe_target(target or '_unknown_target')}/{fingerprint}.{ext}"
        full_path = self.root / rel_path
        self._atomic_write_text(full_path, kernel_code)

        entry = StoredKernel(
            fingerprint=fingerprint,
            target=target,
            language=language,
            path=rel_path,
            op_name=op_name,
            archetype=archetype,
            granularity=granularity,
            perf_us=perf_us,
            correctness_passed=correctness_passed,
            created_at=datetime.now(UTC).isoformat(timespec="seconds"),
            size_bytes=len(kernel_code.encode("utf-8")),
            metadata=dict(metadata or {}),
        )
        manifest = self._read_manifest()
        manifest[fingerprint] = asdict(entry)
        self._write_manifest(manifest)
        return entry

    def get(self, fingerprint: str) -> tuple[StoredKernel, str] | None:
        """Return ``(StoredKernel, kernel_source)`` or None.

        ``COMPGEN_DISABLE_KERNEL_CACHE=1`` forces a cold lookup
        (always returns None) so audit runs can prove a kernel is being
        recomputed, not retrieved.
        """
        if os.environ.get("COMPGEN_DISABLE_KERNEL_CACHE") == "1":
            return None
        manifest = self._read_manifest()
        raw = manifest.get(fingerprint)
        if raw is None:
            return None
        entry = StoredKernel(**{k: v for k, v in raw.items() if k in StoredKernel.__dataclass_fields__})
        full_path = self.root / entry.path
        if not full_path.exists():
            # Manifest is stale — drop the entry.
            self.delete(fingerprint)
            return None
        return entry, full_path.read_text()

    def delete(self, fingerprint: str) -> bool:
        manifest = self._read_manifest()
        raw = manifest.pop(fingerprint, None)
        if raw is None:
            return False
        path = self.root / raw["path"]
        if path.exists():
            path.unlink()
        self._write_manifest(manifest)
        return True

    def list_all(self) -> list[StoredKernel]:
        manifest = self._read_manifest()
        return [
            StoredKernel(**{k: v for k, v in raw.items() if k in StoredKernel.__dataclass_fields__})
            for raw in manifest.values()
        ]

    def iter_kernels(self) -> Iterator[tuple[StoredKernel, str]]:
        for entry in self.list_all():
            full = self.root / entry.path
            if full.exists():
                yield entry, full.read_text()

    # ----- internals -----

    def _read_manifest(self) -> dict[str, dict[str, Any]]:
        if not self._manifest_path.exists():
            return {}
        try:
            return json.loads(self._manifest_path.read_text())
        except json.JSONDecodeError:
            # Corrupt manifest — start fresh; the on-disk files survive
            # but the index is rebuildable via a future ``rescan()``.
            return {}

    def _write_manifest(self, manifest: dict[str, Any]) -> None:
        self._atomic_write_text(self._manifest_path, json.dumps(manifest, indent=2))

    @staticmethod
    def _atomic_write_text(path: Path, content: str) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(content)
        os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Convenience: process-wide singleton
# ---------------------------------------------------------------------------


_singleton: KernelStore | None = None


def shared_store() -> KernelStore:
    """Process-wide :class:`KernelStore` rooted at ``default_store_root()``.

    Tests override via ``set_shared_store(KernelStore(root=tmp_path))``.
    """
    global _singleton
    if _singleton is None:
        _singleton = KernelStore()
    return _singleton


def set_shared_store(store: KernelStore | None) -> None:
    """Replace (or clear) the shared store. For tests; not for production."""
    global _singleton
    _singleton = store


__all__ = [
    "KernelStore",
    "StoredKernel",
    "default_store_root",
    "set_shared_store",
    "shared_store",
]
