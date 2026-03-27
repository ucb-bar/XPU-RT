"""Persistent kernel database.

Stores generated kernels indexed by (pattern_type, target_name, shapes).
The agent checks the DB before running expensive Autocomp searches —
reuse verified kernels across sessions.

Storage: JSON file on disk. Simple but sufficient for research.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class KernelEntry:
    """A stored kernel with metadata.

    Attributes:
        pattern_type: Pattern this kernel handles (e.g., "linear_chain").
        target_name: Hardware target name.
        shapes_key: Canonical string of input/output shapes for lookup.
        kernel_code: The kernel source code.
        language: "triton", "cuda", "python".
        latency_us: Measured latency.
        correct: Whether verification passed.
        speedup: Speedup over reference.
        plan: Optimization plan that produced this kernel.
    """

    pattern_type: str
    target_name: str
    shapes_key: str
    kernel_code: str
    language: str
    latency_us: float
    correct: bool
    speedup: float
    plan: str = ""


def _shapes_key(shapes: dict[str, tuple[int, ...]] | None) -> str:
    """Canonical string key from shapes dict."""
    if not shapes:
        return "unknown"
    parts = []
    for name in sorted(shapes):
        dims = "x".join(str(d) for d in shapes[name])
        parts.append(f"{name}:{dims}")
    return "|".join(parts)


class KernelDB:
    """Persistent kernel database backed by a JSON file."""

    def __init__(self, db_path: str | Path = ".compgen_cache/kernel_db.json") -> None:
        self._path = Path(db_path)
        self._entries: list[KernelEntry] = []
        self._load()

    def _load(self) -> None:
        """Load entries from disk."""
        if self._path.exists():
            data = json.loads(self._path.read_text())
            self._entries = [KernelEntry(**e) for e in data]

    def _save(self) -> None:
        """Persist entries to disk."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = [asdict(e) for e in self._entries]
        self._path.write_text(json.dumps(data, indent=2))

    def lookup(
        self,
        pattern_type: str,
        target_name: str,
        shapes: dict[str, tuple[int, ...]] | None = None,
    ) -> KernelEntry | None:
        """Find a cached kernel matching the pattern, target, and shapes."""
        key = _shapes_key(shapes)
        for entry in self._entries:
            if (entry.pattern_type == pattern_type
                    and entry.target_name == target_name
                    and entry.shapes_key == key
                    and entry.correct):
                return entry
        return None

    def store(self, entry: KernelEntry) -> None:
        """Store a kernel entry. Replaces existing entry with same key if better."""
        # Remove existing entry with same key if this one is better
        self._entries = [
            e for e in self._entries
            if not (e.pattern_type == entry.pattern_type
                    and e.target_name == entry.target_name
                    and e.shapes_key == entry.shapes_key)
        ]
        self._entries.append(entry)
        self._save()

    def best_for(
        self,
        pattern_type: str,
        target_name: str,
    ) -> KernelEntry | None:
        """Get the best kernel for a pattern type on a target (any shape)."""
        matches = [
            e for e in self._entries
            if e.pattern_type == pattern_type and e.target_name == target_name and e.correct
        ]
        if not matches:
            return None
        return min(matches, key=lambda e: e.latency_us)

    def all_entries(self) -> list[KernelEntry]:
        """Get all entries."""
        return list(self._entries)

    def count(self) -> int:
        """Number of stored kernels."""
        return len(self._entries)


__all__ = ["KernelDB", "KernelEntry"]
