"""Cache for verified rewrite families.

Stores verification results keyed by a hash of the pattern and
replacement. This avoids re-verifying the same rewrite across
compilations.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import structlog

from compgen.semantic.backends.xdsl_smt.results import PDLResult

log = structlog.get_logger()


@dataclass
class VerifiedRewriteCache:
    """Persistent cache for rewrite verification results.

    Attributes:
        cache_dir: Directory to store cache files.
    """

    cache_dir: Path | None = None
    _memory: dict[str, PDLResult] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self._memory = {}
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _hash_pattern(pattern_repr: str, replacement_repr: str) -> str:
        """Compute a stable hash for a pattern/replacement pair."""
        content = f"{pattern_repr}|||{replacement_repr}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def lookup(self, pattern_hash: str) -> PDLResult | None:
        """Look up a cached verification result.

        Args:
            pattern_hash: Hash of the pattern/replacement pair.

        Returns:
            Cached PDLResult or None.
        """
        if pattern_hash in self._memory:
            return self._memory[pattern_hash]

        if self.cache_dir is not None:
            path = self.cache_dir / f"{pattern_hash}.json"
            if path.exists():
                try:
                    data = json.loads(path.read_text())
                    result = PDLResult(
                        sound=data["sound"],
                        status=data["status"],
                        bitwidths_checked=data.get("bitwidths_checked", []),
                        unsound_bitwidths=data.get("unsound_bitwidths", []),
                        solver_time_ms=data.get("solver_time_ms", 0.0),
                    )
                    self._memory[pattern_hash] = result
                    return result
                except (json.JSONDecodeError, KeyError):
                    pass
        return None

    def store(self, pattern_hash: str, result: PDLResult) -> None:
        """Store a verification result in the cache.

        Args:
            pattern_hash: Hash of the pattern/replacement pair.
            result: Verification result to cache.
        """
        self._memory[pattern_hash] = result

        if self.cache_dir is not None:
            path = self.cache_dir / f"{pattern_hash}.json"
            data = {
                "sound": result.sound,
                "status": result.status,
                "bitwidths_checked": result.bitwidths_checked,
                "unsound_bitwidths": result.unsound_bitwidths,
                "solver_time_ms": result.solver_time_ms,
            }
            path.write_text(json.dumps(data, indent=2))
            log.debug("rewrite.cache.stored", hash=pattern_hash)


__all__ = ["VerifiedRewriteCache"]
