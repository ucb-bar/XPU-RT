"""In-memory LRU cache for ``PipelineResult``.

Keyed by ``(model_id, options.stable_key(), input_signature)``. Opt-in
— wrap the entry point with :func:`with_cache` or hand the cache
instance to ``compile_through_pipeline_cached``.

The cache stores ``PipelineResult`` objects by identity; callers
should not mutate returned plans / modules. A cached plan is shared
across callers by design.

Bounded by ``max_entries`` (default 64); oldest-first eviction on
overflow.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

from compgen.options import CompGenOptions
from compgen.pipeline.driver import PipelineResult, compile_through_pipeline


@dataclass
class PipelineCacheStats:
    hits: int = 0
    misses: int = 0
    evictions: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return (self.hits / total) if total else 0.0


def _model_identity(model: Any) -> str:
    """Identity string for a model.

    We prefer a hash of the model's state_dict keys + shapes so that
    re-constructed identical models hit the cache. For anything else
    we fall back to ``id()``.
    """
    try:
        import torch

        if isinstance(model, torch.nn.Module):
            sd = model.state_dict()
            parts = []
            for k, v in sd.items():
                if hasattr(v, "shape") and hasattr(v, "dtype"):
                    parts.append(f"{k}:{tuple(v.shape)}:{v.dtype}")
                else:
                    parts.append(f"{k}:?")
            return "|".join(parts)
    except Exception:  # noqa: BLE001
        pass
    return f"id:{id(model)}"


def _input_signature(example_inputs: tuple[Any, ...] | None) -> tuple:
    if example_inputs is None:
        return ()
    sigs: list[tuple] = []
    for t in example_inputs:
        if hasattr(t, "shape") and hasattr(t, "dtype"):
            sigs.append(("tensor", tuple(t.shape), str(t.dtype)))
        else:
            sigs.append(("scalar", type(t).__name__))
    return tuple(sigs)


class PipelineCache:
    """LRU cache wrapping :func:`compile_through_pipeline`.

    Usage::

        cache = PipelineCache(max_entries=32)
        result = cache.compile(model, inputs, options=opts)
        result_again = cache.compile(model, inputs, options=opts)  # hits

    Stats::

        cache.stats.hit_rate  # 0.5
    """

    def __init__(self, max_entries: int = 64) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be >= 1")
        self.max_entries = max_entries
        self._entries: OrderedDict[tuple, PipelineResult] = OrderedDict()
        self.stats = PipelineCacheStats()

    def _make_key(
        self,
        model: Any,
        inputs: tuple[Any, ...] | None,
        options: CompGenOptions,
    ) -> tuple:
        return (
            _model_identity(model),
            options.stable_key(),
            _input_signature(inputs),
        )

    def compile(
        self,
        model: Any,
        example_inputs: tuple[Any, ...] | None = None,
        *,
        options: CompGenOptions | None = None,
        workload_name: str = "unnamed",
        target_name: str = "cuda_a100",
    ) -> PipelineResult:
        if options is None:
            options = CompGenOptions()
        key = self._make_key(model, example_inputs, options)
        cached = self._entries.get(key)
        if cached is not None:
            self.stats.hits += 1
            self._entries.move_to_end(key)
            return cached
        self.stats.misses += 1
        result = compile_through_pipeline(
            model,
            example_inputs=example_inputs,
            options=options,
            workload_name=workload_name,
            target_name=target_name,
        )
        self._entries[key] = result
        if len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)
            self.stats.evictions += 1
        return result

    def clear(self) -> None:
        self._entries.clear()
        self.stats = PipelineCacheStats()

    def __len__(self) -> int:
        return len(self._entries)

    # --- disk-backed persistence (W12.4) ----------------------------

    def save_manifest(self, path: Any) -> None:
        """Write a JSON manifest of cached keys to disk.

        We don't serialize ``PipelineResult`` objects (they hold
        xDSL IR that doesn't round-trip cleanly). The manifest is a
        record of "what was cached in this session" — useful for
        reproducibility + pre-warming on startup.
        """
        import json
        from pathlib import Path

        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload: list[dict[str, Any]] = []
        for key in self._entries:
            model_id, options_key, input_sig = key
            payload.append(
                {
                    "model_id": model_id,
                    "options_key": list(options_key),
                    "input_signature": list(input_sig),
                }
            )
        p.write_text(
            json.dumps(
                {
                    "version": 1,
                    "max_entries": self.max_entries,
                    "entries": payload,
                    "stats": {
                        "hits": self.stats.hits,
                        "misses": self.stats.misses,
                        "evictions": self.stats.evictions,
                    },
                },
                indent=2,
                default=str,
            )
        )

    @classmethod
    def load_manifest(cls, path: Any) -> PipelineCache:
        """Load cache metadata from a manifest written by :meth:`save_manifest`."""
        import json
        from pathlib import Path

        data = json.loads(Path(path).read_text())
        cache = cls(max_entries=int(data.get("max_entries", 64)))
        stats = data.get("stats", {})
        cache.stats.hits = int(stats.get("hits", 0))
        cache.stats.misses = int(stats.get("misses", 0))
        cache.stats.evictions = int(stats.get("evictions", 0))
        return cache


__all__ = [
    "PipelineCache",
    "PipelineCacheStats",
]
