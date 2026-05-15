"""Persist Triton autotune picks across processes.

Triton's ``@triton.autotune(configs=[...], key=["M", "N", "K"])`` decorator
caches the *winning config per key* in process memory only — start a
fresh process and you pay the sweep cost (~1-2 s per unique key) all
over again.

This module persists those picks to JSON on disk so a fresh process
loads them instantly. Cold start drops from seconds-per-shape to
microseconds-per-shape. Combined with Triton's already-built-in
binary cache (``~/.triton/cache``), a deployed model has *zero*
first-call overhead after the first ever run on the host.

Layout under ``~/.compgen/autotune/`` (overridable via
``COMPGEN_AUTOTUNE_CACHE``):

    <kernel_qualname>.json      # one file per autotuned kernel
        {
          "<key_tuple_repr>": {
            "kwargs": {...},
            "num_warps": 4,
            "num_stages": 2,
            "num_ctas": 1,
            "maxnreg": null
          },
          ...
        }
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import triton


def default_cache_root() -> Path:
    override = os.environ.get("COMPGEN_AUTOTUNE_CACHE")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".compgen" / "autotune"


def _kernel_qualname(autotuned_fn: Any) -> str:
    """Stable on-disk filename component for an autotuned JIT function."""
    # ``triton.autotune`` returns an ``Autotuner`` wrapping the JITFunction.
    # The underlying fn name lives at .fn.__name__.
    inner = getattr(autotuned_fn, "fn", autotuned_fn)
    return getattr(inner, "__name__", repr(autotuned_fn))


def _key_to_str(key: Any) -> str:
    """Stable string repr of an autotune cache key (tuple of M,N,K,dtypes,…)."""
    return json.dumps(list(key), default=str)


def _config_to_dict(cfg: triton.Config) -> dict[str, Any]:
    out: dict[str, Any] = {
        "kwargs": dict(cfg.kwargs),
        "num_warps": cfg.num_warps,
        "num_stages": cfg.num_stages,
    }
    if hasattr(cfg, "num_ctas"):
        out["num_ctas"] = cfg.num_ctas
    if hasattr(cfg, "maxnreg"):
        out["maxnreg"] = cfg.maxnreg
    return out


def _dict_to_config(d: dict[str, Any]) -> triton.Config:
    kw = {
        "kwargs": d["kwargs"],
        "num_warps": d.get("num_warps", 4),
        "num_stages": d.get("num_stages", 2),
    }
    if "num_ctas" in d:
        kw["num_ctas"] = d["num_ctas"]
    if "maxnreg" in d and d["maxnreg"] is not None:
        kw["maxnreg"] = d["maxnreg"]
    return triton.Config(**kw)


# ---------------------------------------------------------------------------
# Save / load one autotuned kernel
# ---------------------------------------------------------------------------


def save(autotuned_fn: Any, *, root: Path | None = None) -> Path:
    """Snapshot one autotuned kernel's picks to disk. Returns the path."""
    root = Path(root) if root is not None else default_cache_root()
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{_kernel_qualname(autotuned_fn)}.json"

    cache = getattr(autotuned_fn, "cache", {})
    payload = {_key_to_str(k): _config_to_dict(v) for k, v in cache.items()}
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    os.replace(tmp, path)
    return path


def load(autotuned_fn: Any, *, root: Path | None = None) -> int:
    """Hydrate ``autotuned_fn.cache`` from disk. Returns the number of
    keys loaded. Silently no-op when the file doesn't exist (first run).
    """
    root = Path(root) if root is not None else default_cache_root()
    path = root / f"{_kernel_qualname(autotuned_fn)}.json"
    if not path.exists():
        return 0
    payload = json.loads(path.read_text())
    cache = autotuned_fn.cache
    loaded = 0
    for key_str, cfg_dict in payload.items():
        key = tuple(json.loads(key_str))
        # Triton's cache is keyed by tuple — but our save round-trips
        # through JSON's list representation. Accept any iterable key.
        cache[key] = _dict_to_config(cfg_dict)
        loaded += 1
    return loaded


def save_all(autotuned_fns: Iterable[Any], *, root: Path | None = None) -> dict[str, Path]:
    """Snapshot every autotuned kernel in ``autotuned_fns``."""
    return {_kernel_qualname(fn): save(fn, root=root) for fn in autotuned_fns}


def load_all(autotuned_fns: Iterable[Any], *, root: Path | None = None) -> dict[str, int]:
    """Hydrate every kernel's autotune cache from disk."""
    return {_kernel_qualname(fn): load(fn, root=root) for fn in autotuned_fns}


__all__ = [
    "default_cache_root",
    "load",
    "load_all",
    "save",
    "save_all",
]
