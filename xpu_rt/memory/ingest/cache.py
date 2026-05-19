"""Content-addressed cache for router responses.

A cache hit means a chunk with identical text + identical router prompt
version + identical role hint has already been classified, so we can
reuse the result for free. This is what makes "re-run the ingestion for
a target" cheap.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from xpu_rt.memory.ingest.router import (
    ROUTER_PROMPT_VERSION,
    RouterChunk,
    RouterResult,
)
from xpu_rt.memory.target_knowledge import knowledge_root

logger = logging.getLogger(__name__)


def cache_dir() -> Path:
    override = os.environ.get("XPU_RT_INGEST_CACHE_DIR")
    if override:
        path = Path(override)
    else:
        # Sibling to the per-target dirs, keyed off the same root override.
        path = knowledge_root().parent / ".ingest_cache"
    path.mkdir(parents=True, exist_ok=True)
    return path


def cache_key(item: RouterChunk) -> str:
    """Return the SHA256 hex digest used to address a cache entry.

    Includes the prompt version so a prompt change invalidates the whole
    cache; includes the role hint and isa family so identical text under
    a different hint still re-routes.
    """
    h = hashlib.sha256()
    h.update(ROUTER_PROMPT_VERSION.encode("utf-8"))
    h.update(b"\x00")
    h.update(item.isa_family.encode("utf-8"))
    h.update(b"\x00")
    h.update(item.role_hint.encode("utf-8"))
    h.update(b"\x00")
    h.update(item.source_kind.encode("utf-8"))
    h.update(b"\x00")
    h.update(item.chunk.text.encode("utf-8"))
    return h.hexdigest()


def _result_to_dict(result: RouterResult) -> dict[str, Any]:
    body: dict[str, Any] = {
        "bucket": result.bucket,
        "summary_md": result.summary_md,
        "instructions": [
            asdict(i) if is_dataclass(i) else i for i in result.instructions
        ],
        "intrinsics": [_intrinsic_dict(i) for i in result.intrinsics],
        "parameters": [_param_dict(p) for p in result.parameters],
        "constraints": list(result.constraints),
        "exemplar_tags": list(result.exemplar_tags),
    }
    return body


def _intrinsic_dict(intr: Any) -> dict[str, Any]:
    return {
        "name": intr.name,
        "c_signature": intr.c_signature,
        "summary": intr.summary,
        "notes": intr.notes,
    }


def _param_dict(p: Any) -> dict[str, Any]:
    return {
        "name": p.name,
        "description": p.description,
        "default": p.default,
        "unit": p.unit,
        "values": list(p.values),
    }


def get(item: RouterChunk) -> RouterResult | None:
    """Return a cached :class:`RouterResult` for ``item`` if present."""
    path = cache_dir() / f"{cache_key(item)}.json"
    if not path.exists():
        return None
    try:
        body = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("ingest cache: dropping unreadable entry %s (%s)", path, exc)
        return None
    return RouterResult.from_json(body)


def put(item: RouterChunk, result: RouterResult) -> None:
    """Persist ``result`` to the content-addressed cache.

    Always caches — even an empty "skip" result represents an LLM call
    we already paid for. The marginal inode cost is trivial compared to
    re-routing the same chunk on the next pipeline run.
    """
    path = cache_dir() / f"{cache_key(item)}.json"
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(_result_to_dict(result), indent=2))
    tmp.replace(path)


__all__ = ["cache_dir", "cache_key", "get", "put"]
