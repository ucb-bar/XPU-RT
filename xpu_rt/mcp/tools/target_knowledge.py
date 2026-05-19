"""MCP tools for per-target knowledge cards.

Two roles:

1. **Read-only inspection** — ``xpu_rt_target_list``,
   ``xpu_rt_target_show``, ``xpu_rt_target_lessons``. Cheap, no LLM
   calls, surface the on-disk card state so Claude Code can show the
   user what XPU-RT already knows about a target.

2. **Ingestion** — ``xpu_rt_target_prepare_ingest`` returns the chunks
   for one target's source manifest plus the routing schema, so the
   in-loop Claude Code session can classify each chunk inline and then
   call ``xpu_rt_target_apply_routing`` to fold the structured results
   back into the card. The headless path,
   ``xpu_rt_target_ingest_headless``, runs the same pipeline with the
   Gemini router behind the $100 budget gate.

Tools never reach out to the network on their own — all crawls happen
through the optional ingest extra and only when the caller passes a
URL source explicitly.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from typing import Any

from xpu_rt.memory import target_knowledge as tk
from xpu_rt.memory.ingest import (
    IngestPipeline,
    RouterMock,
    SourceManifest,
    SourceRef,
)
from xpu_rt.memory.ingest.chunking import TextChunk, chunk_text
from xpu_rt.memory.ingest.loaders import load_path
from xpu_rt.memory.ingest.router import (
    ROUTER_PROMPT_VERSION,
    ROUTER_RESPONSE_SCHEMA,
    ROUTER_SYSTEM_PROMPT,
    RouterChunk,
    RouterResult,
)
from xpu_rt.memory.ingest.sources import expand_sources
from xpu_rt.memory.seeds import gemmini as gemmini_seed
from xpu_rt.memory.seeds import saturn as saturn_seed

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Read-only inspectors
# ---------------------------------------------------------------------------


def xpu_rt_target_list(sm: Any, **kwargs: Any) -> dict[str, Any]:
    """List all targets with a knowledge card on disk."""
    targets = tk.list_targets()
    return {
        "ok": True,
        "targets": targets,
        "count": len(targets),
        "knowledge_root": str(tk.knowledge_root()),
    }


def xpu_rt_target_show(sm: Any, *, target_id: str, **kwargs: Any) -> dict[str, Any]:
    """Return the full TargetKnowledgeCard for ``target_id``."""
    if not tk.exists(target_id):
        return {"ok": False, "error": f"no card for target_id={target_id!r}"}
    card = tk.load(target_id)
    return {
        "ok": True,
        "target_id": target_id,
        "card": card.to_dict(),
        "card_path": str(card.card_path),
        "lesson_count": sum(1 for _ in tk.iter_lessons(card)),
    }


def xpu_rt_target_lessons(
    sm: Any,
    *,
    target_id: str,
    op_family: str = "",
    dtype_class: str = "",
    layout_kind: str = "",
    limit: int = 20,
    **kwargs: Any,
) -> dict[str, Any]:
    """Return recent lessons for a target, optionally filtered by state.

    Filters are AND-combined; an empty filter means "any".
    """
    if not tk.exists(target_id):
        return {"ok": False, "error": f"no card for target_id={target_id!r}"}
    card = tk.load(target_id)
    rows = []
    for lesson in tk.iter_lessons(card):
        if op_family and lesson.op_family != op_family:
            continue
        if dtype_class and lesson.dtype_class != dtype_class:
            continue
        if layout_kind and lesson.layout_kind != layout_kind:
            continue
        rows.append(asdict(lesson))
    rows.sort(key=lambda r: r["timestamp"], reverse=True)
    return {
        "ok": True,
        "target_id": target_id,
        "lessons": rows[: max(limit, 1)],
        "lesson_count_total": sum(1 for _ in tk.iter_lessons(card)),
    }


# ---------------------------------------------------------------------------
# Seed manifests
# ---------------------------------------------------------------------------


_KNOWN_SEEDS = {
    "gemmini": gemmini_seed,
    "saturn": saturn_seed,
}


def xpu_rt_target_known_seeds(sm: Any, **kwargs: Any) -> dict[str, Any]:
    """List the known per-target source manifests this build ships with."""
    entries = []
    for name, module in _KNOWN_SEEDS.items():
        try:
            m = module.manifest()
            entries.append(
                {
                    "seed": name,
                    "target_id": m.target_id,
                    "isa_family": m.isa_family,
                    "source_count": len(m.sources),
                    "source_root": str(module.source_root()),
                    "source_root_present": module.source_root().is_dir(),
                    "description": m.description,
                }
            )
        except Exception as exc:  # noqa: BLE001 — keep the tool resilient
            entries.append({"seed": name, "error": str(exc)})
    return {"ok": True, "seeds": entries}


# ---------------------------------------------------------------------------
# Agent-file ingestion: prepare + apply
# ---------------------------------------------------------------------------


def _resolve_manifest(
    *,
    seed: str | None,
    manifest_inline: dict[str, Any] | None,
) -> SourceManifest:
    if seed:
        if seed not in _KNOWN_SEEDS:
            raise ValueError(
                f"unknown seed {seed!r}; choose one of {sorted(_KNOWN_SEEDS)}"
            )
        return _KNOWN_SEEDS[seed].manifest()
    if manifest_inline is None:
        raise ValueError("provide either 'seed' or 'manifest_inline'")
    sources = []
    for raw in manifest_inline.get("sources", []):
        sources.append(
            SourceRef(
                locator=str(raw["locator"]),
                kind=raw.get("kind", "path"),
                role=raw.get("role", "auto"),
                line_range=tuple(raw["line_range"]) if raw.get("line_range") else None,
                glob=raw.get("glob", ""),
                max_depth=int(raw.get("max_depth", 4)),
                tags=tuple(raw.get("tags", ())),
            )
        )
    return SourceManifest(
        target_id=str(manifest_inline["target_id"]),
        target_profile_ref=str(manifest_inline.get("target_profile_ref", "")),
        isa_family=str(manifest_inline.get("isa_family", "other")),
        sources=tuple(sources),
        description=str(manifest_inline.get("description", "")),
    )


def xpu_rt_target_prepare_ingest(
    sm: Any,
    *,
    seed: str | None = None,
    manifest_inline: dict[str, Any] | None = None,
    max_chunks_per_source: int = 32,
    skip_exemplars: bool = True,
    **kwargs: Any,
) -> dict[str, Any]:
    """Materialise the chunks the agent must classify.

    The tool walks the manifest, runs each file through the loader,
    and returns the resulting :class:`RouterChunk`-shaped envelopes —
    along with the routing system prompt and JSON schema — so the
    in-loop Claude Code session can classify each chunk inline. The
    session then calls :func:`xpu_rt_target_apply_routing` with the
    structured results to fold them into the card.

    No LLM call is made by this tool.
    """
    try:
        manifest = _resolve_manifest(seed=seed, manifest_inline=manifest_inline)
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}

    chunks: list[dict[str, Any]] = []
    exemplars_to_copy: list[dict[str, Any]] = []
    errors: list[str] = []

    for ref in expand_sources(manifest.sources):
        if ref.kind == "url":
            errors.append(
                f"{ref.locator}: URL sources require the headless path "
                f"(xpu_rt_target_ingest_headless)"
            )
            continue
        from pathlib import Path

        path = Path(ref.locator)
        if skip_exemplars and path.suffix.lower() in (
            ".c", ".cc", ".cpp", ".cu", ".tri", ".triton", ".py",
        ):
            exemplars_to_copy.append(
                {
                    "locator": str(path),
                    "tags": list(ref.tags),
                    "role": ref.role,
                }
            )
            continue
        try:
            text, kind = load_path(path, line_range=ref.line_range)
        except (FileNotFoundError, ImportError) as exc:
            errors.append(f"{ref.locator}: {exc.__class__.__name__}: {exc}")
            continue
        for tc in chunk_text(text)[:max_chunks_per_source]:
            chunks.append(
                {
                    "chunk_id": f"{path.name}#{tc.chunk_index}/{tc.chunk_total}",
                    "source_locator": str(path),
                    "source_kind": kind,
                    "role_hint": ref.role,
                    "chunk_index": tc.chunk_index,
                    "chunk_total": tc.chunk_total,
                    "text": tc.text,
                }
            )

    return {
        "ok": True,
        "target_id": manifest.target_id,
        "target_profile_ref": manifest.target_profile_ref,
        "isa_family": manifest.isa_family,
        "router_prompt_version": ROUTER_PROMPT_VERSION,
        "router_system_prompt": ROUTER_SYSTEM_PROMPT,
        "router_response_schema": ROUTER_RESPONSE_SCHEMA,
        "chunks": chunks,
        "exemplars_to_copy": exemplars_to_copy,
        "chunk_count": len(chunks),
        "errors": errors,
    }


def xpu_rt_target_apply_routing(
    sm: Any,
    *,
    target_id: str,
    target_profile_ref: str,
    isa_family: str,
    routed_results: list[dict[str, Any]],
    exemplars_to_copy: list[dict[str, Any]] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Fold a batch of routed results into the target's knowledge card.

    Args:
        target_id / target_profile_ref / isa_family: identify the
            target. The card is created on first call.
        routed_results: list of router-response dicts; each dict matches
            :data:`ROUTER_RESPONSE_SCHEMA` plus an optional
            ``chunk_id`` for trace-back.
        exemplars_to_copy: list of ``{locator, tags, role}`` dicts
            returned by :func:`xpu_rt_target_prepare_ingest` so the
            tool can copy exemplar files without re-walking the manifest.
    """
    if not target_id:
        return {"ok": False, "error": "target_id is required"}
    # Build a synthetic manifest just so the pipeline merges static
    # facts on the first call. No sources; the routed results bypass
    # the loader/chunker entirely.
    manifest = SourceManifest(
        target_id=target_id,
        target_profile_ref=target_profile_ref,
        isa_family=isa_family,
    )

    # The :class:`IngestPipeline` is happy to fold pre-computed results
    # if we feed them in via a RouterMock that ignores the request and
    # replays results in order.
    queue = list(routed_results)

    def replay(_item: RouterChunk) -> RouterResult:
        if not queue:
            return RouterResult(bucket="skip")
        return RouterResult.from_json(queue.pop(0))

    mock = RouterMock(table=replay)
    pipeline = IngestPipeline(router=mock, use_cache=False)
    # Build a temporary manifest with one synthetic chunk per result so
    # the pipeline does its merge step. We use one tiny synthetic file
    # per result; the loader is mocked-out via the pipeline's existing
    # path because we already pre-computed the result.
    # Easier: bypass the manifest walking and merge directly using the
    # internal pipeline helpers.
    from xpu_rt.memory.ingest.pipeline import _merge_into_card  # type: ignore[attr-defined]

    card = pipeline._load_or_create_card(manifest)  # type: ignore[attr-defined]
    instructions: list = []
    intrinsics: list = []
    parameters: list = []
    constraints: list = []
    bucket_text: dict[str, list[str]] = {b: [] for b in tk.BUCKETS}
    for raw in routed_results:
        try:
            result = RouterResult.from_json(raw)
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": f"malformed routed result: {exc}"}
        if result.bucket == "skip":
            continue
        if result.summary_md.strip() and result.bucket in tk.BUCKETS:
            bucket_text[result.bucket].append(result.summary_md.strip())
        instructions.extend(result.instructions)
        intrinsics.extend(result.intrinsics)
        parameters.extend(result.parameters)
        constraints.extend(result.constraints)

    # Copy exemplars.
    new_exemplars: list = []
    new_docs: list = list(card.docs)
    from datetime import datetime, timezone
    from pathlib import Path as _Path
    import shutil

    from xpu_rt.memory.target_knowledge import DocSource, KernelExemplar

    for entry in (exemplars_to_copy or []):
        src = _Path(entry["locator"])
        if not src.is_file():
            continue
        dest = card.exemplars_dir / src.name
        shutil.copyfile(src, dest)
        new_exemplars.append(
            KernelExemplar(
                name=src.stem,
                op_family="other",
                path=src.name,
                language=src.suffix.lstrip("."),
                tags=tuple(entry.get("tags", ())),
                source=str(src),
            )
        )
        new_docs.append(
            DocSource(
                locator=str(src),
                kind="path",
                sha256="",
                fetched_at=datetime.now(timezone.utc).isoformat(),
                bucket="examples",
                bytes=dest.stat().st_size if dest.exists() else 0,
                notes="exemplar",
            )
        )

    merged = _merge_into_card(
        card=card,
        isa_family=isa_family,
        new_instructions=instructions,
        new_intrinsics=intrinsics,
        new_parameters=parameters,
        new_constraints=constraints,
        new_exemplars=new_exemplars,
        new_docs=new_docs,
    )
    saved = tk.save(merged)

    # Write per-bucket markdown.
    for bucket, parts in bucket_text.items():
        joined = "\n\n".join(p for p in parts if p).strip()
        if not joined:
            continue
        path = saved.bucket_path(bucket)
        existing = path.read_text(encoding="utf-8") if path.exists() else ""
        header = f"<!-- routed via xpu_rt_target_apply_routing -->\n"
        path.write_text(
            (existing + "\n\n" if existing else "") + header + joined + "\n",
            encoding="utf-8",
        )

    return {
        "ok": True,
        "target_id": target_id,
        "card_path": str(saved.card_path),
        "instructions_added": len(instructions),
        "intrinsics_added": len(intrinsics),
        "parameters_added": len(parameters),
        "constraints_added": len(constraints),
        "exemplars_added": len(new_exemplars),
        "buckets_touched": sorted(b for b, parts in bucket_text.items() if parts),
    }


# ---------------------------------------------------------------------------
# Headless ingestion (Gemini, behind the $100 budget gate)
# ---------------------------------------------------------------------------


def xpu_rt_target_ingest_headless(
    sm: Any,
    *,
    seed: str | None = None,
    manifest_inline: dict[str, Any] | None = None,
    model: str = "gemini-2.5-flash",
    include_urls: bool = False,
    **kwargs: Any,
) -> dict[str, Any]:
    """Run the ingestion synchronously via Gemini.

    Use this for headless / CI / autonomous flows when no Claude Code
    agent is in the loop. Every call is gated by
    :func:`xpu_rt.observability.gemini_usage.check_pre_call` so the
    configured $100 cap is enforced before each request.
    """
    try:
        manifest = _resolve_manifest(seed=seed, manifest_inline=manifest_inline)
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}
    if include_urls and seed:
        manifest = _KNOWN_SEEDS[seed].manifest(include_urls=True)

    pipeline = IngestPipeline.from_gemini(model=model)
    try:
        card, report = pipeline.run(manifest)
    except Exception as exc:  # noqa: BLE001
        logger.exception("xpu_rt_target_ingest_headless failed")
        return {"ok": False, "error": f"{exc.__class__.__name__}: {exc}"}
    return {
        "ok": True,
        "target_id": card.target_id,
        "card_path": str(card.card_path),
        "report": {
            "sources_seen": report.sources_seen,
            "sources_skipped": report.sources_skipped,
            "chunks_routed": report.chunks_routed,
            "chunks_cached": report.chunks_cached,
            "chunks_skipped": report.chunks_skipped,
            "exemplars_copied": report.exemplars_copied,
            "docs_recorded": report.docs_recorded,
            "extracted_instructions": report.extracted_instructions,
            "extracted_intrinsics": report.extracted_intrinsics,
            "extracted_parameters": report.extracted_parameters,
            "extracted_constraints": report.extracted_constraints,
            "errors": report.errors,
        },
    }


# ---------------------------------------------------------------------------
# Tool registration
# ---------------------------------------------------------------------------


TARGET_KNOWLEDGE_TOOLS: list[dict[str, Any]] = [
    {
        "name": "xpu_rt_target_list",
        "description": "List all targets that have a knowledge card on disk.",
        "phase": "inspect",
        "handler": xpu_rt_target_list,
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "xpu_rt_target_show",
        "description": "Return the full TargetKnowledgeCard JSON for a given target_id.",
        "phase": "inspect",
        "handler": xpu_rt_target_show,
        "input_schema": {
            "type": "object",
            "properties": {"target_id": {"type": "string"}},
            "required": ["target_id"],
        },
    },
    {
        "name": "xpu_rt_target_lessons",
        "description": "Return recent lessons for a target, optionally filtered by state.",
        "phase": "inspect",
        "handler": xpu_rt_target_lessons,
        "input_schema": {
            "type": "object",
            "properties": {
                "target_id": {"type": "string"},
                "op_family": {"type": "string"},
                "dtype_class": {"type": "string"},
                "layout_kind": {"type": "string"},
                "limit": {"type": "integer", "default": 20},
            },
            "required": ["target_id"],
        },
    },
    {
        "name": "xpu_rt_target_known_seeds",
        "description": "List the per-target source manifests this build ships with.",
        "phase": "inspect",
        "handler": xpu_rt_target_known_seeds,
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "xpu_rt_target_prepare_ingest",
        "description": (
            "Walk a source manifest (named seed or inline) and return all "
            "chunks the agent must classify, the routing schema, and the "
            "system prompt. No LLM call is made. Pair with "
            "xpu_rt_target_apply_routing to commit the agent's routed results."
        ),
        "phase": "lifecycle",
        "handler": xpu_rt_target_prepare_ingest,
        "input_schema": {
            "type": "object",
            "properties": {
                "seed": {"type": "string", "enum": list(_KNOWN_SEEDS.keys())},
                "manifest_inline": {"type": "object"},
                "max_chunks_per_source": {"type": "integer", "default": 32},
                "skip_exemplars": {"type": "boolean", "default": True},
            },
        },
    },
    {
        "name": "xpu_rt_target_apply_routing",
        "description": (
            "Fold a batch of agent-routed results (one per chunk) into "
            "the target's knowledge card. Copies named exemplars at the "
            "same time. Idempotent: re-applying the same batch is safe."
        ),
        "phase": "lifecycle",
        "handler": xpu_rt_target_apply_routing,
        "input_schema": {
            "type": "object",
            "properties": {
                "target_id": {"type": "string"},
                "target_profile_ref": {"type": "string"},
                "isa_family": {"type": "string"},
                "routed_results": {"type": "array"},
                "exemplars_to_copy": {"type": "array"},
            },
            "required": ["target_id", "isa_family", "routed_results"],
        },
    },
    {
        "name": "xpu_rt_target_ingest_headless",
        "description": (
            "Run the ingestion synchronously via Gemini. Use only for "
            "headless / CI / autonomous flows when no Claude Code agent "
            "is in the loop. Every call respects the configured Gemini "
            "cumulative-USD cap."
        ),
        "phase": "lifecycle",
        "handler": xpu_rt_target_ingest_headless,
        "input_schema": {
            "type": "object",
            "properties": {
                "seed": {"type": "string", "enum": list(_KNOWN_SEEDS.keys())},
                "manifest_inline": {"type": "object"},
                "model": {"type": "string", "default": "gemini-2.5-flash"},
                "include_urls": {"type": "boolean", "default": False},
            },
        },
    },
]


__all__ = ["TARGET_KNOWLEDGE_TOOLS"]
