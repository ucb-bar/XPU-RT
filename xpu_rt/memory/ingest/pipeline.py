"""Orchestrator that turns a SourceManifest into an updated TargetKnowledgeCard.

The pipeline walks every :class:`SourceRef` in the manifest, loads its
content, chunks it, runs each chunk through the chosen :class:`Router`,
caches results, and folds the structured records + per-bucket markdown
back into the target's
:class:`~xpu_rt.memory.target_knowledge.TargetKnowledgeCard`.

Pipeline construction picks the router at call time:

* ``IngestPipeline.from_agent_file(bridge=...)`` — Claude Code in-loop.
* ``IngestPipeline.from_gemini(model=...)`` — Gemini headless.
* ``IngestPipeline(router=RouterMock(...))`` — tests.
"""

from __future__ import annotations

import dataclasses
import logging
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from xpu_rt.memory.ingest import cache as router_cache
from xpu_rt.memory.ingest.chunking import chunk_text
from xpu_rt.memory.ingest.crawl import (
    FetchedDoc,
    IngestExtraNotInstalled,
    fetch_url,
)
from xpu_rt.memory.ingest.loaders import infer_kind, load_path
from xpu_rt.memory.ingest.router import (
    Router,
    RouterAgentFile,
    RouterChunk,
    RouterLLM,
    RouterMock,
    RouterResult,
)
from xpu_rt.memory.ingest.sources import (
    SourceManifest,
    SourceRef,
    expand_sources,
)
from xpu_rt.memory.target_knowledge import (
    BUCKETS,
    DocSource,
    HardwareSpec,
    ISAInstruction,
    IntrinsicSignature,
    KernelExemplar,
    ParameterRange,
    TargetKnowledgeCard,
    exists as card_exists,
    load as load_card,
    save as save_card,
)

logger = logging.getLogger(__name__)


# Code-shaped suffixes that should be copied into exemplars/ rather than
# routed through the LLM (the LLM sees the surrounding doc references
# instead, which is cheaper and more useful).
EXEMPLAR_SUFFIXES: tuple[str, ...] = (".c", ".cc", ".cpp", ".cu", ".tri", ".triton", ".py")


@dataclass
class IngestReport:
    """Summary of one ingestion run."""

    target_id: str
    sources_seen: int = 0
    sources_skipped: int = 0
    chunks_routed: int = 0
    chunks_cached: int = 0
    chunks_skipped: int = 0
    exemplars_copied: int = 0
    docs_recorded: int = 0
    bucket_chars: dict[str, int] = field(default_factory=dict)
    extracted_instructions: int = 0
    extracted_intrinsics: int = 0
    extracted_parameters: int = 0
    extracted_constraints: int = 0
    errors: list[str] = field(default_factory=list)


@dataclass
class IngestPipeline:
    """Drive a :class:`SourceManifest` end-to-end into a knowledge card.

    Construct with a :class:`Router` directly or via one of the
    classmethod helpers (:meth:`from_agent_file`, :meth:`from_gemini`).
    """

    router: Router
    use_cache: bool = True
    fetch_urls: bool = True
    max_chunks_per_source: int = 64

    @classmethod
    def from_agent_file(cls, *, bridge) -> IngestPipeline:  # type: ignore[no-untyped-def]
        return cls(router=RouterAgentFile(bridge=bridge))

    @classmethod
    def from_gemini(cls, *, model: str = "gemini-2.5-flash") -> IngestPipeline:
        return cls(router=RouterLLM(model=model))

    # ----------------------------------------------------------------- run

    def run(self, manifest: SourceManifest) -> tuple[TargetKnowledgeCard, IngestReport]:
        """Execute the manifest; return the updated card + a report."""
        report = IngestReport(target_id=manifest.target_id)
        card = self._load_or_create_card(manifest)

        # Accumulators folded into the card at the end so a mid-run abort
        # (e.g. budget exceeded) still has the card in a clean state.
        bucket_text: dict[str, list[str]] = {b: [] for b in BUCKETS}
        new_instructions: list[ISAInstruction] = []
        new_intrinsics: list[IntrinsicSignature] = []
        new_parameters: list[ParameterRange] = []
        new_constraints: list[str] = []
        new_exemplars: list[KernelExemplar] = []
        new_docs: list[DocSource] = list(card.docs)

        for ref in expand_sources(manifest.sources):
            report.sources_seen += 1
            try:
                self._process_ref(
                    ref=ref,
                    manifest=manifest,
                    card=card,
                    bucket_text=bucket_text,
                    new_instructions=new_instructions,
                    new_intrinsics=new_intrinsics,
                    new_parameters=new_parameters,
                    new_constraints=new_constraints,
                    new_exemplars=new_exemplars,
                    new_docs=new_docs,
                    report=report,
                )
            except IngestExtraNotInstalled:
                raise
            except Exception as exc:  # noqa: BLE001
                logger.exception(
                    "ingest: source %s failed; continuing with remaining sources", ref.locator
                )
                report.errors.append(f"{ref.locator}: {exc.__class__.__name__}: {exc}")
                report.sources_skipped += 1

        merged_card = _merge_into_card(
            card=card,
            isa_family=manifest.isa_family,
            new_instructions=new_instructions,
            new_intrinsics=new_intrinsics,
            new_parameters=new_parameters,
            new_constraints=new_constraints,
            new_exemplars=new_exemplars,
            new_docs=new_docs,
        )
        merged_card = save_card(merged_card)

        # Append routed markdown to per-bucket files.
        for bucket, parts in bucket_text.items():
            joined = "\n\n".join(p for p in parts if p).strip()
            if not joined:
                continue
            existing = ""
            target = merged_card.bucket_path(bucket)
            if target.exists():
                existing = target.read_text(encoding="utf-8")
            header = (
                f"<!-- routed {datetime.now(timezone.utc).isoformat()} | "
                f"router={self.router.name} -->\n"
            )
            target.write_text(
                (existing + "\n\n" if existing else "") + header + joined + "\n",
                encoding="utf-8",
            )
            report.bucket_chars[bucket] = report.bucket_chars.get(bucket, 0) + len(joined)

        return merged_card, report

    # ----------------------------------------------------- internal helpers

    def _load_or_create_card(self, manifest: SourceManifest) -> TargetKnowledgeCard:
        if card_exists(manifest.target_id):
            return load_card(manifest.target_id)
        # First-time card: pull whatever static facts the manifest brought
        # so the on-disk artifact has shape immediately.
        facts = manifest.static_hardware_facts
        spec = HardwareSpec(
            isa_family=manifest.isa_family,
            memory_tiers=tuple(facts.get("memory_tiers", ())),  # type: ignore[arg-type]
            dataflow_modes=tuple(facts.get("dataflow_modes", ())),  # type: ignore[arg-type]
            constraints=tuple(facts.get("constraints", ())),  # type: ignore[arg-type]
        )
        card = TargetKnowledgeCard(
            target_id=manifest.target_id,
            target_profile_ref=manifest.target_profile_ref,
            hardware_spec=spec,
        )
        return save_card(card)

    def _process_ref(
        self,
        *,
        ref: SourceRef,
        manifest: SourceManifest,
        card: TargetKnowledgeCard,
        bucket_text: dict[str, list[str]],
        new_instructions: list[ISAInstruction],
        new_intrinsics: list[IntrinsicSignature],
        new_parameters: list[ParameterRange],
        new_constraints: list[str],
        new_exemplars: list[KernelExemplar],
        new_docs: list[DocSource],
        report: IngestReport,
    ) -> None:
        # Exemplar fast path — code files don't go to the router; copy them
        # straight into the card's exemplars/ dir and emit a typed record.
        if ref.kind == "path" and _looks_like_exemplar(Path(ref.locator)):
            self._record_exemplar(ref, card, new_exemplars, new_docs, report)
            return

        if ref.kind == "url":
            if not self.fetch_urls:
                report.sources_skipped += 1
                return
            fetched = fetch_url(ref.locator)
            self._route_text(
                text=fetched.markdown,
                source_locator=ref.locator,
                source_kind="markdown",
                role_hint=ref.role,
                manifest=manifest,
                bucket_text=bucket_text,
                new_instructions=new_instructions,
                new_intrinsics=new_intrinsics,
                new_parameters=new_parameters,
                new_constraints=new_constraints,
                report=report,
            )
            new_docs.append(
                DocSource(
                    locator=ref.locator,
                    kind="url",
                    sha256=fetched.sha256,
                    fetched_at=fetched.fetched_at,
                    bucket="",
                    bytes=fetched.bytes_count,
                    notes=fetched.title,
                )
            )
            report.docs_recorded += 1
            return

        path = Path(ref.locator)
        try:
            text, kind = load_path(path, line_range=ref.line_range)
        except (FileNotFoundError, ImportError) as exc:
            report.errors.append(f"{ref.locator}: {exc.__class__.__name__}: {exc}")
            report.sources_skipped += 1
            return
        self._route_text(
            text=text,
            source_locator=str(path),
            source_kind=kind,
            role_hint=ref.role,
            manifest=manifest,
            bucket_text=bucket_text,
            new_instructions=new_instructions,
            new_intrinsics=new_intrinsics,
            new_parameters=new_parameters,
            new_constraints=new_constraints,
            report=report,
        )

    def _record_exemplar(
        self,
        ref: SourceRef,
        card: TargetKnowledgeCard,
        new_exemplars: list[KernelExemplar],
        new_docs: list[DocSource],
        report: IngestReport,
    ) -> None:
        src = Path(ref.locator)
        dest = card.exemplars_dir / src.name
        try:
            shutil.copyfile(src, dest)
        except OSError as exc:
            report.errors.append(f"{ref.locator}: copy failed: {exc}")
            return
        new_exemplars.append(
            KernelExemplar(
                name=src.stem,
                op_family=_infer_op_family(src.name, ref.tags),
                path=src.name,
                language=_infer_language(src.suffix),
                tags=ref.tags,
                source=str(src),
            )
        )
        report.exemplars_copied += 1
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

    def _route_text(
        self,
        *,
        text: str,
        source_locator: str,
        source_kind: str,
        role_hint: str,
        manifest: SourceManifest,
        bucket_text: dict[str, list[str]],
        new_instructions: list[ISAInstruction],
        new_intrinsics: list[IntrinsicSignature],
        new_parameters: list[ParameterRange],
        new_constraints: list[str],
        report: IngestReport,
    ) -> None:
        chunks = chunk_text(text)
        if len(chunks) > self.max_chunks_per_source:
            logger.warning(
                "ingest: %s produced %d chunks; truncating to %d",
                source_locator,
                len(chunks),
                self.max_chunks_per_source,
            )
            chunks = chunks[: self.max_chunks_per_source]

        for chunk in chunks:
            item = RouterChunk(
                chunk=chunk,
                source_locator=source_locator,
                source_kind=source_kind,
                role_hint=role_hint,
                target_id=manifest.target_id,
                isa_family=manifest.isa_family,
            )
            cached = router_cache.get(item) if self.use_cache else None
            if cached is not None:
                report.chunks_cached += 1
                result = cached
            else:
                result = self.router.classify(item)
                report.chunks_routed += 1
                if self.use_cache:
                    router_cache.put(item, result)
            self._fold_result(
                result,
                bucket_text=bucket_text,
                new_instructions=new_instructions,
                new_intrinsics=new_intrinsics,
                new_parameters=new_parameters,
                new_constraints=new_constraints,
                report=report,
            )

    def _fold_result(
        self,
        result: RouterResult,
        *,
        bucket_text: dict[str, list[str]],
        new_instructions: list[ISAInstruction],
        new_intrinsics: list[IntrinsicSignature],
        new_parameters: list[ParameterRange],
        new_constraints: list[str],
        report: IngestReport,
    ) -> None:
        if result.bucket == "skip":
            report.chunks_skipped += 1
            return
        if result.bucket in BUCKETS and result.summary_md.strip():
            bucket_text[result.bucket].append(result.summary_md.strip())
        new_instructions.extend(result.instructions)
        new_intrinsics.extend(result.intrinsics)
        new_parameters.extend(result.parameters)
        new_constraints.extend(result.constraints)
        report.extracted_instructions += len(result.instructions)
        report.extracted_intrinsics += len(result.intrinsics)
        report.extracted_parameters += len(result.parameters)
        report.extracted_constraints += len(result.constraints)


# ---------------------------------------------------------------------------
# Merge helpers
# ---------------------------------------------------------------------------


def _merge_into_card(
    *,
    card: TargetKnowledgeCard,
    isa_family: str,
    new_instructions: list[ISAInstruction],
    new_intrinsics: list[IntrinsicSignature],
    new_parameters: list[ParameterRange],
    new_constraints: list[str],
    new_exemplars: list[KernelExemplar],
    new_docs: list[DocSource],
) -> TargetKnowledgeCard:
    """Fold accumulated extractions into ``card`` (deduped by primary key)."""
    spec = card.hardware_spec
    merged_instructions = _dedup_by(
        (*spec.instructions, *new_instructions),
        key=lambda i: i.mnemonic,
    )
    merged_intrinsics = _dedup_by(
        (*spec.intrinsics, *new_intrinsics),
        key=lambda i: i.name,
    )
    merged_parameters = _dedup_by(
        (*spec.parameters, *new_parameters),
        key=lambda p: p.name,
    )
    merged_constraints = tuple(
        dict.fromkeys((*spec.constraints, *new_constraints))
    )
    new_spec = HardwareSpec(
        isa_family=isa_family or spec.isa_family,
        parameters=tuple(merged_parameters),
        memory_tiers=spec.memory_tiers,
        instructions=tuple(merged_instructions),
        intrinsics=tuple(merged_intrinsics),
        dataflow_modes=spec.dataflow_modes,
        constraints=merged_constraints,
    )
    merged_exemplars = _dedup_by(
        (*card.exemplars, *new_exemplars),
        key=lambda e: e.name,
    )
    return dataclasses.replace(
        card,
        hardware_spec=new_spec,
        exemplars=tuple(merged_exemplars),
        docs=tuple(new_docs),
        revision=card.revision + 1,
    )


def _dedup_by(items: Iterable, key) -> list:  # type: ignore[no-untyped-def]
    """First-wins dedup; preserves order so card stays diff-friendly."""
    seen: set = set()
    out: list = []
    for item in items:
        k = key(item)
        if k in seen:
            continue
        seen.add(k)
        out.append(item)
    return out


def _looks_like_exemplar(path: Path) -> bool:
    return path.suffix.lower() in EXEMPLAR_SUFFIXES


def _infer_language(suffix: str) -> str:
    suffix = suffix.lower()
    if suffix == ".c":
        return "c"
    if suffix in (".cc", ".cpp", ".cxx"):
        return "cpp"
    if suffix == ".cu":
        return "cuda"
    if suffix in (".tri", ".triton"):
        return "triton"
    if suffix == ".py":
        return "python"
    return ""


_OP_FAMILY_HEURISTICS: tuple[tuple[str, str], ...] = (
    ("matmul", "matmul"),
    ("gemm", "matmul"),
    ("conv", "conv"),
    ("dwconv", "conv"),
    ("sgemv", "gemv"),
    ("gemv", "gemv"),
    ("daxpy", "axpy"),
    ("dot", "reduce"),
    ("sum", "reduce"),
    ("fft", "fft"),
    ("softmax", "softmax"),
    ("relu", "activation"),
    ("pool", "pool"),
)


def _infer_op_family(name: str, tags: tuple[str, ...]) -> str:
    lowered = name.lower()
    for needle, family in _OP_FAMILY_HEURISTICS:
        if needle in lowered or needle in tags:
            return family
    return "other"


__all__ = [
    "EXEMPLAR_SUFFIXES",
    "FetchedDoc",
    "IngestExtraNotInstalled",
    "IngestPipeline",
    "IngestReport",
]
