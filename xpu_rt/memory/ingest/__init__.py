"""Universal documentation/ISA ingestion for per-target knowledge cards.

Reads local file paths or remote URLs that describe a hardware target,
classifies their content with an LLM into the fixed bucket set
(`isa | architecture | intrinsics | examples | constraints | skip`),
and merges structured records (ISAInstructions, IntrinsicSignatures,
ParameterRanges, constraint strings) plus per-bucket narrative
markdown into the target's
:class:`~xpu_rt.memory.target_knowledge.TargetKnowledgeCard`.

The pipeline is designed to be the *only* path for populating a card —
per-target hand-rolled regex parsers are explicitly discouraged in this
codebase because they overfit to upstream README/header layouts and rot
the moment those upstreams refactor.

The same pipeline drives:

* the static "seed" path (a per-target source manifest names the files
  to ingest from a vendor generator tree),
* the interactive Claude Code path (the user pastes a doc folder or
  URL, the skill calls the same pipeline).

Every LLM request goes through
:func:`xpu_rt.observability.gemini_usage.check_pre_call` so the
configured ``cumulative_usd`` cap is enforced before any spend.
"""

from __future__ import annotations

from xpu_rt.memory.ingest.pipeline import IngestPipeline, IngestReport, IngestExtraNotInstalled
from xpu_rt.memory.ingest.router import (
    AgentFileBridge,
    Router,
    RouterAgentFile,
    RouterChunk,
    RouterLLM,
    RouterMock,
    RouterResult,
)
from xpu_rt.memory.ingest.sources import SourceRef, SourceManifest

__all__ = [
    "AgentFileBridge",
    "IngestExtraNotInstalled",
    "IngestPipeline",
    "IngestReport",
    "Router",
    "RouterAgentFile",
    "RouterChunk",
    "RouterLLM",
    "RouterMock",
    "RouterResult",
    "SourceManifest",
    "SourceRef",
]
