"""Per-target knowledge-card seed extractors.

Each module here knows how to read a specific upstream source tree
(Chipyard generator, vendor SDK, public ISA reference, …) and produce a
fresh :class:`~xpu_rt.memory.target_knowledge.TargetKnowledgeCard`
*without* making any LLM calls. Seeds are deterministic and fast; the
LLM-driven doc-ingestion pipeline layers on top to enrich the card with
narrative buckets after the seed has run.
"""

from __future__ import annotations

__all__: list[str] = []
