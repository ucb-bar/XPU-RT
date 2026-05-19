"""Gemmini source manifest.

Declares the canonical files and URLs that describe the Gemmini target
to the ingestion pipeline. No regex parsing — the universal ingestion
in :mod:`xpu_rt.memory.ingest` consumes this manifest, runs each source
through the router (agent-file by default; Gemini in headless mode),
and produces the
:class:`~xpu_rt.memory.target_knowledge.TargetKnowledgeCard`.

Override the source root with ``XPU_RT_CHIPYARD_GEMMINI_ROOT`` for
machines where Chipyard lives in a different place.
"""

from __future__ import annotations

import os
from pathlib import Path

from xpu_rt.memory.ingest import SourceManifest, SourceRef
from xpu_rt.memory.target_knowledge import MemoryTierSpec

DEFAULT_SOURCE_ROOT = Path("/scratch2/agustin/chipyard/generators/gemmini")
TARGET_ID = "gemmini_mx"
TARGET_PROFILE_REF = "configs/targets/gemmini_mx.yaml"


# Static facts pulled directly from the canonical defaultConfig in
# Configs.scala. These do NOT depend on parsing the README and stay
# correct as long as the upstream defaults don't move; ingestion fills
# the narrative around them.
STATIC_FACTS: dict[str, object] = {
    "memory_tiers": (
        MemoryTierSpec(name="scratchpad", kind="scratchpad", size_bytes=256 * 1024),
        MemoryTierSpec(name="accumulator", kind="accumulator", size_bytes=64 * 1024),
        MemoryTierSpec(name="dram", kind="dram"),
    ),
    "dataflow_modes": ("weight_stationary", "output_stationary"),
    "constraints": (
        "RoCC custom-3 opcode (XCUSTOM_ACC)",
        "scratchpad addresses must be DIM-aligned",
        "single gemmini_loop_ws call must fit in half of scratchpad (double-buffered)",
    ),
}


def source_root() -> Path:
    env = os.environ.get("XPU_RT_CHIPYARD_GEMMINI_ROOT")
    return Path(env) if env else DEFAULT_SOURCE_ROOT


def manifest(*, include_urls: bool = False) -> SourceManifest:
    """Build the ingestion manifest for the Gemmini target.

    Args:
        include_urls: When True, adds public-doc URLs (UCB-BAR repo,
            tutorial slides) that require Crawl4AI. Set False (default)
            for offline / no-extra installs.
    """
    root = source_root()
    sources: list[SourceRef] = [
        SourceRef(
            locator=str(root / "README.md"),
            kind="path",
            role="auto",
            tags=("gemmini", "readme"),
        ),
        SourceRef(
            locator=str(root / "src" / "main" / "scala" / "gemmini" / "Configs.scala"),
            kind="path",
            role="architecture",
            tags=("gemmini", "scala-config", "defaults"),
        ),
        SourceRef(
            locator=str(root / "software" / "gemmini-rocc-tests" / "include" / "gemmini.h"),
            kind="path",
            role="intrinsics",
            tags=("gemmini", "c-header"),
        ),
        SourceRef(
            locator=str(root / "software" / "gemmini-rocc-tests" / "bareMetalC"),
            kind="directory",
            glob="matmul*.c",
            role="examples",
            max_depth=1,
            tags=("gemmini", "matmul", "exemplar"),
        ),
        SourceRef(
            locator=str(root / "software" / "gemmini-rocc-tests" / "bareMetalC" / "conv.c"),
            kind="path",
            role="examples",
            tags=("gemmini", "conv", "exemplar"),
        ),
        SourceRef(
            locator=str(root / "software" / "gemmini-rocc-tests" / "bareMetalC" / "template.c"),
            kind="path",
            role="examples",
            tags=("gemmini", "scaffold", "exemplar"),
        ),
    ]
    if include_urls:
        sources.extend(
            [
                SourceRef(
                    locator="https://github.com/ucb-bar/gemmini/blob/master/README.md",
                    kind="url",
                    role="auto",
                    tags=("gemmini", "upstream"),
                ),
            ]
        )

    return SourceManifest(
        target_id=TARGET_ID,
        target_profile_ref=TARGET_PROFILE_REF,
        isa_family="rocc-systolic",
        sources=tuple(sources),
        static_hardware_facts=STATIC_FACTS,
        description=(
            "Gemmini systolic DNN accelerator (RoCC custom-3 opcode). "
            "Scratchpad+accumulator memory, configurable mesh/tile sizes, "
            "weight-stationary or output-stationary dataflow."
        ),
    )


__all__ = ["DEFAULT_SOURCE_ROOT", "STATIC_FACTS", "TARGET_ID", "TARGET_PROFILE_REF", "manifest", "source_root"]
