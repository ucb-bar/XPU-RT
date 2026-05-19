"""Saturn source manifest.

Saturn is the Berkeley parameterized RISC-V V-extension (RVV 1.0)
vector unit. The manifest names the AsciiDoc reference docs, the Scala
parameters file, and a curated set of vec/opu benchmark kernels for
the ingestion pipeline to consume. No regex parsing here — same shape
as :mod:`xpu_rt.memory.seeds.gemmini`.

Override the source root with ``XPU_RT_CHIPYARD_SATURN_ROOT``.
"""

from __future__ import annotations

import os
from pathlib import Path

from xpu_rt.memory.ingest import SourceManifest, SourceRef
from xpu_rt.memory.target_knowledge import MemoryTierSpec

DEFAULT_SOURCE_ROOT = Path("/scratch2/agustin/chipyard/generators/saturn")
TARGET_ID = "saturn_opu_v128"
TARGET_PROFILE_REF = "configs/targets/saturn_opu_v128.yaml"


# Saturn's memory hierarchy is configurable; these defaults match the
# reference (``refParams``) preset. Override per-instance via ingestion
# of design-space.adoc, which the LLM router will surface.
STATIC_FACTS: dict[str, object] = {
    "memory_tiers": (
        MemoryTierSpec(name="vrf", kind="registers", size_bytes=8 * 1024),
        MemoryTierSpec(name="l2", kind="l2", size_bytes=512 * 1024),
        MemoryTierSpec(name="dram", kind="dram"),
    ),
    "dataflow_modes": ("rvv1.0", "opu-outer-product"),
    "constraints": (
        "RVV 1.0 conformant; VLEN is a configurable parameter",
        "OPU instructions live behind the xopu custom extension",
        "DLEN <= VLEN; element width fixed at SEW per vsetvli",
    ),
}

# Curated benchmark subset — broad enough to span vector + OPU kernels
# without dragging in every vec-* directory (50+ entries).
_BENCHMARK_DIRS: tuple[str, ...] = (
    "vec-sgemm",
    "vec-sgemv",
    "vec-daxpy",
    "vec-dotprod",
    "vec-fft",
    "vec-softmax",
    "vec-iconv2d",
    "opu-gemm",
    "opu-m4-transpose",
    "opu-fused-gemm-transpose",
)


def source_root() -> Path:
    env = os.environ.get("XPU_RT_CHIPYARD_SATURN_ROOT")
    return Path(env) if env else DEFAULT_SOURCE_ROOT


def manifest(*, include_urls: bool = False) -> SourceManifest:
    """Build the ingestion manifest for the Saturn target."""
    root = source_root()
    sources: list[SourceRef] = [
        # README first so the router gets target context.
        SourceRef(
            locator=str(root / "README.md"),
            kind="path",
            role="auto",
            tags=("saturn", "readme"),
        ),
        # AsciiDoc reference manual.
        SourceRef(
            locator=str(root / "docs"),
            kind="directory",
            glob="*.adoc",
            role="auto",
            max_depth=1,
            tags=("saturn", "ref-manual"),
        ),
        # Scala parameter sweep — defaults + named presets.
        SourceRef(
            locator=str(root / "src" / "main" / "scala" / "common" / "Parameters.scala"),
            kind="path",
            role="architecture",
            tags=("saturn", "scala-config", "presets"),
        ),
    ]
    # Curated exemplars.
    bench_root = root / "benchmarks"
    for sub in _BENCHMARK_DIRS:
        bench_dir = bench_root / sub
        sources.append(
            SourceRef(
                locator=str(bench_dir),
                kind="directory",
                glob="*.c",
                role="examples",
                max_depth=1,
                tags=("saturn", sub),
            )
        )
        # Include any .S asm too — RVV assembly is the most informative
        # exemplar shape for the router.
        sources.append(
            SourceRef(
                locator=str(bench_dir),
                kind="directory",
                glob="*.S",
                role="examples",
                max_depth=1,
                tags=("saturn", sub, "asm"),
            )
        )

    if include_urls:
        sources.append(
            SourceRef(
                locator="https://saturn-vectors.org/",
                kind="url",
                role="auto",
                tags=("saturn", "upstream-manual"),
            )
        )

    return SourceManifest(
        target_id=TARGET_ID,
        target_profile_ref=TARGET_PROFILE_REF,
        isa_family="riscv-rvv",
        sources=tuple(sources),
        static_hardware_facts=STATIC_FACTS,
        description=(
            "Saturn — Berkeley parameterized RVV 1.0 vector unit with "
            "optional OPU (outer-product unit) custom extension."
        ),
    )


__all__ = ["DEFAULT_SOURCE_ROOT", "STATIC_FACTS", "TARGET_ID", "TARGET_PROFILE_REF", "manifest", "source_root"]
