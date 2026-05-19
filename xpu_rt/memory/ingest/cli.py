"""``xpu-rt-target`` CLI.

Wraps the universal ingestion pipeline + read-only inspectors with a
tiny click surface. The default flow is **headless** (Gemini, behind
the budget gate). The agent-in-loop flow lives in the
``/xpu-rt-target`` Claude Code skill — it can't be driven from the
shell without a Claude Code session attached.

Sub-commands:

  list       — list known seed manifests and persisted target cards
  show       — print a target's knowledge card as JSON
  seed       — run a known seed manifest (gemmini | saturn) headlessly
  ingest     — run an ad-hoc manifest from a path / URL headlessly
  lessons    — print recent lessons for a target
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click

from xpu_rt.memory import target_knowledge as tk
from xpu_rt.memory.ingest import IngestPipeline, SourceManifest, SourceRef
from xpu_rt.memory.seeds import gemmini as gemmini_seed
from xpu_rt.memory.seeds import saturn as saturn_seed


_SEEDS = {
    "gemmini": gemmini_seed,
    "saturn": saturn_seed,
}


@click.group(help="Build and inspect per-target knowledge cards.")
def main() -> None:
    """xpu-rt-target — per-target knowledge memory."""


@main.command("list")
def list_cmd() -> None:
    """List known seeds and persisted target cards."""
    click.echo("Known seeds (shipped with this build):")
    for name, module in _SEEDS.items():
        manifest = module.manifest()
        root = module.source_root()
        present = "yes" if root.is_dir() else "no"
        click.echo(
            f"  - {name}: target_id={manifest.target_id} "
            f"source_root={root} present={present}"
        )
    click.echo("")
    targets = tk.list_targets()
    if not targets:
        click.echo(f"No persisted cards at {tk.knowledge_root()} (yet).")
        return
    click.echo(f"Persisted cards at {tk.knowledge_root()}:")
    for tid in targets:
        click.echo(f"  - {tid}")


@main.command("show")
@click.argument("target_id")
@click.option("--json-out", "json_out", is_flag=True, help="Emit raw JSON.")
def show_cmd(target_id: str, json_out: bool) -> None:
    """Print the full TargetKnowledgeCard for TARGET_ID."""
    if not tk.exists(target_id):
        click.echo(f"no card for target_id={target_id!r}", err=True)
        sys.exit(2)
    card = tk.load(target_id)
    payload = card.to_dict()
    if json_out:
        click.echo(json.dumps(payload, indent=2))
        return
    spec = payload["hardware_spec"]
    click.echo(f"Target:         {card.target_id}")
    click.echo(f"Profile ref:    {card.target_profile_ref}")
    click.echo(f"ISA family:     {spec['isa_family']}")
    click.echo(f"Revision:       {card.revision} (updated {card.updated_at})")
    click.echo(f"Instructions:   {len(spec['instructions'])}")
    click.echo(f"Intrinsics:     {len(spec['intrinsics'])}")
    click.echo(f"Parameters:     {len(spec['parameters'])}")
    click.echo(f"Memory tiers:   {len(spec['memory_tiers'])}")
    click.echo(f"Exemplars:      {len(card.exemplars)}")
    click.echo(f"Lessons:        {sum(1 for _ in tk.iter_lessons(card))}")
    click.echo(f"Path:           {card.card_path}")


@main.command("seed")
@click.argument("seed", type=click.Choice(sorted(_SEEDS.keys())))
@click.option(
    "--include-urls",
    is_flag=True,
    help="Also fetch the seed's public URLs (requires `uv sync --extra ingest`).",
)
@click.option(
    "--model",
    default="gemini-2.5-flash",
    show_default=True,
    help="Gemini model to use for the headless router.",
)
def seed_cmd(seed: str, include_urls: bool, model: str) -> None:
    """Run a known seed manifest headlessly via Gemini.

    Respects the configured cumulative-USD budget cap. To run with
    Claude Code in the loop instead (no Gemini spend), invoke the
    `/xpu-rt-target` skill from a Claude Code session.
    """
    manifest = _SEEDS[seed].manifest(include_urls=include_urls)
    pipeline = IngestPipeline.from_gemini(model=model)
    click.echo(
        f"Running {seed} seed headlessly (model={model}, "
        f"sources={len(manifest.sources)})…",
        err=True,
    )
    card, report = pipeline.run(manifest)
    click.echo(json.dumps(_report_dict(card, report), indent=2))


@main.command("ingest")
@click.option("--target-id", required=True, help="Card id to populate.")
@click.option(
    "--target-profile-ref",
    default="",
    help="Path of the static target profile YAML this card cross-refs.",
)
@click.option(
    "--isa-family",
    default="other",
    show_default=True,
    help="ISA family label (rocc-systolic, riscv-rvv, cuda-sm, host-cpu, triton, other).",
)
@click.option(
    "--path",
    "paths",
    multiple=True,
    type=click.Path(exists=True),
    help="Local file/dir to ingest. Repeat for multiple sources.",
)
@click.option(
    "--url",
    "urls",
    multiple=True,
    help="URL to ingest (requires `uv sync --extra ingest`).",
)
@click.option(
    "--model",
    default="gemini-2.5-flash",
    show_default=True,
    help="Gemini model for the router.",
)
def ingest_cmd(
    target_id: str,
    target_profile_ref: str,
    isa_family: str,
    paths: tuple[str, ...],
    urls: tuple[str, ...],
    model: str,
) -> None:
    """Run an ad-hoc manifest headlessly."""
    sources: list[SourceRef] = []
    for p in paths:
        path = Path(p)
        sources.append(
            SourceRef(
                locator=str(path),
                kind="directory" if path.is_dir() else "path",
                role="auto",
            )
        )
    for u in urls:
        sources.append(SourceRef(locator=u, kind="url", role="auto"))
    if not sources:
        click.echo("at least one --path or --url is required", err=True)
        sys.exit(2)
    manifest = SourceManifest(
        target_id=target_id,
        target_profile_ref=target_profile_ref,
        isa_family=isa_family,
        sources=tuple(sources),
    )
    pipeline = IngestPipeline.from_gemini(model=model)
    card, report = pipeline.run(manifest)
    click.echo(json.dumps(_report_dict(card, report), indent=2))


@main.command("lessons")
@click.argument("target_id")
@click.option("--op-family", default="", help="Filter lessons by op family.")
@click.option("--dtype-class", default="", help="Filter lessons by dtype class.")
@click.option("--limit", default=10, show_default=True)
def lessons_cmd(target_id: str, op_family: str, dtype_class: str, limit: int) -> None:
    """Print recent lessons for a target."""
    if not tk.exists(target_id):
        click.echo(f"no card for target_id={target_id!r}", err=True)
        sys.exit(2)
    card = tk.load(target_id)
    rows = []
    for lesson in tk.iter_lessons(card):
        if op_family and lesson.op_family != op_family:
            continue
        if dtype_class and lesson.dtype_class != dtype_class:
            continue
        rows.append(lesson)
    rows.sort(key=lambda l: l.timestamp, reverse=True)
    for l in rows[:limit]:
        click.echo(
            f"[{l.timestamp}] op={l.op_family} dt={l.dtype_class} "
            f"layout={l.layout_kind} action={l.action} gain={l.measured_gain:.2f}"
            + (f" notes={l.notes!r}" if l.notes else "")
        )


def _report_dict(card: tk.TargetKnowledgeCard, report) -> dict:  # type: ignore[no-untyped-def]
    return {
        "target_id": card.target_id,
        "card_path": str(card.card_path),
        "revision": card.revision,
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
    }


if __name__ == "__main__":
    main()
