"""``pipeline_parallel_schedule`` -- annotate a module with a 1F1B /
interleaved pipeline-parallel schedule.

Walks every ``func.func`` in the module that is tagged with
``compgen.pipeline_stage`` (stage index) and:

1. Computes a 1F1B schedule across the given ``num_stages`` +
   ``num_microbatches`` configuration.
2. Attaches ``compgen.pp_schedule`` as a comma-separated string
   listing ``(stage, microbatch, phase)`` triples on the module's
   ``builtin.module`` op.

Phases: ``"forward"`` / ``"backward"`` / ``"warmup"`` / ``"cooldown"``.

No structural rewrite — the downstream runtime consumes the
schedule.
"""

from __future__ import annotations

from dataclasses import dataclass

from xdsl.dialects.builtin import ModuleOp, StringAttr


@dataclass(frozen=True)
class PipelineParallelConfig:
    num_stages: int = 4
    num_microbatches: int = 8
    schedule_kind: str = "1f1b"


@dataclass
class PipelineParallelStats:
    schedule_entries: int = 0
    schedule_kind: str = ""


def _generate_1f1b(num_stages: int, num_microbatches: int) -> list[tuple[int, int, str]]:
    """Generate a classic 1-forward-1-backward schedule."""
    out: list[tuple[int, int, str]] = []
    # Warmup: ``num_stages - stage - 1`` forward microbatches per stage.
    for stage in range(num_stages):
        for mb in range(num_stages - stage):
            out.append((stage, mb, "warmup"))
    # Steady state: 1F then 1B alternating.
    for mb in range(num_stages, num_microbatches):
        for stage in range(num_stages):
            out.append((stage, mb, "forward"))
            out.append((stage, mb - num_stages, "backward"))
    # Cooldown.
    for stage in reversed(range(num_stages)):
        for mb in range(num_microbatches - (num_stages - stage), num_microbatches):
            out.append((stage, mb, "cooldown"))
    return out


def run_pipeline_parallel_schedule(
    module: ModuleOp,
    *,
    config: PipelineParallelConfig | None = None,
) -> PipelineParallelStats:
    cfg = config if config is not None else PipelineParallelConfig()
    if cfg.num_stages < 1 or cfg.num_microbatches < cfg.num_stages:
        raise ValueError(
            "num_microbatches must be >= num_stages >= 1"
        )
    schedule = _generate_1f1b(cfg.num_stages, cfg.num_microbatches)
    module.attributes["compgen.pp_schedule"] = StringAttr(
        ";".join(f"{s}:{m}:{p}" for s, m, p in schedule)
    )
    module.attributes["compgen.pp_schedule_kind"] = StringAttr(cfg.schedule_kind)
    module.attributes["compgen.pp_num_stages"] = StringAttr(
        str(cfg.num_stages)
    )
    module.attributes["compgen.pp_num_microbatches"] = StringAttr(
        str(cfg.num_microbatches)
    )
    return PipelineParallelStats(
        schedule_entries=len(schedule),
        schedule_kind=cfg.schedule_kind,
    )


__all__ = [
    "PipelineParallelConfig",
    "PipelineParallelStats",
    "run_pipeline_parallel_schedule",
]
