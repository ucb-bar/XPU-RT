"""Offline scaffolding for per-island DLC construction.

This module documents the contract for materialising the
``executor_artifact`` of a finer-than-whole-net island. It does
NOT execute end-to-end today — the user runs the offline pipeline
when they want per-op execution on board.

The pipeline:

1. Slice the source ONNX with ``onnx.utils.extract_model`` to get a
   sub-ONNX per island.
2. Convert each sub-ONNX to a DLC via the QAIRT Docker image
   (``models/qnn/Dockerfile.qnn-convert``) — the same path that
   produced today's whole-net DLCs.
3. Quantise (``qairt-quantizer``) for DSP / HTA / HTP backends.
4. Push the per-island DLCs to ``/root/contexts/<island_id>.dlc``.
5. (Optional) Pre-finalise on the board via
   ``qnn-context-binary-generator`` → ``<island_id>_<backend>.bin``.

The closed-loop planner *consumes* the resulting
``ArtifactRef``s through ``Island.executor_artifact``. When an
artifact is absent the planner flags the island
``planner_visible_only`` and refuses to schedule it.

Why not auto-run this from the agent loop? Two reasons:

* QAIRT conversion is interactive (Docker image, signed SDK
  download, large memory footprint). Better as a deliberate human
  step than as a side-effect of the MCP loop.
* Per-op ONNX slicing has correctness gotchas (initialisers,
  shape-inference, dynamic axes) that warrant offline iteration.

The functions below are pure planners over a manifest. The
actual conversion is invoked by the user via
``scripts/build_per_island_dlcs.py`` (out of scope for this plan
iteration; created when the user is ready for true per-op runs).
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

from xpu_rt.targets.backends.qnn.granularity_proposal import (
    ArtifactRef,
    GranularityProposal,
    Island,
)


@dataclasses.dataclass(frozen=True)
class IslandBuildPlan:
    """How one island should be materialised across backends.

    ``onnx_slice_inputs`` and ``onnx_slice_outputs`` are the
    boundary tensor names ``onnx.utils.extract_model`` needs;
    callers populate them from a graph dossier.
    """

    island_id: str
    onnx_slice_inputs: tuple[str, ...]
    onnx_slice_outputs: tuple[str, ...]
    quantise_for: tuple[str, ...]   # backends needing quantised DLC
    remote_dir: str = "/root/contexts"


def plan_island_builds(
    proposal: GranularityProposal,
    *,
    onnx_inputs: dict[str, tuple[str, ...]],
    onnx_outputs: dict[str, tuple[str, ...]],
    backends: tuple[str, ...] = ("CPU", "GPU", "DSP", "HTA", "HTP"),
    remote_dir: str = "/root/contexts",
) -> list[IslandBuildPlan]:
    """Produce a build plan for each island missing an artifact.

    Islands that already carry an ``executor_artifact`` for every
    backend they care about are skipped.
    """
    plans: list[IslandBuildPlan] = []
    for isl in proposal.islands:
        need_for: list[str] = []
        candidates = isl.backend_candidates or backends
        for b in candidates:
            if isl.executor_artifact.get(b) is None:
                need_for.append(b)
        if not need_for:
            continue
        quantise = tuple(b for b in need_for if b in ("DSP", "HTA", "HTP"))
        plans.append(IslandBuildPlan(
            island_id=isl.island_id,
            onnx_slice_inputs=tuple(onnx_inputs.get(isl.island_id, ())),
            onnx_slice_outputs=tuple(onnx_outputs.get(isl.island_id, ())),
            quantise_for=quantise,
            remote_dir=remote_dir,
        ))
    return plans


def register_built_artifacts(
    proposal: GranularityProposal,
    artifact_manifest_path: Path | str,
) -> GranularityProposal:
    """Update a proposal's islands with artifacts from a manifest.

    ``artifact_manifest_path`` is a JSON file written by the
    offline build pipeline:

        {"<island_id>": {"<backend>": {"remote_path": ...,
                                       "kind": "dlc" | "context_binary",
                                       "sha256": ...}}}

    Returns a new ``GranularityProposal`` with
    ``Island.executor_artifact`` populated. Cells absent from the
    manifest remain ``None`` (and the planner keeps treating those
    islands as planner-visible-only).
    """
    try:
        manifest = json.loads(Path(artifact_manifest_path).read_text())
    except (OSError, json.JSONDecodeError):
        return proposal
    new_islands: list[Island] = []
    for isl in proposal.islands:
        new_art = dict(isl.executor_artifact)
        entries = manifest.get(isl.island_id) or {}
        for b, info in entries.items():
            new_art[b] = ArtifactRef(
                remote_path=str(info["remote_path"]),
                kind=str(info["kind"]),
                backend=b,
                sha256=str(info.get("sha256", "")),
            )
        new_islands.append(dataclasses.replace(isl, executor_artifact=new_art))
    return dataclasses.replace(proposal, islands=tuple(new_islands))
