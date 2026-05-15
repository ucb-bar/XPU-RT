"""Multi-level IR analysis snapshot emitter.

Reads a real ``compile_model()`` output directory and produces the
8 IR snapshot files under
``<run_dir>/02_graph_analysis/analysis_snapshots/``.

Approach is **post-hoc**: rather than instrumenting every pipeline
stage (which would be invasive and high-risk), this module scans
the artifacts each stage already writes to disk and synthesizes
:class:`compgen.analysis.ir_snapshots.IRAnalysisSnapshot` records.
Levels with no producer on disk emit a typed ``not_available``
with the specific reason (``stage_not_run`` /
``artifact_missing``).

Mapping (each level → source artifacts inspected):

* ``fx_graph``         → ``exported_program.pt2`` + ``graph_breaks.json``
* ``payload_ir``       → ``payload.mlir``
* ``recipe_ir``        → ``recipe*.yaml`` + ``gap_analysis.json``
* ``tile_ir``          → ``transforms/*.mlir``
* ``dialect_ir``       → ``generated_kernels/*/`` per-provider dirs
* ``kernel_artifact``  → ``generated_kernels/`` flattened
* ``execution_plan``   → ``execution_plan.yaml`` + ``memory_plan.yaml``
* ``runtime_profile``  → ``verification_report.json`` profiler tail

The emitter is **idempotent** and **safe to re-run** — it writes
fresh snapshot files each call.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from compgen.analysis.ir_snapshots import (
    IRAnalysisSnapshot,
    RegionSummary,
    UnsupportedProvider,
    make_available,
    make_not_available,
    write_snapshots,
)


@dataclass(frozen=True)
class EmitResult:
    """Per-level emit outcome."""

    level: str
    status: str
    path: Path
    region_count: int = 0
    not_available_reason: str = ""


def _read_json(path: Path) -> dict | list | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _read_yaml(path: Path) -> Any:
    if not path.is_file():
        return None
    try:
        return yaml.safe_load(path.read_text())
    except (yaml.YAMLError, OSError):
        return None


def _peek_mlir_ops(mlir_path: Path, *, limit: int = 40) -> list[str]:
    """Scan an MLIR text file and pull out distinct op names.

    Reads up to ``limit`` distinct op names (e.g.
    ``linalg.matmul``, ``arith.addf``). Returns them in the order
    they were first seen so the snapshot reflects the actual
    structure of the IR rather than a deduplicated set.
    """

    if not mlir_path.is_file():
        return []
    seen: list[str] = []
    seen_set: set[str] = set()
    try:
        for line in mlir_path.read_text().splitlines():
            stripped = line.strip()
            # Crude but effective: look for ``foo.bar(`` or ``foo.bar `` patterns.
            for token in stripped.split():
                if "." in token and "(" in token:
                    op = token.split("(", 1)[0]
                    if 2 <= len(op.split(".")[-1]) <= 60 and op not in seen_set:
                        seen.append(op)
                        seen_set.add(op)
                        if len(seen) >= limit:
                            return seen
    except OSError:
        pass
    return seen


# ---------------------------------------------------------------------------
# Per-level builders
# ---------------------------------------------------------------------------


def _build_fx_snapshot(run_dir: Path) -> IRAnalysisSnapshot:
    pt2 = run_dir / "exported_program.pt2"
    breaks = _read_json(run_dir / "graph_breaks.json")
    if not pt2.is_file() and breaks is None:
        return make_not_available(
            level="fx_graph",
            reason="artifact_missing",
            detail="no exported_program.pt2 or graph_breaks.json in run_dir",
        )
    regions: list[RegionSummary] = []
    if isinstance(breaks, dict):
        for i, region_info in enumerate(breaks.get("regions", []) or []):
            regions.append(
                RegionSummary(
                    region_id=str(region_info.get("region_id", f"fx_region_{i}")),
                    ops=tuple(region_info.get("ops", ())),
                    extras={k: v for k, v in region_info.items() if k not in ("region_id", "ops")},
                )
            )
    if not regions:
        # We saw the artifact but couldn't extract regions — emit a
        # single placeholder region so the agent surface is populated.
        regions = [
            RegionSummary(
                region_id="fx_root",
                ops=("torch.export",),
                extras={"source_pt2_present": pt2.is_file()},
            )
        ]
    return make_available(
        level="fx_graph",
        source_artifact=str(pt2 if pt2.is_file() else run_dir / "graph_breaks.json"),
        regions=regions,
        detail=f"{len(regions)} region(s) from FX export",
    )


def _build_payload_snapshot(run_dir: Path) -> IRAnalysisSnapshot:
    mlir = run_dir / "payload.mlir"
    if not mlir.is_file():
        return make_not_available(
            level="payload_ir",
            reason="artifact_missing",
            detail=f"no payload.mlir at {mlir}",
        )
    ops = _peek_mlir_ops(mlir)
    if not ops:
        return make_available(
            level="payload_ir",
            source_artifact=str(mlir),
            regions=[
                RegionSummary(
                    region_id="payload_root",
                    ops=(),
                    extras={"note": "payload.mlir present but no ops parsed"},
                )
            ],
        )
    return make_available(
        level="payload_ir",
        source_artifact=str(mlir),
        regions=[
            RegionSummary(
                region_id="payload_root",
                ops=tuple(ops),
                fusion_candidates=tuple(
                    op for op in ops if "linalg" in op or "arith" in op
                )[:8],
            )
        ],
        detail=f"{len(ops)} distinct op kinds parsed from payload.mlir",
    )


def _build_recipe_snapshot(run_dir: Path) -> IRAnalysisSnapshot:
    candidates = list(run_dir.glob("recipe*.yaml"))
    gap = _read_json(run_dir / "gap_analysis.json")
    if not candidates and gap is None:
        return make_not_available(
            level="recipe_ir",
            reason="artifact_missing",
            detail="no recipe*.yaml or gap_analysis.json on disk",
        )
    regions: list[RegionSummary] = []
    if isinstance(gap, dict):
        for i, item in enumerate(gap.get("decisions", []) or []):
            regions.append(
                RegionSummary(
                    region_id=str(item.get("region_id", f"recipe_decision_{i}")),
                    ops=tuple(item.get("ops", ())),
                    extras={k: v for k, v in item.items() if k not in ("region_id", "ops")},
                )
            )
    if not regions:
        regions = [
            RegionSummary(
                region_id="recipe_root",
                ops=(),
                extras={"recipe_files_count": len(candidates)},
            )
        ]
    return make_available(
        level="recipe_ir",
        source_artifact=str(
            candidates[0] if candidates else run_dir / "gap_analysis.json"
        ),
        regions=regions,
        detail=f"{len(regions)} recipe decision(s)",
    )


def _build_tile_snapshot(run_dir: Path) -> IRAnalysisSnapshot:
    transforms = list((run_dir / "transforms").glob("*.mlir")) if (run_dir / "transforms").is_dir() else []
    if not transforms:
        return make_not_available(
            level="tile_ir",
            reason="artifact_missing",
            detail="no transforms/*.mlir on disk",
        )
    regions: list[RegionSummary] = []
    for t in transforms:
        ops = _peek_mlir_ops(t, limit=12)
        regions.append(
            RegionSummary(
                region_id=t.stem,
                ops=tuple(ops),
                extras={"source": str(t)},
            )
        )
    return make_available(
        level="tile_ir",
        source_artifact=str(transforms[0].parent),
        regions=regions,
        detail=f"{len(regions)} transform script(s)",
    )


def _build_dialect_snapshot(run_dir: Path) -> IRAnalysisSnapshot:
    gen = run_dir / "generated_kernels"
    if not gen.is_dir():
        return make_not_available(
            level="dialect_ir",
            reason="stage_not_run",
            detail="no generated_kernels/ directory",
        )
    by_provider = [p for p in gen.iterdir() if p.is_dir()]
    if not by_provider:
        return make_not_available(
            level="dialect_ir",
            reason="artifact_missing",
            detail="generated_kernels/ has no subdirectories",
        )
    regions: list[RegionSummary] = []
    for p in sorted(by_provider):
        ops: tuple[str, ...] = ()
        # Look for any .mlir under this provider's dir.
        mlir_files = list(p.rglob("*.mlir"))
        if mlir_files:
            ops = tuple(_peek_mlir_ops(mlir_files[0], limit=10))
        regions.append(
            RegionSummary(
                region_id=f"dialect:{p.name}",
                ops=ops,
                supported_providers=(p.name,),
                extras={"artifacts": len(list(p.iterdir()))},
            )
        )
    return make_available(
        level="dialect_ir",
        source_artifact=str(gen),
        regions=regions,
        detail=f"{len(regions)} dialect provider(s) produced artifacts",
    )


def _build_kernel_artifact_snapshot(run_dir: Path) -> IRAnalysisSnapshot:
    gen = run_dir / "generated_kernels"
    if not gen.is_dir():
        return make_not_available(
            level="kernel_artifact",
            reason="stage_not_run",
            detail="no generated_kernels/ directory",
        )
    all_files = [p for p in gen.rglob("*") if p.is_file()]
    if not all_files:
        return make_not_available(
            level="kernel_artifact",
            reason="artifact_missing",
            detail="generated_kernels/ is empty",
        )
    from collections import Counter

    ext_counts = Counter(p.suffix for p in all_files if p.suffix)
    return make_available(
        level="kernel_artifact",
        source_artifact=str(gen),
        regions=[
            RegionSummary(
                region_id="kernel_artifacts",
                ops=tuple(sorted(ext_counts.keys())),
                extras={
                    "file_count": len(all_files),
                    "extension_counts": dict(ext_counts),
                },
            )
        ],
        detail=f"{len(all_files)} kernel-artifact file(s)",
    )


def _build_execution_plan_snapshot(run_dir: Path) -> IRAnalysisSnapshot:
    plan = _read_yaml(run_dir / "execution_plan.yaml")
    mem = _read_yaml(run_dir / "memory_plan.yaml")
    if plan is None and mem is None:
        return make_not_available(
            level="execution_plan",
            reason="stage_not_run",
            detail="no execution_plan.yaml or memory_plan.yaml on disk",
        )
    region_summaries: list[RegionSummary] = []
    if isinstance(plan, dict):
        for i, op in enumerate(plan.get("operations") or plan.get("ops") or []):
            if isinstance(op, dict):
                region_summaries.append(
                    RegionSummary(
                        region_id=str(op.get("region_id", op.get("name", f"plan_op_{i}"))),
                        ops=(str(op.get("kind", op.get("op", "op"))),),
                        extras={k: v for k, v in op.items() if isinstance(v, (str, int, float, bool))},
                    )
                )
    if not region_summaries:
        region_summaries = [
            RegionSummary(
                region_id="execution_plan_root",
                ops=(),
                extras={
                    "plan_present": plan is not None,
                    "memory_plan_present": mem is not None,
                },
            )
        ]
    return make_available(
        level="execution_plan",
        source_artifact=str(
            run_dir / ("execution_plan.yaml" if plan is not None else "memory_plan.yaml")
        ),
        regions=region_summaries,
        detail=f"{len(region_summaries)} planned op(s)",
    )


def _build_runtime_profile_snapshot(run_dir: Path) -> IRAnalysisSnapshot:
    ver = _read_json(run_dir / "verification_report.json")
    if ver is None:
        return make_not_available(
            level="runtime_profile",
            reason="stage_not_run",
            detail="no verification_report.json on disk",
        )
    levels = ver.get("levels_run", []) if isinstance(ver, dict) else []
    return make_available(
        level="runtime_profile",
        source_artifact=str(run_dir / "verification_report.json"),
        regions=[
            RegionSummary(
                region_id="verifier_summary",
                ops=tuple(levels) if levels else (),
                extras={
                    "passed": (
                        ver.get("status") if isinstance(ver, dict) else None
                    ),
                    "levels_passed": (
                        ver.get("levels_passed") if isinstance(ver, dict) else []
                    ),
                },
            )
        ],
        detail=f"verification levels: {len(levels)}",
    )


# ---------------------------------------------------------------------------
# Top-level emitter
# ---------------------------------------------------------------------------


def emit_snapshots_for_run(
    run_dir: str | Path,
    *,
    snapshots_dir_name: str = "02_graph_analysis/analysis_snapshots",
) -> dict[str, EmitResult]:
    """Read ``run_dir`` and write the 8 snapshot files.

    Returns a {level: EmitResult} map so the caller can audit
    which levels populated vs were typed-not-available.
    """

    rd = Path(run_dir)
    if not rd.is_dir():
        raise FileNotFoundError(f"run_dir {rd} is not a directory")
    builders = {
        "fx_graph": _build_fx_snapshot,
        "payload_ir": _build_payload_snapshot,
        "recipe_ir": _build_recipe_snapshot,
        "tile_ir": _build_tile_snapshot,
        "dialect_ir": _build_dialect_snapshot,
        "kernel_artifact": _build_kernel_artifact_snapshot,
        "execution_plan": _build_execution_plan_snapshot,
        "runtime_profile": _build_runtime_profile_snapshot,
    }
    snapshots = {level: builder(rd) for level, builder in builders.items()}
    out_dir = rd / snapshots_dir_name
    paths = write_snapshots(snapshots, out_dir)
    return {
        level: EmitResult(
            level=level,
            status=snap.status,
            path=paths[level],
            region_count=len(snap.regions),
            not_available_reason=snap.not_available_reason,
        )
        for level, snap in snapshots.items()
    }


__all__ = ["EmitResult", "emit_snapshots_for_run"]
