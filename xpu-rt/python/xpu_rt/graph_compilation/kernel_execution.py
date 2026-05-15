"""Kernel Execution Foundation.

Read-only consumer of 's `transformed_payload.real.mlir` artifact.
Compiles + executes the SetTileParams matmul region on real hardware
(GPU via Triton, CPU via libcompgen_rt + cffi C codegen) and verifies
numerical equality vs the eager baseline.

Layered alongside the FX-level evidence ( /
); never mutates any FX-level artifact.

Hard non-goals:

- No mutation of payload.mlir, candidate_actions.json, region_map.json,
  cost_preview_v2.json, llm_graph_view.json, or any readiness report.
- No new candidate generation, no new transforms.
- No compiler-core imports (compgen.ir / compgen.capture / compgen.pipeline).
- No raise into the pipeline. Best-effort with typed status fallbacks
  on every error path.

Output layout::

    02_graph_analysis/kernel_execution/
        compiled_kernel_run_gpu.json # .GPU artifact
        compiled_kernel_run_cpu.json # .CPU artifact
        kernel_execution_summary.md
        triton_kernel_<region>.py      # generated Triton source (when GPU runs)
        cpu_kernel_<region>.c          # generated CPU C source (when CPU runs)

Opt-in via ``COMPGEN_RUN_KERNELS=1``. Default OFF.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _utcnow() -> str:
    return datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


@dataclass(frozen=True)
class KernelExecutionResult:
    overall: str            # "ok" | "partial" | "not_run"
    out_dir: Path
    gpu_status: str         # "compiled" | "device_unavailable" | "compile_failed" | "run_failed" | "not_run" | "not_applicable"
    cpu_status: str         # "compiled" | "library_unavailable" | "compile_failed" | "run_failed" | "not_run" | "not_applicable"
    gpu_artifact: Path | None
    cpu_artifact: Path | None
    summary_md_path: Path


def _selected_set_tile_params(run_dir: Path) -> dict[str, Any] | None:
    """Return the real-transform manifest IFF the committed
    candidate is an executable SetTileParams. Otherwise None."""
    manifest_path = (
        run_dir / "03_recipe_planning" / "real_lowering"
        / "real_transform_manifest.json"
    )
    m = _read_json(manifest_path)
    if m is None:
        return None
    if m.get("real_transform_kind") not in (
        "executable_structured_ir", "executable_with_boundary_handling",
    ):
        return None
    selected = m.get("selected_recipe") or {}
    if selected.get("recipe_kind") != "SetTileParams":
        return None
    return m


def run_kernel_execution(run_dir: Path) -> KernelExecutionResult:
    """entry point. Reads the manifest; dispatches to GPU
    and CPU sub-tracks; emits typed artifacts and a short markdown
    summary. Best-effort on every step.
    """
    run_dir = Path(run_dir).resolve()
    ga = run_dir / "02_graph_analysis"
    out_dir = ga / "kernel_execution"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_md_path = out_dir / "kernel_execution_summary.md"

    manifest = _selected_set_tile_params(run_dir)
    if manifest is None:
        # Not applicable for this run (e.g. non-tile candidate, or
        # didn't run because stop_after was too early).
        body = (
            "# Kernel Execution — not_applicable\n\n"
            "- The committed candidate is not a SetTileParams executable "
            "real transform. M-19 only runs for SetTileParams; other "
            "candidate kinds are out of scope for this milestone.\n"
        )
        summary_md_path.write_text(body, encoding="utf-8")
        return KernelExecutionResult(
            overall="not_run", out_dir=out_dir,
            gpu_status="not_applicable", cpu_status="not_applicable",
            gpu_artifact=None, cpu_artifact=None,
            summary_md_path=summary_md_path,
        )

    # Pull the matmul shape + tile + region/candidate ids.
    sig = manifest.get("matmul_signature") or {}
    sel = manifest.get("selected_recipe") or {}
    M = int(sig.get("M") or 0)
    N = int(sig.get("N") or 0)
    K = int(sig.get("K") or 0)
    tile = sel.get("tile") or {}
    tM = int(tile.get("M") or 0)
    tN = int(tile.get("N") or 0)
    tK = int(tile.get("K") or 0)
    region_id = str(sel.get("region") or "")
    candidate_id = str(sel.get("selected_candidate_id") or "")
    recipe_op_id = str(sel.get("recipe_op_id") or "")

    common = {
        "schema_version": "compiled_kernel_run_v1",
        "model_id": manifest.get("model_id", ""),
        "target_id": manifest.get("target_id", ""),
        "region_id": region_id,
        "candidate_id": candidate_id,
        "recipe_op_id": recipe_op_id,
        "matmul_shape": {"M": M, "N": N, "K": K},
        "tile": {"M": tM, "N": tN, "K": tK},
        "transformed_payload_real_mlir":
            "03_recipe_planning/real_lowering/transformed_payload.real.mlir",
        "transformed_payload_real_mlir_sha256":
            manifest.get("outputs", {}).get(
                "transformed_payload_real_sha256", ""
            ),
        "iterations": 32,
        "warmup": 4,
    }

    # GPU sub-track.
    gpu_artifact: Path | None = None
    gpu_status = "not_run"
    try:
        from compgen.graph_compilation.kernel_execution_gpu import (
            run_gpu_track,
        )
        gpu_artifact = run_gpu_track(out_dir=out_dir, common=common)
        gpu_artifact_obj = _read_json(gpu_artifact) if gpu_artifact else None
        gpu_status = (
            (gpu_artifact_obj or {}).get("compile_status")
            or "not_run"
        )
    except Exception as exc:  # noqa: BLE001
        gpu_status = "internal_error"
        body = {
            **common, "track": "gpu_triton",
            "compile_status": "internal_error",
            "run_status": "not_run",
            "note": f"{type(exc).__name__}: {exc}",
            "generated_at_utc": _utcnow(),
        }
        gpu_artifact = out_dir / "compiled_kernel_run_gpu.json"
        gpu_artifact.write_text(
            json.dumps(body, indent=2, sort_keys=True), encoding="utf-8",
        )

    # CPU sub-track.
    cpu_artifact: Path | None = None
    cpu_status = "not_run"
    try:
        from compgen.graph_compilation.kernel_execution_cpu import (
            run_cpu_track,
        )
        cpu_artifact = run_cpu_track(out_dir=out_dir, common=common)
        cpu_artifact_obj = _read_json(cpu_artifact) if cpu_artifact else None
        cpu_status = (
            (cpu_artifact_obj or {}).get("compile_status")
            or "not_run"
        )
    except Exception as exc:  # noqa: BLE001
        cpu_status = "internal_error"
        body = {
            **common, "track": "cpu_compgen_rt",
            "compile_status": "internal_error",
            "run_status": "not_run",
            "note": f"{type(exc).__name__}: {exc}",
            "generated_at_utc": _utcnow(),
        }
        cpu_artifact = out_dir / "compiled_kernel_run_cpu.json"
        cpu_artifact.write_text(
            json.dumps(body, indent=2, sort_keys=True), encoding="utf-8",
        )

    # Summary.
    overall = "ok" if gpu_status == "compiled" or cpu_status == "compiled" else "partial"
    if gpu_status in ("not_applicable", "not_run") and cpu_status in (
        "not_applicable", "not_run",
    ):
        overall = "not_run"

    body = (
        f"# Kernel Execution — {overall}\n\n"
        f"- region: `{region_id}`\n"
        f"- candidate: `{candidate_id}`\n"
        f"- matmul: ({M}, {N}, {K})  tile: ({tM}, {tN}, {tK})\n"
        f"- GPU track: `{gpu_status}`\n"
        f"- CPU track: `{cpu_status}`\n"
    )
    summary_md_path.write_text(body, encoding="utf-8")

    return KernelExecutionResult(
        overall=overall, out_dir=out_dir,
        gpu_status=gpu_status, cpu_status=cpu_status,
        gpu_artifact=gpu_artifact, cpu_artifact=cpu_artifact,
        summary_md_path=summary_md_path,
    )
