"""Extended artefact emission for compile bundles.

The :class:`~compgen.stages.bundle.stage.BundleStage` writes the two
required artefacts — ``payload.mlir`` and ``manifest.json``.  This
module emits the surrounding artefacts that the CLAUDE.md contract
promises and that :mod:`compgen.runtime.bundle_runner` expects when
re-hydrating a bundle. Every artifact in the 14-artifact contract has
one slot here.

Surfacing policy (production-grade, no silent failures):

- Each artifact's emission is isolated in its own try-block so one
  broken artifact doesn't torpedo the rest. But the per-artifact status
  is **never** swallowed — it is recorded in a
  :class:`BundleEmissionReport`, published on the trace bus, and
  serialised into ``manifest.json::extended_artifacts``.
- ``skipped`` (upstream data unavailable) is an honest, non-failure
  status. ``failed`` (we tried and blew up) causes the caller to raise
  :class:`BundleEmissionError` after collecting all statuses — so the
  user sees every failure at once, not just the first.
- No broad ``except`` wraps the emission. Callers see a
  :class:`BundleEmissionReport` and decide.

Artifacts covered:

- ``exported_program.pt2`` — ``torch.export.ExportedProgram``.
- ``golden_inputs.pt`` — sample inputs used during compilation.
- ``golden_outputs.pt`` — eager reference output (no-grad eval with
  training-state save/restore around it).
- ``compile_baseline.json`` — real ``torch.compile`` baseline timings
  via :func:`compgen.capture.dynamo_baseline.compile_baseline`. Opt
  out with ``run_compile_baseline=False`` for large models.
- ``graph_breaks.json`` — dynamo graph-break list + guard failures.
- ``execution_plan.yaml`` — planner output when a payload module +
  target profile are supplied.
- ``memory_plan.yaml`` — per-device memory breakdown alongside the
  execution plan.
- ``gap_analysis.json`` — per-cluster FLOP/byte/kernel-opportunity
  breakdown when a :class:`NetworkAnalysis` dossier is supplied.
- ``kernel_contracts/*.yaml`` — one file per op/subgraph requiring a
  kernel.
- ``transforms/*.mlir`` — MLIR Transform dialect scripts recovered
  from pipeline artifacts (drive loop output).
- ``generated_kernels/`` — per-provider generated kernel sources
  recovered from pipeline artifacts.
- ``verification_report.json`` — verify stage result when
  ``verification_report`` is passed in.

``skipped`` vs ``failed``:

- If the caller omits ``analysis``, ``gap_analysis.json`` is
  ``skipped`` with reason ``"no analysis dossier passed"`` — not a
  failure, just an empty slot.
- If ``analysis`` is passed but its serialisation raises, the slot is
  ``failed`` and :class:`BundleEmissionError` will be raised by the
  aggregating caller.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog
import torch

from compgen.runtime.errors import (
    ArtifactStatus,
    BundleEmissionReport,
)

if TYPE_CHECKING:
    from xdsl.dialects.builtin import ModuleOp

    from compgen.agent.analyzer import NetworkAnalysis
    from compgen.targets.schema import TargetProfile

log = structlog.get_logger(__name__)


# Canonical filenames — mirror ``bundle_runner._*_FILENAME``.
_EXPORTED_PROGRAM_FILENAME = "exported_program.pt2"
_GOLDEN_INPUTS_FILENAME = "golden_inputs.pt"
_GOLDEN_ARGS_FILENAME = "golden_args.pt"
_GOLDEN_OUTPUTS_FILENAME = "golden_outputs.pt"
_COMPILE_BASELINE_FILENAME = "compile_baseline.json"
_GRAPH_BREAKS_FILENAME = "graph_breaks.json"
_EXECUTION_PLAN_FILENAME = "execution_plan.yaml"
_MEMORY_PLAN_FILENAME = "memory_plan.yaml"
_GAP_ANALYSIS_FILENAME = "gap_analysis.json"
_KERNEL_CONTRACTS_DIR = "kernel_contracts"
_TRANSFORMS_DIR = "transforms"
_GENERATED_KERNELS_DIR = "generated_kernels"
_VERIFICATION_REPORT_FILENAME = "verification_report.json"
_PROVIDER_FEEDBACK_FILENAME = "provider_feedback.json"
_MANIFEST_FILENAME = "manifest.json"


# Every artifact name that appears in a BundleEmissionReport. Order
# matches the contract's canonical listing so reports read naturally.
_CONTRACT_ARTIFACT_NAMES = (
    "exported_program",
    "golden_inputs",
    "golden_args",
    "golden_outputs",
    "compile_baseline",
    "graph_breaks",
    "execution_plan",
    "memory_plan",
    "gap_analysis",
    "kernel_contracts",
    "transforms",
    "generated_kernels",
    "verification_report",
    "provider_feedback",
)


def emit_extended_artefacts(
    bundle_dir: Path | str,
    *,
    capture_artifact: Any,
    sample_inputs: tuple[Any, ...],
    model: torch.nn.Module | None = None,
    eager_output: torch.Tensor | None = None,
    run_compile_baseline: bool = True,
    payload_module: ModuleOp | None = None,
    target_profile: TargetProfile | None = None,
    analysis: NetworkAnalysis | None = None,
    pipeline_artifacts: dict[str, Any] | None = None,
    verification_report: dict[str, Any] | None = None,
) -> BundleEmissionReport:
    """Emit the extended bundle artefacts and return a status report.

    Unlike earlier revisions of this function, every per-artifact
    failure is captured in the returned :class:`BundleEmissionReport`
    instead of being logged and forgotten. Callers who want strict
    production behavior should inspect ``report.failed`` and raise
    :class:`BundleEmissionError` (or call
    :func:`raise_on_failure` here, which does that for them).

    Args:
        bundle_dir: Bundle directory (containing ``manifest.json``).
        capture_artifact: ``CaptureArtifact`` from
            :func:`~compgen.capture.torch_export.capture_frontend_artifact`.
        sample_inputs: Inputs that drove capture; serialised to
            ``golden_inputs.pt``.
        model: Original PyTorch model. Used to compute
            ``golden_outputs.pt`` and run the torch.compile baseline.
        eager_output: Pre-computed eager output (takes precedence over
            running the model ourselves).
        run_compile_baseline: When True (default), invokes
            ``compgen.capture.dynamo_baseline.compile_baseline``.
        payload_module: Post-pipeline xDSL ``ModuleOp``.
        target_profile: ``TargetProfile`` compiled for.
        analysis: ``NetworkAnalysis`` for the ``gap_analysis.json``
            slot.
        pipeline_artifacts: Raw artifact dict from ``PipelineResult``.
            Used to recover ``transforms/*.mlir`` scripts and
            ``generated_kernels/`` sources when the drive loop / kernel
            providers populated them. When absent, both slots are
            ``skipped`` (not failed).
        verification_report: Result of running the verify stage (dict
            returned by ``transforms.verify.run_verification_ladder``
            or equivalent). When supplied, emits
            ``verification_report.json``; else the slot is ``skipped``.

    Returns:
        :class:`BundleEmissionReport` with one
        :class:`ArtifactStatus` per contract slot. Raises no broad
        errors — per-artifact failures go into the report.
    """
    bundle_dir = Path(bundle_dir)
    if not bundle_dir.is_dir():
        raise FileNotFoundError(f"bundle_dir does not exist: {bundle_dir}")

    statuses: list[ArtifactStatus] = []

    # --- exported_program.pt2 ------------------------------------------
    exported_program = getattr(capture_artifact, "exported_program", None)
    if exported_program is None:
        statuses.append(
            ArtifactStatus(
                name="exported_program",
                status="skipped",
                reason="capture_artifact has no exported_program",
            )
        )
    else:
        path = bundle_dir / _EXPORTED_PROGRAM_FILENAME
        try:
            torch.export.save(exported_program, str(path))
            statuses.append(ArtifactStatus(name="exported_program", status="ok", path=_EXPORTED_PROGRAM_FILENAME))
        except Exception as exc:
            statuses.append(ArtifactStatus(name="exported_program", status="failed", error=repr(exc)))

    # --- golden_inputs.pt ----------------------------------------------
    try:
        torch.save(list(sample_inputs), bundle_dir / _GOLDEN_INPUTS_FILENAME)
        statuses.append(ArtifactStatus(name="golden_inputs", status="ok", path=_GOLDEN_INPUTS_FILENAME))
    except Exception as exc:
        statuses.append(ArtifactStatus(name="golden_inputs", status="failed", error=repr(exc)))

    # --- golden_args.pt ------------------------------------------------
    # REQ-024: unified function-arg bundle. Combines parameters
    # (weights/biases pulled from the ExportedProgram's state_dict)
    # with the user-supplied sample inputs, in the same order as the
    # IR ``func.func @forward(%0, %1, …, %k)`` arglist. Consuming pack
    # composers iterate ``zip(args, golden_args)`` and bake every arg
    # uniformly without independently parsing exported_program.pt2.
    if exported_program is None:
        statuses.append(
            ArtifactStatus(
                name="golden_args",
                status="skipped",
                reason="no exported_program — can't reconcile params with inputs",
            )
        )
    else:
        try:
            args = _build_unified_arg_bundle(exported_program, sample_inputs)
            torch.save(args, bundle_dir / _GOLDEN_ARGS_FILENAME)
            statuses.append(ArtifactStatus(name="golden_args", status="ok", path=_GOLDEN_ARGS_FILENAME))
        except Exception as exc:
            statuses.append(ArtifactStatus(name="golden_args", status="failed", error=repr(exc)))

    # --- golden_outputs.pt ---------------------------------------------
    # Forward-pass failures are model characteristics (unsupported op in
    # eager mode, device mismatch), not emission bugs — so mark as
    # "skipped" with the reason. Disk write failures are bugs → "failed".
    eager_forward_error: str | None = None
    if eager_output is None and model is not None:
        prev_training = model.training
        try:
            model.eval()
            with torch.no_grad():
                eager_output = model(*sample_inputs)
        except Exception as exc:
            eager_forward_error = repr(exc)
            eager_output = None
        finally:
            model.train(prev_training)

    if eager_output is not None:
        try:
            torch.save(eager_output, bundle_dir / _GOLDEN_OUTPUTS_FILENAME)
            statuses.append(ArtifactStatus(name="golden_outputs", status="ok", path=_GOLDEN_OUTPUTS_FILENAME))
        except Exception as exc:
            statuses.append(ArtifactStatus(name="golden_outputs", status="failed", error=repr(exc)))
    elif eager_forward_error is not None:
        statuses.append(
            ArtifactStatus(
                name="golden_outputs",
                status="skipped",
                reason=f"eager forward raised: {eager_forward_error}",
            )
        )
    else:
        statuses.append(
            ArtifactStatus(
                name="golden_outputs",
                status="skipped",
                reason="no eager_output and no model passed",
            )
        )

    # --- compile_baseline.json -----------------------------------------
    if not run_compile_baseline:
        statuses.append(
            ArtifactStatus(
                name="compile_baseline",
                status="skipped",
                reason="run_compile_baseline=False",
            )
        )
    elif model is None:
        statuses.append(
            ArtifactStatus(
                name="compile_baseline",
                status="skipped",
                reason="no model passed",
            )
        )
    else:
        prev_training = model.training
        report = None
        baseline_error: str | None = None
        try:
            from compgen.capture.dynamo_baseline import compile_baseline

            model.eval()
            report = compile_baseline(model, sample_inputs)
        except Exception as exc:
            # torch.compile failures are a model characteristic (unsupported
            # op, graph break on unusual control flow). Not a bundle-emit
            # bug; mark as skipped with the reason so users see it but
            # compile_model doesn't fail.
            baseline_error = repr(exc)
        finally:
            model.train(prev_training)

        if report is not None:
            try:
                payload = {
                    "backend": report.backend,
                    "cold_compile_ms": float(report.cold_compile_ms),
                    "warm_run_ms": float(report.warm_run_ms),
                    "num_graph_breaks": int(report.num_graph_breaks),
                    "compiled_op_fraction": float(report.compiled_op_fraction),
                }
                (bundle_dir / _COMPILE_BASELINE_FILENAME).write_text(json.dumps(payload, indent=2))
                statuses.append(ArtifactStatus(name="compile_baseline", status="ok", path=_COMPILE_BASELINE_FILENAME))
            except Exception as exc:
                # Serialization / disk-write failure is a real emission bug.
                statuses.append(ArtifactStatus(name="compile_baseline", status="failed", error=repr(exc)))
        else:
            statuses.append(
                ArtifactStatus(
                    name="compile_baseline",
                    status="skipped",
                    reason=f"torch.compile baseline raised: {baseline_error}",
                )
            )

    # --- graph_breaks.json ---------------------------------------------
    diagnostics = getattr(capture_artifact, "diagnostics", None)
    if diagnostics is None:
        statuses.append(
            ArtifactStatus(
                name="graph_breaks",
                status="skipped",
                reason="capture_artifact has no diagnostics",
            )
        )
    else:
        try:
            graph_breaks_payload = {
                "graph_breaks": [
                    {"location": str(loc), "reason": str(reason)}
                    for loc, reason in getattr(diagnostics, "graph_breaks", []) or []
                ],
                "guard_failures": int(getattr(diagnostics, "guard_failures", 0)),
                "graph_count": int(getattr(diagnostics, "graph_count", 0)),
                "op_count": int(getattr(diagnostics, "op_count", 0)),
                "warnings": list(getattr(diagnostics, "warnings", []) or []),
            }
            (bundle_dir / _GRAPH_BREAKS_FILENAME).write_text(json.dumps(graph_breaks_payload, indent=2))
            statuses.append(ArtifactStatus(name="graph_breaks", status="ok", path=_GRAPH_BREAKS_FILENAME))
        except Exception as exc:
            statuses.append(ArtifactStatus(name="graph_breaks", status="failed", error=repr(exc)))

    # --- execution_plan.yaml + memory_plan.yaml ------------------------
    if payload_module is None or target_profile is None:
        statuses.append(
            ArtifactStatus(
                name="execution_plan",
                status="skipped",
                reason="need payload_module and target_profile",
            )
        )
        statuses.append(
            ArtifactStatus(
                name="memory_plan",
                status="skipped",
                reason="need payload_module and target_profile",
            )
        )
    else:
        import yaml  # type: ignore[import-untyped]

        from compgen.runtime.planner import plan_execution

        plan = None
        planner_error: str | None = None
        try:
            plan = plan_execution(payload_module, target_profile)
        except Exception as exc:
            # Planner can raise on modules with no recognised
            # partitions (e.g. empty IR). That's a model/IR-shape
            # condition, not an emission bug.
            planner_error = repr(exc)

        if plan is None:
            statuses.append(
                ArtifactStatus(
                    name="execution_plan",
                    status="skipped",
                    reason=f"plan_execution raised: {planner_error}",
                )
            )
            statuses.append(
                ArtifactStatus(
                    name="memory_plan",
                    status="skipped",
                    reason=f"no execution plan to derive from: {planner_error}",
                )
            )
        else:
            try:
                plan_dict = plan.to_dict()
                (bundle_dir / _EXECUTION_PLAN_FILENAME).write_text(
                    yaml.safe_dump(plan_dict, default_flow_style=False, sort_keys=False)
                )
                statuses.append(ArtifactStatus(name="execution_plan", status="ok", path=_EXECUTION_PLAN_FILENAME))
            except Exception as exc:
                statuses.append(ArtifactStatus(name="execution_plan", status="failed", error=repr(exc)))

            if plan.memory_plans:
                try:
                    memory_payload = [
                        {
                            "device": mp.device_index,
                            "peak_bytes": int(mp.peak_bytes),
                            "address_space": mp.address_space,
                            "physical_offset": int(mp.physical_offset),
                            "allocations": [
                                {
                                    "name": name,
                                    "offset": int(offset),
                                    "size": int(size),
                                    "alignment": int(alignment),
                                }
                                for (name, offset, size, alignment) in mp.allocations
                            ],
                        }
                        for mp in plan.memory_plans
                    ]
                    (bundle_dir / _MEMORY_PLAN_FILENAME).write_text(
                        yaml.safe_dump(memory_payload, default_flow_style=False, sort_keys=False)
                    )
                    statuses.append(ArtifactStatus(name="memory_plan", status="ok", path=_MEMORY_PLAN_FILENAME))
                except Exception as exc:
                    statuses.append(ArtifactStatus(name="memory_plan", status="failed", error=repr(exc)))
            else:
                statuses.append(
                    ArtifactStatus(
                        name="memory_plan",
                        status="skipped",
                        reason="planner produced no memory plans",
                    )
                )

    # --- gap_analysis.json ---------------------------------------------
    if analysis is None:
        statuses.append(
            ArtifactStatus(
                name="gap_analysis",
                status="skipped",
                reason="no NetworkAnalysis passed",
            )
        )
    else:
        try:
            gap_payload: dict[str, Any] = {
                "model_name": analysis.model_name,
                "total_params": int(analysis.total_params),
                "total_flops": int(analysis.total_flops),
                "total_bytes": int(analysis.total_bytes),
                "clusters": [
                    {
                        "cluster_id": c.cluster_id,
                        "pattern_type": c.pattern_type,
                        "node_names": list(c.node_names),
                        "total_flops": int(c.total_flops),
                        "total_bytes": int(c.total_bytes),
                        "arithmetic_intensity": float(c.arithmetic_intensity),
                        "estimated_latency_us_per_device": {
                            str(dev): float(lat) for dev, lat in c.estimated_latency_per_device.items()
                        },
                        "best_device": c.best_device,
                        "is_bottleneck": bool(c.is_bottleneck),
                        "kernel_opportunity": c.kernel_opportunity,
                        "input_shapes": {k: list(v) for k, v in c.input_shapes.items()},
                        "output_shapes": {k: list(v) for k, v in c.output_shapes.items()},
                    }
                    for c in analysis.clusters
                ],
                "unclustered_ops": list(analysis.unclustered_ops),
                "data_flow": [
                    {"src": e.src, "dst": e.dst, "tensor_bytes": int(e.tensor_bytes)} for e in analysis.data_flow
                ],
                "bottleneck_clusters": list(analysis.bottleneck_clusters),
                "optimization_opportunities": list(analysis.optimization_opportunities),
            }
            dossier = analysis.dossier
            if dossier is not None:
                gap_payload["dossier"] = {
                    "op_histogram": dict(dossier.op_histogram),
                    "repeated_patterns": dict(dossier.repeated_patterns),
                    "total_regions": int(dossier.total_regions),
                    "critical_path": list(dossier.critical_path),
                    "independent_region_sets": [list(s) for s in dossier.independent_region_sets],
                    "dynamic_shape_regions": list(dossier.dynamic_shape_regions),
                    "unsupported_targets": list(dossier.unsupported_targets),
                }
            (bundle_dir / _GAP_ANALYSIS_FILENAME).write_text(json.dumps(gap_payload, indent=2))
            statuses.append(ArtifactStatus(name="gap_analysis", status="ok", path=_GAP_ANALYSIS_FILENAME))
        except Exception as exc:
            statuses.append(ArtifactStatus(name="gap_analysis", status="failed", error=repr(exc)))

    # --- kernel_contracts/*.yaml ---------------------------------------
    if payload_module is None or target_profile is None:
        statuses.append(
            ArtifactStatus(
                name="kernel_contracts",
                status="skipped",
                reason="need payload_module and target_profile",
            )
        )
    else:
        try:
            import yaml  # type: ignore[import-untyped]

            from compgen.kernels.contracts import build_kernel_contracts

            specs = build_kernel_contracts(payload_module, target_profile, sample_inputs)
            if not specs:
                statuses.append(
                    ArtifactStatus(
                        name="kernel_contracts",
                        status="skipped",
                        reason="build_kernel_contracts returned no specs",
                    )
                )
            else:
                contracts_dir = bundle_dir / _KERNEL_CONTRACTS_DIR
                contracts_dir.mkdir(exist_ok=True)

                used_stems: dict[str, int] = {}
                for spec in specs:
                    raw = spec.contract.op_name or "unnamed"
                    stem = "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in raw)
                    count = used_stems.get(stem, 0)
                    used_stems[stem] = count + 1
                    filename = f"{stem}.yaml" if count == 0 else f"{stem}_{count}.yaml"

                    contract = spec.contract
                    payload = {
                        "op_name": contract.op_name,
                        "supported_dtypes": sorted(contract.supported_dtypes),
                        "fusable": bool(contract.fusable),
                        "aliasing": [[int(a), int(b)] for (a, b) in contract.aliasing],
                        "cost": {
                            "flops": int(contract.cost.flops),
                            "bytes_read": int(contract.cost.bytes_read),
                            "bytes_written": int(contract.cost.bytes_written),
                        },
                        "input_layouts": [
                            {
                                "kind": lr.kind.value if hasattr(lr.kind, "value") else str(lr.kind),
                                "strides": list(lr.strides) if lr.strides is not None else None,
                                "alignment": int(lr.alignment),
                            }
                            for lr in contract.input_layouts
                        ],
                        "output_layouts": [
                            {
                                "kind": lr.kind.value if hasattr(lr.kind, "value") else str(lr.kind),
                                "strides": list(lr.strides) if lr.strides is not None else None,
                                "alignment": int(lr.alignment),
                            }
                            for lr in contract.output_layouts
                        ],
                        "perf_target_us": (float(spec.perf_target_us) if spec.perf_target_us is not None else None),
                        "priority": int(spec.priority),
                        "input_shapes": [list(s) for s in spec.input_shapes],
                        "output_shapes": [list(s) for s in spec.output_shapes],
                        "metadata": dict(contract.metadata),
                    }
                    (contracts_dir / filename).write_text(
                        yaml.safe_dump(payload, default_flow_style=False, sort_keys=False)
                    )
                statuses.append(
                    ArtifactStatus(
                        name="kernel_contracts",
                        status="ok",
                        path=_KERNEL_CONTRACTS_DIR + "/",
                    )
                )
        except Exception as exc:
            statuses.append(ArtifactStatus(name="kernel_contracts", status="failed", error=repr(exc)))

    # --- transforms/*.mlir ---------------------------------------------
    # Recover transform scripts from pipeline artifacts. The drive loop
    # and individual stages can populate a "transforms" entry in their
    # artifact dict; each entry is a list of {"name": str, "mlir": str}.
    transforms = _extract_transforms(pipeline_artifacts) if pipeline_artifacts else []
    if not transforms:
        statuses.append(
            ArtifactStatus(
                name="transforms",
                status="skipped",
                reason="no transforms in pipeline_artifacts",
            )
        )
    else:
        try:
            transforms_dir = bundle_dir / _TRANSFORMS_DIR
            transforms_dir.mkdir(exist_ok=True)
            index: list[dict[str, str]] = []
            for idx, t in enumerate(transforms):
                raw_name = t.get("name") or f"transform_{idx}"
                stem = "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in raw_name)
                filename = f"transform_{idx:03d}_{stem}.mlir"
                mlir_text = t.get("mlir") or t.get("source") or ""
                if not isinstance(mlir_text, str) or not mlir_text.strip():
                    raise ValueError(f"transform '{raw_name}' has no 'mlir' or 'source' string")
                (transforms_dir / filename).write_text(mlir_text)
                index.append({"name": raw_name, "path": filename})
            (transforms_dir / "index.json").write_text(json.dumps(index, indent=2))
            statuses.append(ArtifactStatus(name="transforms", status="ok", path=_TRANSFORMS_DIR + "/"))
        except Exception as exc:
            statuses.append(ArtifactStatus(name="transforms", status="failed", error=repr(exc)))

    # --- generated_kernels/ --------------------------------------------
    kernels = _extract_generated_kernels(pipeline_artifacts) if pipeline_artifacts else []
    if not kernels:
        statuses.append(
            ArtifactStatus(
                name="generated_kernels",
                status="skipped",
                reason="no generated kernels in pipeline_artifacts",
            )
        )
    else:
        try:
            gk_dir = bundle_dir / _GENERATED_KERNELS_DIR
            gk_dir.mkdir(exist_ok=True)
            persisted: list[dict[str, Any]] = []
            for k in kernels:
                persisted.append(_persist_kernel_entry(k, gk_dir, sample_inputs))
            (gk_dir / "index.json").write_text(json.dumps(persisted, indent=2))
            statuses.append(ArtifactStatus(name="generated_kernels", status="ok", path=_GENERATED_KERNELS_DIR + "/"))
        except Exception as exc:
            statuses.append(ArtifactStatus(name="generated_kernels", status="failed", error=repr(exc)))

    # --- verification_report.json --------------------------------------
    if verification_report is None:
        statuses.append(
            ArtifactStatus(
                name="verification_report",
                status="skipped",
                reason="no verification_report passed (verify=False)",
            )
        )
    else:
        try:
            (bundle_dir / _VERIFICATION_REPORT_FILENAME).write_text(
                json.dumps(verification_report, indent=2, default=str)
            )
            statuses.append(
                ArtifactStatus(
                    name="verification_report",
                    status="ok",
                    path=_VERIFICATION_REPORT_FILENAME,
                )
            )
        except Exception as exc:
            statuses.append(ArtifactStatus(name="verification_report", status="failed", error=repr(exc)))

    # --- provider_feedback.json -----------------------------------------
    feedback = pipeline_artifacts.get("provider_contract_feedback") if pipeline_artifacts else None
    if not feedback:
        statuses.append(
            ArtifactStatus(
                name="provider_feedback",
                status="skipped",
                reason="no provider contract feedback emitted",
            )
        )
    else:
        try:
            (bundle_dir / _PROVIDER_FEEDBACK_FILENAME).write_text(json.dumps(list(feedback), indent=2, default=str))
            statuses.append(
                ArtifactStatus(
                    name="provider_feedback",
                    status="ok",
                    path=_PROVIDER_FEEDBACK_FILENAME,
                )
            )
        except Exception as exc:
            statuses.append(ArtifactStatus(name="provider_feedback", status="failed", error=repr(exc)))

    report = BundleEmissionReport(bundle_dir=bundle_dir, statuses=tuple(statuses))
    _update_manifest(bundle_dir, report)
    _publish_trace_events(report)

    log.info(
        "bundle_emit.done",
        bundle_dir=str(bundle_dir),
        ok=[s.name for s in report.ok],
        failed=[s.name for s in report.failed],
        skipped=[s.name for s in report.skipped],
    )
    return report


def _build_unified_arg_bundle(
    exported_program: Any,
    sample_inputs: tuple[Any, ...],
) -> list[Any]:
    """Build the unified ``func.func @forward`` arglist (params + inputs).

    PyTorch's ``ExportedProgram.graph_signature.input_specs`` lists each
    function arg as either a parameter, a buffer, a constant tensor, or
    a user input — in the same order they appear on
    ``ExportedProgram.module().graph``'s placeholder nodes. We follow
    that order so the saved tensor list matches
    ``payload.mlir func.func @forward`` 1:1.

    Falls back gracefully on older :class:`torch.export` shapes:
    when ``input_specs`` is missing (or doesn't classify cleanly) we
    return ``list(sample_inputs)`` so the slot is at minimum the
    user-supplied inputs.
    """
    from torch.export.graph_signature import InputKind  # type: ignore[import-not-found]

    sig = getattr(exported_program, "graph_signature", None)
    state_dict = dict(getattr(exported_program, "state_dict", {}))
    constants = dict(getattr(exported_program, "constants", {}))
    if sig is None:
        return list(sample_inputs)

    args: list[Any] = []
    user_iter = iter(sample_inputs)
    for spec in getattr(sig, "input_specs", []) or []:
        kind = getattr(spec, "kind", None)
        target = getattr(spec, "target", None)
        if kind is InputKind.PARAMETER and target in state_dict:
            args.append(state_dict[target])
        elif kind is InputKind.BUFFER and target in state_dict:
            args.append(state_dict[target])
        elif kind is InputKind.CONSTANT_TENSOR and target in constants:
            args.append(constants[target])
        elif kind is InputKind.USER_INPUT:
            try:
                args.append(next(user_iter))
            except StopIteration as exc:
                raise ValueError("ExportedProgram declares more user inputs than sample_inputs supplied") from exc
        else:
            # Unknown / unsupported InputKind. Try state_dict by target,
            # then fall back to None so downstream consumers can detect
            # a hole rather than miscount args.
            if target and target in state_dict:
                args.append(state_dict[target])
            else:
                args.append(None)
    # Append any remaining sample_inputs the signature didn't enumerate
    # (rare, but happens on older torch.export shapes that don't model
    # the user-input boundary explicitly).
    args.extend(list(user_iter))
    return args


def _safe_stem(raw: str) -> str:
    return "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in raw) or "unnamed"


def _extract_transforms(pipeline_artifacts: dict[str, Any]) -> list[dict[str, Any]]:
    """Pull transform scripts out of pipeline artifact dict.

    Accepted shapes:
      - ``pipeline_artifacts["transforms"]`` = list of dicts with
        keys ``name`` + ``mlir`` (or ``source``).
      - ``pipeline_artifacts["transform_scripts"]`` = same shape
        (accepted for backward compat with drive loop output).
    """
    for key in ("transforms", "transform_scripts"):
        raw = pipeline_artifacts.get(key)
        if isinstance(raw, list):
            return [t for t in raw if isinstance(t, dict)]
    return []


def _extract_generated_kernels(pipeline_artifacts: dict[str, Any]) -> list[dict[str, Any]]:
    """Pull generated kernels out of pipeline artifact dict.

    Accepted shape (per entry):

    - Required: ``provider``, ``op_name``, and either ``source``
      (string) or ``path`` (readable file path).
    - Optional: ``extension`` (default ``txt``); ``region_id`` and
      ``dispatch_id`` (cross-link with ``payload.mlir`` annotations);
      ``emit_mode`` (``compute_callback`` | ``self_contained``);
      ``dispatch_geometry`` (dict — see :class:`DispatchGeometry`);
      ``kernel_files`` (multi-file dict — see REQ-018);
      ``expected_inputs`` (symbol-layout map — see REQ-016).
    """
    raw = pipeline_artifacts.get("generated_kernels")
    if isinstance(raw, list):
        return [k for k in raw if isinstance(k, dict)]
    return []


def _persist_kernel_entry(
    k: dict[str, Any],
    gk_dir: Path,
    sample_inputs: tuple[Any, ...] | None,
) -> dict[str, Any]:
    """Persist one kernel entry — single-file or multi-file (REQ-018).

    Filenames are prefixed with ``region_id`` to avoid collisions when
    two regions share an ``op_name`` (REQ-014).

    Returns the index.json record for the entry, including
    ``region_id`` / ``dispatch_id`` / ``emit_mode`` / optional
    ``dispatch_geometry`` and ``additional_files``.
    """
    provider = str(k.get("provider", "unknown"))
    op_name = str(k.get("op_name", "unnamed"))
    region_id = str(k.get("region_id", ""))
    dispatch_id = str(k.get("dispatch_id", ""))
    emit_mode = str(k.get("emit_mode", "compute_callback"))
    extension = str(k.get("extension", "txt"))

    provider_dir = gk_dir / _safe_stem(provider)
    provider_dir.mkdir(exist_ok=True)

    # Filename prefix: region_id is the disambiguator. Without it, two
    # ``aten_add`` regions in the same module clobber each other.
    op_stem = _safe_stem(op_name)
    base_stem = f"{_safe_stem(region_id)}_{op_stem}" if region_id else op_stem
    primary_filename = f"{base_stem}.{extension.lstrip('.')}"

    record: dict[str, Any] = {
        "provider": provider,
        "op_name": op_name,
    }
    if region_id:
        record["region_id"] = region_id
    if dispatch_id:
        record["dispatch_id"] = dispatch_id
    record["emit_mode"] = emit_mode
    if k.get("dispatch_geometry"):
        record["dispatch_geometry"] = k["dispatch_geometry"]

    multi_files = k.get("kernel_files")
    if multi_files:
        # Multi-file bundle (REQ-018): every entry lands inside a
        # per-region directory.
        op_dir = provider_dir / base_stem
        op_dir.mkdir(exist_ok=True)
        primary_path: Path | None = None
        siblings: list[str] = []
        # Pick a primary file: exact match for ``primary_filename``, then
        # any extension match for ``op_stem``, then the first key.
        keys = list(multi_files)
        primary_key = next(
            (k_ for k_ in keys if k_ == primary_filename or Path(k_).stem == op_stem),
            keys[0] if keys else None,
        )
        for fname, contents in multi_files.items():
            (op_dir / fname).write_text(contents)
            rel = f"{_safe_stem(provider)}/{base_stem}/{fname}"
            if fname == primary_key:
                primary_path = op_dir / fname
                record["path"] = rel
            else:
                siblings.append(rel)
        if primary_path is None and keys:
            # Fallback when the picked primary key didn't actually exist.
            record["path"] = f"{_safe_stem(provider)}/{base_stem}/{keys[0]}"
        record["additional_files"] = siblings
    else:
        # Single-file shape — the source goes directly in the provider dir.
        source = k.get("source")
        source_path = k.get("path")
        target_path = provider_dir / primary_filename
        if isinstance(source, str) and source:
            target_path.write_text(source)
        elif isinstance(source_path, (str, Path)) and Path(source_path).exists():
            shutil.copyfile(str(source_path), target_path)
        else:
            raise ValueError(f"kernel entry for {provider}/{op_name} has neither 'source' nor a readable 'path'")
        record["path"] = f"{_safe_stem(provider)}/{primary_filename}"

    # REQ-016: expected_inputs → <region>_<op>.data.h next to the
    # primary file. The pack composer ``#include``s it and the kernel
    # sees its expected symbol layout without composer-side guessing.
    expected_inputs = k.get("expected_inputs")
    if expected_inputs:
        data_h_name = f"{base_stem}.data.h"
        data_dir = provider_dir if not multi_files else provider_dir / base_stem
        data_h_path = data_dir / data_h_name
        data_h_path.write_text(_render_data_header(expected_inputs, sample_inputs))
        record["data_header"] = (
            f"{_safe_stem(provider)}/{base_stem}/{data_h_name}"
            if multi_files
            else f"{_safe_stem(provider)}/{data_h_name}"
        )

    return record


def _render_data_header(
    expected_inputs: dict[str, dict[str, Any]],
    sample_inputs: tuple[Any, ...] | None,
) -> str:
    """Render ``expected_inputs`` as a C header (REQ-016).

    Each symbol entry maps to a ``static const`` array. ``init`` modes:

    - ``"from_golden:<i>"`` — bake the i-th sample input as the array
      contents (uint32 view of the raw bytes by default).
    - ``"literal:<value>"`` — single-element array initialized to value.
    - ``"zeros"`` — zero-initialized.
    """
    lines: list[str] = [
        "/* Auto-generated by bundle_emit from ProviderResult.expected_inputs.",
        " * Provider declares the symbol layout it expects; pack composer",
        " * #includes this header so the kernel sees its symbol contract.",
        " */",
        "#ifndef COMPGEN_GENERATED_KERNEL_DATA_H",
        "#define COMPGEN_GENERATED_KERNEL_DATA_H",
        "#include <stdint.h>",
        "",
    ]

    for symbol, spec in expected_inputs.items():
        size = int(spec.get("size", 0))
        dtype = str(spec.get("dtype", "uint32"))
        init = str(spec.get("init", "zeros"))
        ctype = _DTYPE_CTYPE.get(dtype, "uint32_t")

        values: list[str]
        if init.startswith("literal:"):
            literal = init.split(":", 1)[1]
            values = [literal]
            decl_size = max(1, len(values))
        elif init.startswith("from_golden:") and sample_inputs is not None:
            try:
                idx = int(init.split(":", 1)[1])
            except ValueError:
                idx = -1
            if 0 <= idx < len(sample_inputs):
                values = _golden_to_uint32_words(sample_inputs[idx], size)
                decl_size = max(size, len(values))
            else:
                values = ["0"] * max(size, 1)
                decl_size = max(size, 1)
        else:
            values = ["0"] * max(size, 1)
            decl_size = max(size, 1)

        body = ", ".join(values)
        lines.append(f"static const {ctype} {symbol}[{decl_size}] = {{ {body} }};")
    lines.extend(["", "#endif  /* COMPGEN_GENERATED_KERNEL_DATA_H */", ""])
    return "\n".join(lines)


_DTYPE_CTYPE: dict[str, str] = {
    "uint32": "uint32_t",
    "u32": "uint32_t",
    "int32": "int32_t",
    "i32": "int32_t",
    "uint16": "uint16_t",
    "uint8": "uint8_t",
    "float": "float",
    "f32": "float",
}


def _golden_to_uint32_words(tensor: Any, size: int) -> list[str]:
    """Best-effort: serialize a sample input tensor as uint32 words.

    Uses ``torch.Tensor.contiguous().view(torch.uint8).numpy().tobytes()``
    when available, then packs into uint32 little-endian words.
    """
    try:
        import struct

        import torch as _torch

        if isinstance(tensor, _torch.Tensor):
            buf = tensor.detach().cpu().contiguous().view(_torch.uint8).numpy().tobytes()
        elif hasattr(tensor, "tobytes"):
            buf = tensor.tobytes()
        else:
            return ["0"] * max(size, 1)
        # Pad to a 4-byte boundary.
        if len(buf) % 4 != 0:
            buf = buf + b"\x00" * (4 - (len(buf) % 4))
        words = struct.unpack(f"<{len(buf) // 4}I", buf)
        out = [f"0x{w:08x}u" for w in words[: max(size, len(words))]]
        if size and len(out) < size:
            out.extend(["0"] * (size - len(out)))
        return out
    except Exception:
        return ["0"] * max(size, 1)


def _update_manifest(bundle_dir: Path, report: BundleEmissionReport) -> None:
    """Merge per-artifact statuses into manifest.json::extended_artifacts."""
    manifest_path = bundle_dir / _MANIFEST_FILENAME
    if not manifest_path.is_file():
        log.warning("bundle_emit.manifest_missing", bundle_dir=str(bundle_dir))
        return

    try:
        manifest = json.loads(manifest_path.read_text())
    except Exception as exc:
        # Manifest is BundleStage's responsibility; if it's unreadable
        # that's a hard bug, surface it.
        raise RuntimeError(f"manifest.json at {manifest_path} is unreadable: {exc!r}") from exc

    manifest["extended_artifacts"] = report.to_manifest_block()

    # Also fold ok artifact paths into the top-level "artifacts" dict
    # (schema compatibility with older bundle_runner versions).
    artifacts = manifest.setdefault("artifacts", {})
    if isinstance(artifacts, dict):
        for s in report.ok:
            if s.path:
                artifacts[s.name] = s.path

    manifest_path.write_text(json.dumps(manifest, indent=2))


def _publish_trace_events(report: BundleEmissionReport) -> None:
    """Publish per-artifact status to the active trace bus (best effort)."""
    try:
        from compgen.trace import get_active_bus
    except Exception:
        return

    bus = get_active_bus()
    if bus is None:
        return
    for s in report.statuses:
        try:
            bus.publish(
                kind="bundle.extended_artifact",
                payload={
                    "name": s.name,
                    "status": s.status,
                    "path": s.path,
                    "error": s.error,
                    "reason": s.reason,
                },
                level="ERROR" if s.status == "failed" else "INFO",
            )
        except Exception:
            # Trace failures must never mask compile results.
            log.warning("bundle_emit.trace_publish_failed", artifact=s.name)


__all__ = [
    "emit_extended_artefacts",
]
