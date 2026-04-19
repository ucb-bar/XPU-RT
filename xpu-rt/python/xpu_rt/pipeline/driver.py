"""End-to-end pipeline driver that runs all 22 Wave 1-6 passes.

Modelled on hexagon-mlir's
``qcom_hexagon_backend/lib/Conversion/LinalgToLLVM/LinalgToLLVMPass.cpp``
(conditional pass chains + option-driven orchestration).

Entry point: :func:`compile_through_pipeline`.

    result = compile_through_pipeline(
        exported_program,
        options=cuda_a100_defaults(),
    )
    # result.module : xDSL ModuleOp after all IR passes.
    # result.execution_plan : Phase 5 ExecutionPlan.
    # result.stage_reports : per-pass stats for observability.

The driver enforces the wave order:

    Wave 0:  FX → xDSL via ``bridge_fx_graph``
    Wave 1:  decompose_concat, fold_transposes_into_dots,
             demote_contraction_inputs, set_numerics_policy
    Wave 2:  raise_special_ops, fuse_softmax_to_triton
    Wave 3:  propagate_transposes, plan_reduction
    Wave 4:  lower_quantized_matmul, lower_quantized_conv,
             fuse_dequant_matmul, normalize_subbyte
    Wave 5:  lower_conv_to_img2col, match_library_call
    Wave 6:  assign_memory_space → assign_queue → assign_streams
             → plan_buffers → insert_copies → alias_io_buffers
             → dma_overlap → insert_host_offload
             → normalize_subbyte_post_layout

Each pass reads its enable-flag from the passed ``CompGenOptions``;
when a flag is ``False`` the pass is skipped and the stage report
records ``skipped=True``.

The driver builds a minimal ``ExecutionPlan`` from the xDSL module
before Phase 5 (one region per ``func.func``, one buffer per
named SSA value that crosses a region boundary). Full structural
lifting from Recipe IR lands in Wave 8+; this minimal builder is
enough to exercise the Phase 5 passes end-to-end.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import structlog
from xdsl.dialects.builtin import ModuleOp

from compgen.capture.torch_mlir_bridge import bridge_fx_graph
from compgen.options import CompGenOptions
from compgen.runtime.execution_plan import ExecutionPlan
from compgen.runtime.plan_builder import ExecutionPlanBuilder

log = structlog.get_logger()


# --- per-pass imports (lazy for clean error messages) ------------------------


def _import_passes() -> dict[str, Any]:
    """Lazy-import every pass so stack traces on import errors point
    at the specific missing file instead of a giant top-level import."""
    from compgen.ir.payload.passes.rewrites.alias_io_buffers import (
        run_alias_io_buffers,
    )
    from compgen.ir.payload.passes.rewrites.assign_memory_space import (
        AssignMemorySpaceConfig,
        run_assign_memory_space,
    )
    from compgen.ir.payload.passes.rewrites.assign_queue import (
        run_assign_queue,
    )
    from compgen.ir.payload.passes.rewrites.assign_streams import (
        run_assign_streams,
    )
    from compgen.ir.payload.passes.rewrites.decompose_concat import (
        run_decompose_concat,
    )
    from compgen.ir.payload.passes.rewrites.demote_contraction_inputs import (
        DemoteContractionInputsConfig,
        run_demote_contraction_inputs,
    )
    from compgen.ir.payload.passes.rewrites.dma_overlap import (
        DMAOverlapConfig,
        run_dma_overlap,
    )
    from compgen.ir.payload.passes.rewrites.fold_transposes_into_dots import (
        run_fold_transposes_into_dots,
    )
    from compgen.ir.payload.passes.rewrites.fuse_dequant_matmul import (
        FuseDequantMatmulConfig,
        run_fuse_dequant_matmul,
    )
    from compgen.ir.payload.passes.rewrites.fuse_softmax_to_triton import (
        FuseSoftmaxToTritonConfig,
        run_fuse_softmax_to_triton,
    )
    from compgen.ir.payload.passes.rewrites.insert_copies import (
        run_insert_copies,
    )
    from compgen.ir.payload.passes.rewrites.insert_host_offload import (
        run_insert_host_offload,
    )
    from compgen.ir.payload.passes.rewrites.lower_conv_to_img2col import (
        run_lower_conv_to_img2col,
    )
    from compgen.ir.payload.passes.rewrites.lower_quantized_conv import (
        run_lower_quantized_conv,
    )
    from compgen.ir.payload.passes.rewrites.lower_quantized_matmul import (
        LowerQuantizedMatmulConfig,
        run_lower_quantized_matmul,
    )
    from compgen.ir.payload.passes.rewrites.match_library_call import (
        MatchLibraryCallConfig,
        run_match_library_call,
    )
    from compgen.ir.payload.passes.rewrites.normalize_subbyte import (
        run_normalize_subbyte,
    )
    from compgen.ir.payload.passes.rewrites.normalize_subbyte_post_layout import (
        run_normalize_subbyte_post_layout,
    )
    from compgen.ir.payload.passes.rewrites.plan_buffers import (
        run_plan_buffers,
    )
    from compgen.ir.payload.passes.rewrites.plan_reduction import (
        PlanReductionConfig,
        run_plan_reduction,
    )
    from compgen.ir.payload.passes.rewrites.propagate_transposes import (
        PropagateTransposesConfig,
        run_propagate_transposes,
    )
    from compgen.ir.payload.passes.rewrites.raise_special_ops import (
        run_raise_special_ops,
    )
    from compgen.ir.payload.passes.rewrites.set_numerics_policy import (
        NumericsPolicy,
        run_set_numerics_policy,
    )
    return {
        "run_alias_io_buffers": run_alias_io_buffers,
        "AssignMemorySpaceConfig": AssignMemorySpaceConfig,
        "run_assign_memory_space": run_assign_memory_space,
        "run_assign_queue": run_assign_queue,
        "run_assign_streams": run_assign_streams,
        "run_decompose_concat": run_decompose_concat,
        "DemoteContractionInputsConfig": DemoteContractionInputsConfig,
        "run_demote_contraction_inputs": run_demote_contraction_inputs,
        "DMAOverlapConfig": DMAOverlapConfig,
        "run_dma_overlap": run_dma_overlap,
        "run_fold_transposes_into_dots": run_fold_transposes_into_dots,
        "FuseDequantMatmulConfig": FuseDequantMatmulConfig,
        "run_fuse_dequant_matmul": run_fuse_dequant_matmul,
        "FuseSoftmaxToTritonConfig": FuseSoftmaxToTritonConfig,
        "run_fuse_softmax_to_triton": run_fuse_softmax_to_triton,
        "run_insert_copies": run_insert_copies,
        "run_insert_host_offload": run_insert_host_offload,
        "run_lower_conv_to_img2col": run_lower_conv_to_img2col,
        "run_lower_quantized_conv": run_lower_quantized_conv,
        "LowerQuantizedMatmulConfig": LowerQuantizedMatmulConfig,
        "run_lower_quantized_matmul": run_lower_quantized_matmul,
        "MatchLibraryCallConfig": MatchLibraryCallConfig,
        "run_match_library_call": run_match_library_call,
        "run_normalize_subbyte": run_normalize_subbyte,
        "run_normalize_subbyte_post_layout": run_normalize_subbyte_post_layout,
        "run_plan_buffers": run_plan_buffers,
        "PlanReductionConfig": PlanReductionConfig,
        "run_plan_reduction": run_plan_reduction,
        "PropagateTransposesConfig": PropagateTransposesConfig,
        "run_propagate_transposes": run_propagate_transposes,
        "run_raise_special_ops": run_raise_special_ops,
        "NumericsPolicy": NumericsPolicy,
        "run_set_numerics_policy": run_set_numerics_policy,
    }


# --- stage reports ----------------------------------------------------------


@dataclass
class PipelineStageReport:
    name: str
    wave: int
    skipped: bool = False
    skipped_reason: str = ""
    stats: Any = None


@dataclass
class PipelineResult:
    """Full pipeline output.

    Attributes:
        module: final xDSL ModuleOp after all IR passes.
        execution_plan: final ExecutionPlan after all Phase 5 passes.
        bridge_path: ``"torch_mlir"`` / ``"fx_importer"`` / ``"failed"``.
        stage_reports: per-pass stats.
        options: the options used for this compile.
    """

    module: ModuleOp | None
    execution_plan: ExecutionPlan
    bridge_path: str
    stage_reports: list[PipelineStageReport] = field(default_factory=list)
    options: CompGenOptions | None = None

    @property
    def stages_run(self) -> int:
        return sum(1 for r in self.stage_reports if not r.skipped)

    @property
    def stages_skipped(self) -> int:
        return sum(1 for r in self.stage_reports if r.skipped)


# --- helper: build minimal ExecutionPlan from xDSL module ------------------


def _build_minimal_execution_plan(
    module: ModuleOp,
    workload: str,
    target: str,
) -> ExecutionPlan:
    """Build a first-draft ExecutionPlan from the xDSL module.

    One region per ``func.func``, one buffer per op that produces a
    tensor result of non-trivial size. Everything starts on the
    ``cuda:0`` device with an empty queue (filled in by later
    passes).
    """
    from xdsl.dialects.builtin import TensorType

    b = ExecutionPlanBuilder(workload, target)
    tick = 0
    for op in module.walk():
        if op.name == "func.func":
            region_id = op.sym_name.data if hasattr(op, "sym_name") else f"r_{tick}"
            b.add_region(region_id, "cuda:0", "")
            tick += 1
    # One buffer per named SSA value with a static tensor type.
    buf_tick = 0
    seen_buffers: set[str] = set()
    for op in module.walk():
        for i, res in enumerate(op.results):
            t = res.type
            if not isinstance(t, TensorType):
                continue
            shape = list(t.get_shape())
            if any(d < 0 for d in shape):
                continue
            elem_bits = getattr(t.get_element_type(), "bitwidth", 32)
            elem_bytes = max(1, (elem_bits + 7) // 8)
            size_bytes = elem_bytes
            for d in shape:
                size_bytes *= d
            bid = f"{op.name.replace('.', '_')}_r{i}_t{buf_tick}"
            if bid in seen_buffers:
                continue
            seen_buffers.add(bid)
            b.add_buffer(
                buffer_id=bid,
                size_bytes=size_bytes,
                memory_space="",
                first_use_tick=buf_tick,
                last_use_tick=buf_tick + 1,
            )
            buf_tick += 1
    return b.build()


# --- driver ----------------------------------------------------------------


def _run_with_report(
    passes: dict[str, Any],
    fn_key: str,
    wave: int,
    enabled: bool,
    name: str,
    *,
    args: tuple = (),
    kwargs: dict | None = None,
) -> PipelineStageReport:
    if kwargs is None:
        kwargs = {}
    if not enabled:
        return PipelineStageReport(name=name, wave=wave, skipped=True, skipped_reason="disabled")
    try:
        stats = passes[fn_key](*args, **kwargs)
        return PipelineStageReport(name=name, wave=wave, stats=stats)
    except Exception as exc:  # noqa: BLE001
        log.warning(f"pipeline.{name}.failed", error=str(exc))
        return PipelineStageReport(
            name=name, wave=wave, skipped=True, skipped_reason=f"error: {exc}"
        )


def compile_through_pipeline(
    model_or_exported: Any,
    example_inputs: tuple[Any, ...] | None = None,
    *,
    options: CompGenOptions | None = None,
    workload_name: str = "unnamed",
    target_name: str = "cuda_a100",
) -> PipelineResult:
    """Compile a model through all 22 passes.

    ``model_or_exported`` can be:
    - an ``nn.Module`` (in which case ``example_inputs`` is required), or
    - an ``ExportedProgram`` from ``torch.export``.

    ``options`` controls which passes run. When ``None``, a
    conservative all-off ``CompGenOptions()`` is used (only the
    bridge runs, everything else is skipped).
    """
    if options is None:
        options = CompGenOptions()

    passes = _import_passes()

    # --- Wave 0: bridge ----------------------------------------------------
    if hasattr(model_or_exported, "graph") and example_inputs is None:
        # Already an ExportedProgram; its .graph_module is the bridge input.
        from compgen.ir.payload.import_fx import FXImporter
        importer = FXImporter()
        module = importer.import_graph(model_or_exported)
        bridge_path = "fx_importer"
    else:
        if example_inputs is None:
            example_inputs = ()
        bridge = bridge_fx_graph(model_or_exported, example_inputs)
        module = bridge.module
        bridge_path = bridge.path_taken
        if module is None:
            return PipelineResult(
                module=None,
                execution_plan=ExecutionPlan(workload=workload_name, target=target_name),
                bridge_path=bridge_path,
                stage_reports=[
                    PipelineStageReport(
                        name="bridge_fx_graph", wave=0,
                        skipped=True, skipped_reason="bridge failed",
                    )
                ],
                options=options,
            )

    reports: list[PipelineStageReport] = [
        PipelineStageReport(name="bridge_fx_graph", wave=0),
    ]

    # --- Wave 1: structural / numerics ------------------------------------
    reports.append(_run_with_report(
        passes, "run_decompose_concat", 1,
        options.enable_decompose_concat, "decompose_concat",
        args=(module,),
    ))
    reports.append(_run_with_report(
        passes, "run_fold_transposes_into_dots", 1,
        options.enable_fold_transposes_into_dots, "fold_transposes_into_dots",
        args=(module,),
    ))
    if options.enable_demote_contraction_inputs:
        from xdsl.dialects.builtin import BFloat16Type, Float16Type
        target_t = BFloat16Type() if options.demote_target_type == "bf16" else Float16Type()
        cfg = passes["DemoteContractionInputsConfig"](target_type=target_t)
        reports.append(_run_with_report(
            passes, "run_demote_contraction_inputs", 1, True,
            "demote_contraction_inputs",
            args=(module,),
            kwargs={"config": cfg},
        ))
    else:
        reports.append(PipelineStageReport(
            name="demote_contraction_inputs", wave=1, skipped=True,
            skipped_reason="disabled",
        ))
    reports.append(_run_with_report(
        passes, "run_set_numerics_policy", 1,
        options.enable_set_numerics_policy, "set_numerics_policy",
        args=(module,),
    ))

    # --- Wave 2: semantic detection ---------------------------------------
    reports.append(_run_with_report(
        passes, "run_raise_special_ops", 2,
        options.enable_raise_special_ops, "raise_special_ops",
        args=(module,),
    ))
    if options.enable_fuse_softmax_to_triton:
        cfg = passes["FuseSoftmaxToTritonConfig"](
            kernel_family_allowlist=options.kernel_family_allowlist,
        )
        reports.append(_run_with_report(
            passes, "run_fuse_softmax_to_triton", 2, True,
            "fuse_softmax_to_triton",
            args=(module,),
            kwargs={"config": cfg},
        ))
    else:
        reports.append(PipelineStageReport(
            name="fuse_softmax_to_triton", wave=2, skipped=True,
            skipped_reason="disabled",
        ))

    # --- Wave 3: layout / reduction ---------------------------------------
    if options.enable_propagate_transposes:
        cfg = passes["PropagateTransposesConfig"](
            aggressiveness=options.transpose_aggressiveness,
        )
        reports.append(_run_with_report(
            passes, "run_propagate_transposes", 3, True,
            "propagate_transposes",
            args=(module,),
            kwargs={"config": cfg},
        ))
    else:
        reports.append(PipelineStageReport(
            name="propagate_transposes", wave=3, skipped=True,
            skipped_reason="disabled",
        ))
    if options.enable_plan_reduction:
        cfg = passes["PlanReductionConfig"](policy=options.reduction_policy)
        reports.append(_run_with_report(
            passes, "run_plan_reduction", 3, True, "plan_reduction",
            args=(module,), kwargs={"config": cfg},
        ))
    else:
        reports.append(PipelineStageReport(
            name="plan_reduction", wave=3, skipped=True, skipped_reason="disabled",
        ))

    # --- Wave 4: quantization ---------------------------------------------
    if options.enable_lower_quantized_matmul:
        cfg = passes["LowerQuantizedMatmulConfig"](policy=options.quantized_matmul_policy)
        reports.append(_run_with_report(
            passes, "run_lower_quantized_matmul", 4, True,
            "lower_quantized_matmul",
            args=(module,), kwargs={"config": cfg},
        ))
    else:
        reports.append(PipelineStageReport(
            name="lower_quantized_matmul", wave=4, skipped=True, skipped_reason="disabled",
        ))
    reports.append(_run_with_report(
        passes, "run_lower_quantized_conv", 4,
        options.enable_lower_quantized_conv, "lower_quantized_conv",
        args=(module,),
    ))
    if options.enable_fuse_dequant_matmul:
        cfg = passes["FuseDequantMatmulConfig"](
            reassoc_safe_only=options.fuse_dequant_reassoc_safe,
        )
        reports.append(_run_with_report(
            passes, "run_fuse_dequant_matmul", 4, True,
            "fuse_dequant_matmul",
            args=(module,), kwargs={"config": cfg},
        ))
    else:
        reports.append(PipelineStageReport(
            name="fuse_dequant_matmul", wave=4, skipped=True, skipped_reason="disabled",
        ))
    reports.append(_run_with_report(
        passes, "run_normalize_subbyte", 4,
        options.enable_normalize_subbyte, "normalize_subbyte",
        args=(module,),
    ))

    # --- Wave 5: large structural -----------------------------------------
    reports.append(_run_with_report(
        passes, "run_lower_conv_to_img2col", 5,
        options.enable_lower_conv_to_img2col, "lower_conv_to_img2col",
        args=(module,),
    ))
    if options.enable_match_library_call:
        cfg = passes["MatchLibraryCallConfig"](
            library_allowlist=tuple(options.library_allowlist),
        )
        reports.append(_run_with_report(
            passes, "run_match_library_call", 5, True,
            "match_library_call",
            args=(module,), kwargs={"config": cfg},
        ))
    else:
        reports.append(PipelineStageReport(
            name="match_library_call", wave=5, skipped=True, skipped_reason="disabled",
        ))

    # --- Wave 6: Phase 5 runtime ------------------------------------------
    plan = _build_minimal_execution_plan(module, workload_name, target_name)

    if options.enable_assign_memory_space:
        cfg = passes["AssignMemorySpaceConfig"](
            vtcm_bytes=options.vtcm_bytes,
            scratch_memory_space="vtcm" if options.vtcm_bytes > 0 else "scratchpad",
        )
        reports.append(_run_with_report(
            passes, "run_assign_memory_space", 6, True,
            "assign_memory_space",
            args=(plan,), kwargs={"config": cfg},
        ))
    else:
        reports.append(PipelineStageReport(
            name="assign_memory_space", wave=6, skipped=True, skipped_reason="disabled",
        ))

    reports.append(_run_with_report(
        passes, "run_assign_queue", 6, options.enable_assign_queue,
        "assign_queue", args=(plan,),
    ))
    reports.append(_run_with_report(
        passes, "run_assign_streams", 6, options.enable_assign_streams,
        "assign_streams", args=(plan,),
    ))
    reports.append(_run_with_report(
        passes, "run_plan_buffers", 6, options.enable_plan_buffers,
        "plan_buffers", args=(plan,),
    ))
    reports.append(_run_with_report(
        passes, "run_insert_copies", 6, options.enable_insert_copies,
        "insert_copies", args=(plan,),
    ))
    reports.append(_run_with_report(
        passes, "run_alias_io_buffers", 6, options.enable_alias_io_buffers,
        "alias_io_buffers", args=(plan,),
    ))
    if options.enable_dma_overlap:
        cfg = passes["DMAOverlapConfig"]()
        reports.append(_run_with_report(
            passes, "run_dma_overlap", 6, True, "dma_overlap",
            args=(plan,), kwargs={"config": cfg},
        ))
    else:
        reports.append(PipelineStageReport(
            name="dma_overlap", wave=6, skipped=True, skipped_reason="disabled",
        ))
    reports.append(_run_with_report(
        passes, "run_insert_host_offload", 6, options.enable_insert_host_offload,
        "insert_host_offload", args=(plan,),
    ))
    reports.append(_run_with_report(
        passes, "run_normalize_subbyte_post_layout", 6,
        options.enable_normalize_subbyte_post_layout,
        "normalize_subbyte_post_layout", args=(plan,),
    ))

    # Verify plan post-conditions when any Wave 6 pass ran.
    try:
        plan.validate()
    except Exception as exc:  # noqa: BLE001
        reports.append(PipelineStageReport(
            name="_plan_validate", wave=6,
            skipped=True, skipped_reason=f"validate failed: {exc}",
        ))

    return PipelineResult(
        module=module,
        execution_plan=plan,
        bridge_path=bridge_path,
        stage_reports=reports,
        options=options,
    )


__all__ = [
    "PipelineResult",
    "PipelineStageReport",
    "compile_through_pipeline",
]
