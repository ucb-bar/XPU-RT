"""FP8 E4M3 quantization for NPU deployment via torchAO.

This package provides:

- ``FP8E4M3Po2Config``: torchAO-compatible config for ``quantize_()``.
- ``FP8E4M3Po2Tensor``: Custom tensor subclass with FP8 dispatch.
- ``ExportableFP8Attention``: Unfused attention with BF16 softmax.
- ``SmolVLAQuantRecipe``: Per-component quantization recipe for smolVLA.
- ``NpuOpCategory``: NPU operator classification.
- ``rewrite_for_export()``: Pre-export module rewriting.
- ``npu_alignment_check()``: NPU hardware constraint verification.
- ``QuantizedModelPipeline``: Generalizable quantize→capture→analyze→lower pipeline.
- ``analyze_for_npu()``: FX graph analysis for NPU op coverage.
"""

from compgen.quantization.attention import ExportableFP8Attention, FP8AttentionConfig
from compgen.quantization.export_wrappers import (
    ExportableFP8Conv2d,
    ExportableFP8Linear,
    rewrite_for_export,
)
from compgen.quantization.fp8_config import FP8E4M3Po2Config
from compgen.quantization.fp8_ops import (
    FP8_E4M3_DTYPE,
    FP8_E4M3_MAX,
    FP8_E4M3_MAX_PO2,
    dequantize_fp8_e4m3,
    quantize_dequantize_fp8_po2,
    quantize_fp8_e4m3_po2,
)
from compgen.quantization.fp8_tensor import FP8E4M3Po2Tensor
from compgen.quantization.npu_op_map import NpuOpCategory, NpuQuantDecision, classify_op
from compgen.quantization.smolvla_recipe import (
    SmolVLAComponent,
    SmolVLAQuantRecipe,
    apply_smolvla_quantization,
    default_npu_recipe,
    infer_component,
)
from compgen.quantization.graph_analyzer import QuantizedGraphAnalysis, analyze_for_npu, format_analysis_report
from compgen.quantization.pipeline import PipelineReport, QuantizedModelPipeline
from compgen.quantization.verify import NpuAlignmentResult, npu_alignment_check

__all__ = [
    "ExportableFP8Attention",
    "ExportableFP8Conv2d",
    "ExportableFP8Linear",
    "FP8AttentionConfig",
    "FP8E4M3Po2Config",
    "FP8E4M3Po2Tensor",
    "FP8_E4M3_DTYPE",
    "FP8_E4M3_MAX",
    "FP8_E4M3_MAX_PO2",
    "NpuAlignmentResult",
    "NpuOpCategory",
    "NpuQuantDecision",
    "SmolVLAComponent",
    "SmolVLAQuantRecipe",
    "apply_smolvla_quantization",
    "classify_op",
    "default_npu_recipe",
    "dequantize_fp8_e4m3",
    "infer_component",
    "npu_alignment_check",
    "quantize_dequantize_fp8_po2",
    "quantize_fp8_e4m3_po2",
    "rewrite_for_export",
    "QuantizedGraphAnalysis",
    "QuantizedModelPipeline",
    "PipelineReport",
    "analyze_for_npu",
    "format_analysis_report",
]
