"""Generalizable quantized model pipeline.

Chains: load → quantize → verify → capture → analyze → lower → bundle.
Works with any ``nn.Module``, any quantization recipe, and any target op map.

Produces the standard CompGen artifact contract output::

    <output_dir>/
        golden_inputs.pt
        golden_outputs.pt
        graph_analysis.json
        alignment_report.json
        payload.mlir                  # per partition or combined
        kernel_contracts/*.yaml
        verification_report.json
        manifest.json

Usage::

    from compgen.quantization.pipeline import QuantizedModelPipeline

    pipeline = QuantizedModelPipeline(
        model=my_model,
        sample_inputs=(x,),
        model_name="my_model_fp8",
        quant_config=QuantizationConfig(scheme="fp8_e4m3_po2_npu"),
        output_dir="artifacts/my_model_fp8",
    )
    report = pipeline.run()
"""

from __future__ import annotations

import io
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog
import torch
import torch.nn as nn
from xdsl.dialects.builtin import ModuleOp
from xdsl.printer import Printer

from compgen.capture.torch_export import CaptureArtifact, capture_dynamo_partitions
from compgen.capture.torchao_pipeline import QuantizationConfig, apply_quantization
from compgen.ir.payload.contracts import KernelContract, extract_contracts
from compgen.ir.payload.import_fx import ImportDiagnostic, fx_to_xdsl
from compgen.quantization.export_wrappers import rewrite_for_export
from compgen.quantization.graph_analyzer import (
    QuantizedGraphAnalysis,
    analyze_for_npu,
    analyze_fx_graphs,
    format_analysis_report,
)
from compgen.quantization.verify import NpuAlignmentResult, npu_alignment_check

logger = structlog.get_logger()


@dataclass
class PipelineReport:
    """Comprehensive report from the quantized model pipeline.

    Attributes:
        model_name: Identifier for this pipeline run.
        param_count: Total model parameter count.
        quantization_applied: Whether quantization was successfully applied.
        alignment_result: NPU alignment verification result.
        graph_analysis: FX graph op coverage analysis.
        capture_artifact: The CaptureArtifact from dynamo partitioning.
        payload_ir_text: Combined Payload IR as MLIR text.
        import_diagnostics: Diagnostics from FX→xDSL import.
        kernel_contracts: Extracted kernel contracts.
        artifact_dir: Path to saved artifacts.
        errors: List of non-fatal errors encountered.
        timings: Timing for each pipeline step.
    """

    model_name: str = "model"
    param_count: int = 0
    quantization_applied: bool = False
    alignment_result: NpuAlignmentResult | None = None
    graph_analysis: QuantizedGraphAnalysis | None = None
    capture_artifact: CaptureArtifact | None = None
    payload_ir_text: str | None = None
    import_diagnostics: list[ImportDiagnostic] = field(default_factory=list)
    kernel_contracts: list[KernelContract] = field(default_factory=list)
    artifact_dir: Path | None = None
    errors: list[str] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        """One-line summary."""
        q = "FP8" if self.quantization_applied else "none"
        cov = f"{self.graph_analysis.coverage_pct:.1f}%" if self.graph_analysis else "N/A"
        aligned = "PASS" if self.alignment_result and self.alignment_result.passed else "FAIL"
        ir = "OK" if self.payload_ir_text else "N/A"
        return (
            f"{self.model_name}: params={self.param_count:,}, quant={q}, "
            f"coverage={cov}, alignment={aligned}, IR={ir}, errors={len(self.errors)}"
        )


class QuantizedModelPipeline:
    """Reusable pipeline: quantize → verify → capture → analyze → lower → bundle.

    Args:
        model: PyTorch model (nn.Module).
        sample_inputs: Tuple of sample input tensors.
        model_name: Name for reporting and artifact directory.
        quant_config: Quantization configuration (e.g., ``QuantizationConfig``).
            If ``None``, quantization is skipped.
        target_op_map: Op map for graph analysis.  Defaults to NPU op map.
        output_dir: Directory for artifact output.  If ``None``, artifacts
            are not saved to disk.
    """

    def __init__(
        self,
        model: nn.Module,
        sample_inputs: tuple[torch.Tensor, ...],
        *,
        model_name: str = "model",
        quant_config: QuantizationConfig | None = None,
        target_op_map: dict[str, Any] | None = None,
        output_dir: str | Path | None = None,
    ) -> None:
        self._model = model
        self._sample_inputs = sample_inputs
        self._model_name = model_name
        self._quant_config = quant_config
        self._target_op_map = target_op_map
        self._output_dir = Path(output_dir) if output_dir is not None else None
        self._report = PipelineReport(model_name=model_name)
        self._report.param_count = sum(p.numel() for p in model.parameters())

    # ------------------------------------------------------------------
    # Individual steps
    # ------------------------------------------------------------------

    def step_quantize(self) -> QuantizedModelPipeline:
        """Apply quantization to the model."""
        if self._quant_config is None:
            logger.info("pipeline_skip_quantize", reason="no quant_config")
            return self

        t0 = time.monotonic()
        try:
            self._model = apply_quantization(self._model, self._quant_config)
            self._report.quantization_applied = True
            logger.info("pipeline_quantize", scheme=self._quant_config.scheme)
        except Exception as e:
            self._report.errors.append(f"quantize: {e}")
            logger.warning("pipeline_quantize_failed", error=str(e))
        self._report.timings["quantize"] = time.monotonic() - t0
        return self

    def step_verify_alignment(self) -> NpuAlignmentResult:
        """Verify NPU hardware alignment."""
        t0 = time.monotonic()
        result = npu_alignment_check(self._model)
        self._report.alignment_result = result
        self._report.timings["verify_alignment"] = time.monotonic() - t0
        logger.info(
            "pipeline_alignment",
            passed=result.passed,
            fp8_linears=result.fp8_linear_count,
        )
        return result

    def step_rewrite_for_export(self) -> QuantizedModelPipeline:
        """Rewrite quantized modules for torch.export compatibility."""
        t0 = time.monotonic()
        try:
            rewrite_for_export(self._model)
        except Exception as e:
            self._report.errors.append(f"rewrite_for_export: {e}")
            logger.warning("pipeline_rewrite_failed", error=str(e))
        self._report.timings["rewrite_for_export"] = time.monotonic() - t0
        return self

    def step_capture(self) -> CaptureArtifact:
        """Capture the model via TorchDynamo partitioned capture."""
        t0 = time.monotonic()
        artifact = capture_dynamo_partitions(self._model, self._sample_inputs)
        self._report.capture_artifact = artifact
        self._report.timings["capture"] = time.monotonic() - t0
        logger.info(
            "pipeline_capture",
            partitions=artifact.graph_count,
            graph_breaks=artifact.graph_break_count,
        )
        return artifact

    def step_analyze_graph(self) -> QuantizedGraphAnalysis:
        """Analyze captured FX graphs for target op coverage."""
        t0 = time.monotonic()
        artifact = self._report.capture_artifact
        if artifact is None or not artifact.graphs:
            self._report.errors.append("analyze_graph: no captured graphs available")
            return QuantizedGraphAnalysis()

        graphs = list(artifact.graphs)
        if self._target_op_map is not None:
            analysis = analyze_fx_graphs(graphs, self._target_op_map)
        else:
            analysis = analyze_for_npu(graphs)

        self._report.graph_analysis = analysis
        self._report.timings["analyze_graph"] = time.monotonic() - t0
        logger.info(
            "pipeline_analyze",
            total_ops=analysis.total_ops,
            coverage_pct=f"{analysis.coverage_pct:.1f}%",
            mxu_ops=analysis.estimated_mxu_ops,
        )
        return analysis

    def step_build_patterns(self) -> None:
        """Build the kernel pattern catalog and generate golden data."""
        t0 = time.monotonic()
        artifact = self._report.capture_artifact
        if artifact is None or not artifact.graphs:
            return

        try:
            from compgen.kernels.patterns.detection import detect_patterns_in_graphs
            from compgen.kernels.patterns.catalog import build_pattern_catalog, export_pattern_catalog
            from compgen.kernels.golden.generator import generate_golden_for_pattern
            from compgen.kernels.golden.export import export_golden_data, export_test_harness
            from compgen.transforms.graph_passes import run_decomposition_on_graphs

            graphs = list(artifact.graphs)

            # Run decomposition passes (annotate fusion patterns)
            decomp_stats = run_decomposition_on_graphs(graphs)
            logger.info("pipeline_decompose", **decomp_stats)

            # Detect patterns
            detected = detect_patterns_in_graphs(graphs)
            logger.info("pipeline_patterns_detected", count=len(detected))

            # Build catalog
            patterns = build_pattern_catalog(detected)
            logger.info("pipeline_pattern_catalog", patterns=len(patterns))

            # Generate golden data for each pattern (small + real)
            golden_cases = []
            for pattern in patterns:
                for variant in ("small", "real"):
                    try:
                        tc = generate_golden_for_pattern(pattern, variant=variant)
                        if tc.expected_output is not None:
                            golden_cases.append(tc)
                    except Exception as e:
                        logger.debug("golden_gen_skip", pattern=pattern.pattern_id, variant=variant, error=str(e))

            # Export to artifacts
            if self._output_dir:
                patterns_dir = self._output_dir / "kernel_patterns"
                export_pattern_catalog(patterns, patterns_dir)

                golden_dir = self._output_dir / "kernel_patterns"
                export_golden_data(golden_cases, golden_dir)

                # Export test harnesses
                for tc in golden_cases:
                    harness_dir = golden_dir / tc.pattern_id
                    export_test_harness(tc, harness_dir)

                logger.info("pipeline_golden_data", cases=len(golden_cases), dir=str(golden_dir))

            # Store in report for reference
            self._report.metadata["pattern_count"] = len(patterns)
            self._report.metadata["golden_cases"] = len(golden_cases)

        except Exception as e:
            self._report.errors.append(f"build_patterns: {e}")
            logger.warning("pipeline_patterns_failed", error=str(e))

        self._report.timings["build_patterns"] = time.monotonic() - t0

    def step_to_payload_ir(self) -> ModuleOp | None:
        """Convert captured FX graphs to Payload IR."""
        t0 = time.monotonic()
        artifact = self._report.capture_artifact
        if artifact is None:
            self._report.errors.append("to_payload_ir: no capture artifact")
            return None

        # Use the first graph partition for Payload IR conversion
        if artifact.graphs:
            try:
                # Export the first partition through fx_to_xdsl
                first_graph = artifact.graphs[0]
                # Build a minimal exported program from the dynamo graph
                module, diagnostics = fx_to_xdsl(first_graph)
                self._report.import_diagnostics.extend(diagnostics)

                # Extract kernel contracts
                contracts = extract_contracts(module)
                self._report.kernel_contracts = contracts

                # Serialize to MLIR text
                stream = io.StringIO()
                Printer(stream=stream).print(module)
                self._report.payload_ir_text = stream.getvalue()

                logger.info(
                    "pipeline_payload_ir",
                    ops=len(contracts),
                    diagnostics=len(diagnostics),
                )
                self._report.timings["to_payload_ir"] = time.monotonic() - t0
                return module
            except Exception as e:
                self._report.errors.append(f"to_payload_ir: {e}")
                logger.warning("pipeline_payload_ir_failed", error=str(e))

        self._report.timings["to_payload_ir"] = time.monotonic() - t0
        return None

    def step_save_artifacts(self) -> Path | None:
        """Save all artifacts to the output directory (standard contract)."""
        if self._output_dir is None:
            return None

        t0 = time.monotonic()
        out = self._output_dir
        out.mkdir(parents=True, exist_ok=True)
        (out / "kernel_contracts").mkdir(exist_ok=True)

        # Golden inputs
        try:
            torch.save(self._sample_inputs, out / "golden_inputs.pt")
        except Exception as e:
            self._report.errors.append(f"save golden_inputs: {e}")

        # Golden outputs (reference from unquantized forward)
        try:
            with torch.no_grad():
                ref_out = self._model(*self._sample_inputs)
            if isinstance(ref_out, torch.Tensor):
                torch.save(ref_out, out / "golden_outputs.pt")
            elif isinstance(ref_out, tuple):
                torch.save(ref_out[0] if ref_out else None, out / "golden_outputs.pt")
        except Exception as e:
            self._report.errors.append(f"save golden_outputs: {e}")

        # Graph analysis
        if self._report.graph_analysis is not None:
            (out / "graph_analysis.json").write_text(self._report.graph_analysis.to_json())

        # Alignment report
        if self._report.alignment_result is not None:
            ar = self._report.alignment_result
            (out / "alignment_report.json").write_text(json.dumps({
                "passed": ar.passed,
                "fp8_linear_count": ar.fp8_linear_count,
                "fp8_conv2d_count": ar.fp8_conv2d_count,
                "fp8_attention_count": ar.fp8_attention_count,
                "non_po2_scales": ar.non_po2_scales,
                "unquantized_linears": ar.unquantized_linears,
                "warnings": ar.warnings,
                "errors": ar.errors,
            }, indent=2))

        # Payload IR
        if self._report.payload_ir_text:
            (out / "payload.mlir").write_text(self._report.payload_ir_text)

        # Kernel contracts
        for i, contract in enumerate(self._report.kernel_contracts):
            contract_path = out / "kernel_contracts" / f"contract_{i:04d}_{contract.op_name.replace('.', '_')}.yaml"
            import yaml
            contract_path.write_text(yaml.dump({
                "op_name": contract.op_name,
                "supported_dtypes": list(contract.supported_dtypes),
                "fusable": contract.fusable,
                "cost": {
                    "flops": contract.cost.flops,
                    "bytes_read": contract.cost.bytes_read,
                    "bytes_written": contract.cost.bytes_written,
                },
            }, default_flow_style=False))

        # Verification report (combines alignment + graph analysis)
        verification = {
            "model_name": self._report.model_name,
            "param_count": self._report.param_count,
            "quantization_applied": self._report.quantization_applied,
            "alignment_passed": self._report.alignment_result.passed if self._report.alignment_result else None,
            "graph_coverage_pct": self._report.graph_analysis.coverage_pct if self._report.graph_analysis else None,
            "errors": self._report.errors,
            "timings": self._report.timings,
        }
        (out / "verification_report.json").write_text(json.dumps(verification, indent=2))

        # Manifest
        manifest = {
            "model_name": self._report.model_name,
            "param_count": self._report.param_count,
            "quantization_scheme": self._quant_config.scheme if self._quant_config else None,
            "target": "npu" if self._target_op_map is None else "custom",
            "artifacts": [
                str(p.relative_to(out)) for p in sorted(out.rglob("*")) if p.is_file()
            ],
        }
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2))

        self._report.artifact_dir = out
        self._report.timings["save_artifacts"] = time.monotonic() - t0
        logger.info("pipeline_artifacts_saved", dir=str(out))
        return out

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def run(self) -> PipelineReport:
        """Run the full pipeline: quantize → verify → rewrite → capture → analyze → IR → save.

        Returns:
            Comprehensive ``PipelineReport``.
        """
        total_t0 = time.monotonic()

        # 1. Quantize
        self.step_quantize()

        # 2. Verify alignment
        if self._report.quantization_applied:
            self.step_verify_alignment()

        # 3. Rewrite for export
        if self._report.quantization_applied:
            self.step_rewrite_for_export()

        # 4. Capture
        self.step_capture()

        # 5. Analyze graph
        self.step_analyze_graph()

        # 6. Build pattern catalog + golden data
        self.step_build_patterns()

        # 7. Convert to Payload IR
        self.step_to_payload_ir()

        # 8. Save artifacts
        self.step_save_artifacts()

        self._report.timings["total"] = time.monotonic() - total_t0
        logger.info("pipeline_complete", summary=self._report.summary())
        return self._report


__all__ = [
    "PipelineReport",
    "QuantizedModelPipeline",
]
