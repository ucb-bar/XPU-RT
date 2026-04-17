"""SmolVLA end-to-end FP8 quantization, analysis, and compilation.

Loads the real SmolVLA model from HuggingFace (via Understanding-PI0 +
LeRobot), applies FP8 E4M3 po2 quantization for NPU deployment, captures
via TorchDynamo, analyzes every op for NPU coverage, lowers to Payload IR,
and produces the standard CompGen artifact bundle.

Usage::

    python -m compgen.quantization.smolvla_e2e --output-dir artifacts/smolvla_fp8_npu

Or from Python::

    from compgen.quantization.smolvla_e2e import run_smolvla_npu_pipeline
    report = run_smolvla_npu_pipeline()
"""

from __future__ import annotations

import argparse
from pathlib import Path

import structlog

from compgen.capture.torchao_pipeline import QuantizationConfig
from compgen.quantization.graph_analyzer import format_analysis_report
from compgen.quantization.pipeline import QuantizedModelPipeline

logger = structlog.get_logger()


def run_smolvla_npu_pipeline(
    output_dir: str | Path = "artifacts/smolvla_fp8_npu",
    device: str = "cpu",
) -> object:
    """Run the full SmolVLA FP8 NPU pipeline.

    Steps:
        1. Load SmolVLA via Understanding-PI0 + LeRobot
        2. Apply FP8 E4M3 po2 quantization (NPU recipe)
        3. Verify NPU alignment (all scales po2, softmax BF16)
        4. Rewrite for export
        5. Capture via TorchDynamo partitioned capture
        6. Analyze FX graph for NPU op coverage
        7. Convert to Payload IR
        8. Save standard artifact bundle

    Args:
        output_dir: Directory for artifact output.
        device: Device for model loading (``"cpu"`` recommended for capture).

    Returns:
        ``PipelineReport`` with comprehensive results.
    """
    from compgen.models.robotics import load_smolvla_bundle

    logger.info("smolvla_e2e_start", output_dir=str(output_dir), device=device)

    # 1. Load SmolVLA
    logger.info("loading_smolvla")
    wrapper, flat_inputs, num_cams = load_smolvla_bundle(device=device)
    param_count = sum(p.numel() for p in wrapper.parameters())
    logger.info(
        "smolvla_loaded",
        params=f"{param_count:,}",
        input_tensors=len(flat_inputs),
        num_cams=num_cams,
    )

    # 2. Build and run pipeline
    pipeline = QuantizedModelPipeline(
        model=wrapper,
        sample_inputs=flat_inputs,
        model_name="smolvla_fp8_npu",
        quant_config=QuantizationConfig(scheme="fp8_e4m3_po2_npu"),
        output_dir=output_dir,
    )
    report = pipeline.run()

    # 2b. Generate deduplicated kernel contracts
    kernel_contracts_list: list[object] = []
    if report.capture_artifact and report.capture_artifact.graphs:
        from compgen.kernels.providers.npu_contracts import (
            export_contracts_autocomp,
            export_contracts_yaml,
            format_contracts_report,
            generate_npu_kernel_contracts,
        )

        logger.info("generating_kernel_contracts")
        kernel_contracts_list = generate_npu_kernel_contracts(
            list(report.capture_artifact.graphs),
            model=wrapper,
        )
        logger.info("kernel_contracts_generated", count=len(kernel_contracts_list))

        # Export YAML contracts
        contracts_dir = Path(output_dir) / "kernel_contracts"
        export_contracts_yaml(kernel_contracts_list, contracts_dir)

        # Export autocomp-ready packages
        autocomp_dir = Path(output_dir) / "autocomp_problems"
        export_contracts_autocomp(kernel_contracts_list, autocomp_dir)

        logger.info(
            "kernel_contracts_exported",
            yaml_dir=str(contracts_dir),
            autocomp_dir=str(autocomp_dir),
        )

    # 3. Print summary
    print("\n" + "=" * 70)
    print("  SmolVLA FP8 NPU Pipeline — Results")
    print("=" * 70)
    print(f"\n  {report.summary()}")

    if report.alignment_result:
        ar = report.alignment_result
        print(f"\n  NPU Alignment: {'PASSED' if ar.passed else 'FAILED'}")
        print(f"    FP8 linears:    {ar.fp8_linear_count}")
        print(f"    FP8 conv2d:     {ar.fp8_conv2d_count}")
        print(f"    FP8 attention:  {ar.fp8_attention_count}")
        if ar.unquantized_linears:
            print(f"    Unquantized:    {len(ar.unquantized_linears)}")
        if ar.errors:
            print(f"    Errors:         {len(ar.errors)}")

    if report.graph_analysis:
        print(f"\n{format_analysis_report(report.graph_analysis)}")

    if kernel_contracts_list:
        from compgen.kernels.providers.npu_contracts import format_contracts_report as fmt_kc
        print(f"\n{fmt_kc(kernel_contracts_list)}")

    if report.metadata.get("pattern_count"):
        print(f"\n  Kernel Patterns: {report.metadata['pattern_count']} reusable patterns")
        print(f"  Golden Data: {report.metadata.get('golden_cases', 0)} test cases generated")

    if report.payload_ir_text:
        ir_lines = report.payload_ir_text.count("\n")
        print(f"\n  Payload IR: {ir_lines} lines")

    if report.artifact_dir:
        print(f"\n  Artifacts saved to: {report.artifact_dir}")

    if report.errors:
        print(f"\n  Errors ({len(report.errors)}):")
        for err in report.errors:
            print(f"    - {err}")

    print(f"\n  Timings:")
    for step, t in sorted(report.timings.items()):
        print(f"    {step}: {t:.2f}s")

    print("\n" + "=" * 70)
    return report


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="SmolVLA end-to-end FP8 quantization + NPU compilation pipeline",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/smolvla_fp8_npu",
        help="Directory for artifact output (default: artifacts/smolvla_fp8_npu)",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device for model loading (default: cpu)",
    )
    args = parser.parse_args()

    run_smolvla_npu_pipeline(output_dir=args.output_dir, device=args.device)


if __name__ == "__main__":
    main()
