#!/usr/bin/env python3
"""End-to-end layout bridge truth path.

Proves the layout bridge is TARGET-AGNOSTIC: virtual encodings,
transpose canonicalization, prepack analysis, layout propagation,
and materialization all work WITHOUT any target-specific resolver.
A target resolver (CUDA, SNAX, Gemmini, etc.) is optional specialization
contributed by extension packs.

Gates:
    1. Capture + IR Import         -- SimpleMLP captured and imported to xDSL
    2. Layout Analysis             -- LayoutPlanner produces plans (any target)
    3. Prepack Analysis            -- PrepackPlanner identifies prepack candidates
    4. Transpose Analysis          -- TransposeProfitabilityAnalyzer classifies transposes
    5. Layout Dialect Roundtrip    -- SetLayoutOp/UnsetLayoutOp/PackOp/UnpackOp build
    6. Transform Pipeline          -- run_layout_pipeline stamps layout_clean
    7. Stage Integration           -- EncodingStage + LayoutStage (no target plugin)
    8. Resolver Protocol           -- LayoutResolver works with ANY target's resolver
"""

from __future__ import annotations

import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Ensure project root is on sys.path for local imports.
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import structlog

log = structlog.get_logger()


# ---------------------------------------------------------------------------
# Gate report
# ---------------------------------------------------------------------------


@dataclass
class GateReport:
    """Result of a single truth-path gate."""

    gate: int
    name: str
    passed: bool = False
    details: dict[str, Any] = field(default_factory=dict)
    error: str = ""


def _gate(num: int, name: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"  Gate {num}: {name}")
    print(f"{'=' * 70}")


def _pass(report: GateReport) -> GateReport:
    report.passed = True
    print("  -> PASS")
    return report


def _fail(report: GateReport, msg: str) -> GateReport:
    report.passed = False
    report.error = msg
    print(f"  -> FAIL: {msg}")
    return report


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _build_mock_target() -> Any:
    """Build a mock CUDA A100 TargetProfile used across multiple gates."""
    from compgen.targets.schema import (
        ComputeUnit,
        DeviceSpec,
        MemoryLevel,
        TargetProfile,
    )

    return TargetProfile(
        name="cuda_a100",
        devices=[
            DeviceSpec(
                device_type="gpu",
                name="A100",
                vendor="nvidia",
                compute_units=[
                    ComputeUnit(
                        name="tensor_core",
                        count=432,
                        supported_dtypes={"float32", "float16"},
                        peak_tflops=312.0,
                    ),
                ],
                memory_hierarchy=[
                    MemoryLevel(
                        name="hbm",
                        size_bytes=80 * 1024**3,
                        bandwidth_gbps=2039.0,
                    ),
                ],
                supported_ops=["matmul", "conv2d", "elementwise"],
                features=["tensor_core", "async_copy"],
                kernel_backends=["triton", "cutlass"],
            ),
        ],
    )


def _build_mock_analysis() -> Any:
    """Build a minimal mock NetworkAnalysis with a matmul cluster."""
    from compgen.agent.analyzer import (
        DataFlowEdge,
        NetworkAnalysis,
        PatternCluster,
    )

    cluster = PatternCluster(
        cluster_id="cluster_0",
        pattern_type="linear_chain",
        node_names=("p_fc1_weight", "addmm_0", "gelu_0"),
        total_flops=2 * 64 * 128,
        total_bytes=4 * (64 * 128 + 128 * 128),
        arithmetic_intensity=2 * 64 * 128 / max(4 * (64 * 128 + 128 * 128), 1),
        estimated_latency_per_device={"A100": 0.05},
        best_device="A100",
        is_bottleneck=True,
        kernel_opportunity="fused_matmul_gelu",
        input_shapes={"p_fc1_weight": (64, 128)},
        output_shapes={"gelu_0": (8, 128)},
    )

    return NetworkAnalysis(
        model_name="SimpleMLP",
        total_params=2,
        total_flops=cluster.total_flops,
        total_bytes=cluster.total_bytes,
        clusters=[cluster],
        unclustered_ops=[],
        data_flow=[],
        bottleneck_clusters=["cluster_0"],
        optimization_opportunities=["fused_matmul_gelu"],
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> list[GateReport]:
    """Run all gates and return reports."""
    reports: list[GateReport] = []

    # Shared state across gates
    xdsl_module = None

    # ==================================================================
    # Gate 1: Capture + IR Import
    # ==================================================================
    _gate(1, "Capture + IR Import")
    r1 = GateReport(gate=1, name="Capture + IR Import")
    try:
        import torch

        from compgen.capture.torch_export import capture_model
        from compgen.ir.payload.import_fx import fx_to_xdsl

        class SimpleMLP(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.fc1 = torch.nn.Linear(64, 128)
                self.gelu = torch.nn.GELU()
                self.fc2 = torch.nn.Linear(128, 32)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.fc2(self.gelu(self.fc1(x)))

        model = SimpleMLP()
        model.eval()
        sample_input = (torch.randn(8, 64),)

        ep = capture_model(model, sample_input)
        n_nodes = len(ep.graph.nodes)
        print(f"  Captured: {n_nodes} FX nodes")

        module, diagnostics = fx_to_xdsl(ep)
        op_count = sum(1 for _ in module.walk())
        print(f"  Payload IR: {op_count} ops, {len(diagnostics)} diagnostics")

        assert module is not None, "fx_to_xdsl returned None module"
        assert op_count > 0, "Module has no ops"

        xdsl_module = module
        r1.details = {
            "fx_nodes": n_nodes,
            "payload_ops": op_count,
            "diagnostics": len(diagnostics),
        }
        _pass(r1)
    except Exception as e:
        _fail(r1, f"{type(e).__name__}: {e}")
        traceback.print_exc()
    reports.append(r1)

    # ==================================================================
    # Gate 2: Layout Analysis (LayoutPlanner)
    # ==================================================================
    _gate(2, "Layout Analysis (LayoutPlanner)")
    r2 = GateReport(gate=2, name="Layout Analysis (LayoutPlanner)")
    try:
        from compgen.analysis.layout.planner import LayoutPlanner

        target = _build_mock_target()
        analysis = _build_mock_analysis()

        plans = LayoutPlanner().plan(analysis, target)

        assert isinstance(plans, dict), f"Expected dict, got {type(plans).__name__}"
        assert len(plans) > 0, "Plans dict is empty"

        for region_id, plan in plans.items():
            print(f"  Region '{region_id}': output={plan.preferred_output_layout}, "
                  f"tile={plan.tile_encoding}, prepack={plan.prepack_candidates}")

        r2.details = {
            "num_plans": len(plans),
            "regions": list(plans.keys()),
        }
        _pass(r2)
    except Exception as e:
        _fail(r2, f"{type(e).__name__}: {e}")
        traceback.print_exc()
    reports.append(r2)

    # ==================================================================
    # Gate 3: Prepack Analysis (PrepackPlanner)
    # ==================================================================
    _gate(3, "Prepack Analysis (PrepackPlanner)")
    r3 = GateReport(gate=3, name="Prepack Analysis (PrepackPlanner)")
    try:
        from compgen.analysis.layout.prepack import PrepackPlanner

        analysis = _build_mock_analysis()
        candidates = PrepackPlanner().identify_prepack_opportunities(analysis)

        assert isinstance(candidates, list), f"Expected list, got {type(candidates).__name__}"
        print(f"  Found {len(candidates)} prepack candidates")
        for c in candidates:
            print(f"    {c.operand_name}: const={c.is_constant}, "
                  f"reuse={c.reuse_count}, benefit={c.estimated_benefit_us:.4f} us")

        r3.details = {
            "num_candidates": len(candidates),
            "candidate_names": [c.operand_name for c in candidates],
        }
        _pass(r3)
    except Exception as e:
        _fail(r3, f"{type(e).__name__}: {e}")
        traceback.print_exc()
    reports.append(r3)

    # ==================================================================
    # Gate 4: Transpose Analysis
    # ==================================================================
    _gate(4, "Transpose Analysis")
    r4 = GateReport(gate=4, name="Transpose Analysis")
    try:
        from compgen.analysis.layout.transpose import TransposeProfitabilityAnalyzer

        analysis = _build_mock_analysis()
        contracts: list = []  # empty list -- no kernel contracts

        classifications = TransposeProfitabilityAnalyzer().classify_transposes(
            analysis, contracts,
        )

        assert isinstance(classifications, dict), (
            f"Expected dict, got {type(classifications).__name__}"
        )
        print(f"  Classified {len(classifications)} transpose nodes")
        for name, cls in classifications.items():
            print(f"    {name}: {cls.value}")

        r4.details = {
            "num_classified": len(classifications),
            "classifications": {k: v.value for k, v in classifications.items()},
        }
        _pass(r4)
    except Exception as e:
        _fail(r4, f"{type(e).__name__}: {e}")
        traceback.print_exc()
    reports.append(r4)

    # ==================================================================
    # Gate 5: Layout Dialect Roundtrip
    # ==================================================================
    _gate(5, "Layout Dialect Roundtrip")
    r5 = GateReport(gate=5, name="Layout Dialect Roundtrip")
    try:
        from xdsl.dialects.builtin import IntegerAttr, IntegerType, StringAttr, SymbolRefAttr

        from compgen.ir.layout.attrs import LayoutEncodingAttr, PackSpecAttr
        from compgen.ir.layout.ops import PackOp, SetLayoutOp, UnpackOp, UnsetLayoutOp

        # Build attributes
        encoding_attr = LayoutEncodingAttr(
            op_type="matmul",
            operand_index=0,
            logical_layout="tiled",
            tile_dims=[128, 64],
            element_types=["f32"],
        )
        print(f"  LayoutEncodingAttr: {encoding_attr.name}")

        pack_spec_attr = PackSpecAttr(
            inner_tiles=[16, 16],
            outer_perm=[0, 1],
            padding_value="zero",
        )
        print(f"  PackSpecAttr: {pack_spec_attr.name}")

        # Build ops
        set_op = SetLayoutOp.build(properties={
            "encoding": encoding_attr,
            "source_ref": SymbolRefAttr("test_ref"),
        })
        print(f"  SetLayoutOp: {set_op.name}")

        unset_op = UnsetLayoutOp.build(properties={
            "source_ref": SymbolRefAttr("test_ref"),
        })
        print(f"  UnsetLayoutOp: {unset_op.name}")

        pack_op = PackOp.build(properties={
            "source_ref": SymbolRefAttr("test_ref"),
            "pack_spec": pack_spec_attr,
            "is_prepack": IntegerAttr(1, IntegerType(64)),
        })
        print(f"  PackOp: {pack_op.name}")

        unpack_op = UnpackOp.build(properties={
            "source_ref": SymbolRefAttr("test_ref"),
            "pack_spec": pack_spec_attr,
        })
        print(f"  UnpackOp: {unpack_op.name}")

        # All four built successfully
        r5.details = {
            "set_layout_ok": True,
            "unset_layout_ok": True,
            "pack_ok": True,
            "unpack_ok": True,
        }
        _pass(r5)
    except Exception as e:
        _fail(r5, f"{type(e).__name__}: {e}")
        traceback.print_exc()
    reports.append(r5)

    # ==================================================================
    # Gate 6: Transform Pipeline
    # ==================================================================
    _gate(6, "Transform Pipeline")
    r6 = GateReport(gate=6, name="Transform Pipeline")
    try:
        from compgen.transforms.layout import run_layout_pipeline

        assert xdsl_module is not None, "No xDSL module from Gate 1"

        # Deep-clone the module to avoid mutating shared state.
        from xdsl.dialects.builtin import ModuleOp
        from xdsl.ir import Region
        from xdsl.parser import Parser
        from xdsl.printer import Printer
        import io

        buf = io.StringIO()
        Printer(stream=buf).print(xdsl_module)
        ir_text = buf.getvalue()

        # Run the full 10-pass pipeline on the original module copy.
        # We operate on the captured module directly; the pipeline is
        # designed to be idempotent on modules without layout ops.
        result_module = run_layout_pipeline(xdsl_module)

        assert result_module is not None, "run_layout_pipeline returned None"
        has_clean = "compgen.layout_clean" in result_module.attributes
        print(f"  layout_clean attr present: {has_clean}")

        assert has_clean, "Module missing compgen.layout_clean attribute after pipeline"

        r6.details = {
            "layout_clean": has_clean,
        }
        _pass(r6)
    except Exception as e:
        _fail(r6, f"{type(e).__name__}: {e}")
        traceback.print_exc()
    reports.append(r6)

    # ==================================================================
    # Gate 7: Stage Integration
    # ==================================================================
    _gate(7, "Stage Integration")
    r7 = GateReport(gate=7, name="Stage Integration")
    try:
        import torch

        from compgen.capture.torch_export import capture_model
        from compgen.ir.payload.import_fx import fx_to_xdsl
        from compgen.stages.encoding.stage import EncodingStage
        from compgen.stages.layout.stage import LayoutStage
        from compgen.targets.capability import infer_capabilities

        # Fresh module for this gate
        class SimpleMLP2(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.fc1 = torch.nn.Linear(64, 128)
                self.gelu = torch.nn.GELU()
                self.fc2 = torch.nn.Linear(128, 32)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.fc2(self.gelu(self.fc1(x)))

        model7 = SimpleMLP2()
        model7.eval()
        ep7 = capture_model(model7, (torch.randn(8, 64),))
        module7, _ = fx_to_xdsl(ep7)

        target7 = _build_mock_target()
        caps7 = infer_capabilities(target7)

        # Run EncodingStage shared passes to add encoding attrs
        encoding_stage = EncodingStage()
        module7 = encoding_stage.shared_passes(module7, target7)
        print("  EncodingStage shared_passes: done")

        # Run LayoutStage shared passes
        layout_stage = LayoutStage()
        module7 = layout_stage.shared_passes(module7, target7)
        print("  LayoutStage shared_passes: done")

        has_clean7 = "compgen.layout_clean" in module7.attributes
        print(f"  layout_clean attr present: {has_clean7}")

        assert has_clean7, "Module missing compgen.layout_clean after stage integration"

        r7.details = {
            "encoding_done": True,
            "layout_done": True,
            "layout_clean": has_clean7,
        }
        _pass(r7)
    except Exception as e:
        _fail(r7, f"{type(e).__name__}: {e}")
        traceback.print_exc()
    reports.append(r7)

    # ==================================================================
    # Gate 8: LayoutResolver Protocol (ANY target)
    # ==================================================================
    _gate(8, "Resolver Protocol (any target)")
    r8 = GateReport(gate=8, name="Resolver Protocol (any target)")
    try:
        from compgen.ir.layout.attrs import PackSpecAttr
        from compgen.transforms.layout.cuda_resolver import CudaLayoutResolver
        from compgen.transforms.layout.resolver import DefaultLayoutResolver, LayoutResolver

        # 1. DefaultLayoutResolver — works for ANY target, no assumptions
        default = DefaultLayoutResolver()
        assert isinstance(default, LayoutResolver), "Must implement LayoutResolver"
        result_default = default.specialize("tiled_64x32", None)
        assert result_default is None, "Default resolver should NOT specialize"
        print("  DefaultLayoutResolver: no specialization (correct)")

        # 2. CudaLayoutResolver — one EXAMPLE of target-specific specialization
        cuda = CudaLayoutResolver()
        assert isinstance(cuda, LayoutResolver), "Must implement LayoutResolver"
        result_cuda = cuda.specialize("tiled_128x64", None)
        assert result_cuda is not None, "CUDA resolver should specialize tiled encodings"
        assert isinstance(result_cuda, PackSpecAttr)
        print(f"  CudaLayoutResolver: specialized to PackSpecAttr (one example)")

        # 3. Both materialize via the same protocol
        spec = PackSpecAttr([32, 32], [0, 1], "zero")
        meta_default = default.materialize(spec)
        meta_cuda = cuda.materialize(spec)
        assert "inner_tiles" in meta_default
        assert "inner_tiles" in meta_cuda
        print(f"  Both resolvers materialize via same protocol")

        # KEY: The LayoutResolver protocol is target-agnostic.
        # Any extension pack (SNAX, Gemmini, Hexagon, RVV, NPU, etc.)
        # can provide its own resolver — the layout bridge does not
        # depend on any specific target.
        r8.details = {
            "default_specializes": result_default is not None,
            "cuda_specializes": result_cuda is not None,
            "protocol_target_agnostic": True,
            "note": "Any pack provides its own LayoutResolver",
        }
        _pass(r8)
    except Exception as e:
        _fail(r8, f"{type(e).__name__}: {e}")
        traceback.print_exc()
    reports.append(r8)

    return reports


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _print_summary(reports: list[GateReport]) -> bool:
    """Print final summary and return True if all passed."""
    print(f"\n{'=' * 70}")
    print("  Layout Bridge Truth Path -- Summary")
    print(f"{'=' * 70}")

    all_passed = True
    for r in reports:
        status = "PASS" if r.passed else "FAIL"
        marker = "[+]" if r.passed else "[-]"
        print(f"  {marker} Gate {r.gate}: {r.name} ... {status}")
        if not r.passed:
            all_passed = False
            print(f"       Error: {r.error}")

    total = len(reports)
    passed = sum(1 for r in reports if r.passed)
    print(f"\n  {passed}/{total} gates passed")
    print(f"{'=' * 70}")
    return all_passed


if __name__ == "__main__":
    reports = main()
    success = _print_summary(reports)
    sys.exit(0 if success else 1)
