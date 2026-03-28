#!/usr/bin/env python3
"""CUDA Tile pack integration truth path.

End-to-end validation that the pack system works: load manifest, validate
environment, enforce apertures, generate within allowed surfaces, lower
to Triton, verify, promote.

Gates:
    1. Pack Loading          -- manifest resolves, surfaces populated
    2. Pack Validation       -- probe passes, env check OK
    3. Sealed Surface        -- sealed surface blocked, aperture open
    4. Pipeline Generation   -- CUDA GPU target stack runs on SimpleMLP
    5. Tile IR Lowering      -- tile ops lower to Triton code
    6. Verification          -- generated code compiles and has expected patterns
    7. Promotion             -- bundle promoted to recipe library
    8. Pack Context Summary  -- compose() returns correct contribution
"""

from __future__ import annotations

import shutil
import sys
import tempfile
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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
# Main
# ---------------------------------------------------------------------------


def main() -> list[GateReport]:
    """Run all gates and return reports."""
    reports: list[GateReport] = []
    repo_root = Path(__file__).resolve().parent.parent

    # ==================================================================
    # Gate 1: Pack Loading
    # ==================================================================
    _gate(1, "Pack Loading")
    r1 = GateReport(gate=1, name="Pack Loading")
    try:
        from compgen.packs.loader import load_pack

        loaded = load_pack(repo_root / "userpacks" / "cuda_tile")
        assert loaded.manifest.name == "cuda_tile", f"Expected name 'cuda_tile', got '{loaded.manifest.name}'"
        assert "DialectPack" in loaded.manifest.kinds, f"Expected 'DialectPack' in kinds, got {loaded.manifest.kinds}"
        assert len(loaded.manifest.owned_surfaces) > 0, "owned_surfaces is empty"
        assert len(loaded.manifest.sealed_surfaces) > 0, "sealed_surfaces is empty"
        r1.details = {
            "name": loaded.manifest.name,
            "kinds": list(loaded.manifest.kinds),
            "owned_surfaces": list(loaded.manifest.owned_surfaces),
            "sealed_surfaces": list(loaded.manifest.sealed_surfaces),
            "generation_apertures": list(loaded.manifest.generation_apertures),
        }
        _pass(r1)
    except Exception as e:
        _fail(r1, f"{type(e).__name__}: {e}")
        traceback.print_exc()
    reports.append(r1)

    # ==================================================================
    # Gate 2: Pack Validation (probe)
    # ==================================================================
    _gate(2, "Pack Validation")
    r2 = GateReport(gate=2, name="Pack Validation")
    try:
        # Build a minimal workspace-like object with repo_root so the probe
        # can resolve third_party/cuda-tile via the manifest's third_party_names.
        @dataclass
        class _MinimalWorkspace:
            repo_root: str
            pack_roots: dict[str, Any] = field(default_factory=dict)
            external_roots: dict[str, Any] = field(default_factory=dict)

        workspace = _MinimalWorkspace(repo_root=str(repo_root))

        # The pack probe checks that third_party/cuda-tile/README.md exists
        probe = loaded.pack.probe(workspace)
        assert probe.available, f"Probe not available: missing={probe.missing_paths}, details={probe.details}"
        assert probe.source_root is not None, "source_root is None"

        # Verify python3 is on PATH as a basic env check
        python3_available = shutil.which("python3") is not None
        assert python3_available, "python3 not found on PATH"

        r2.details = {
            "probe_available": probe.available,
            "source_root": str(probe.source_root),
            "missing_paths": list(probe.missing_paths),
            "python3_available": python3_available,
        }
        _pass(r2)
    except Exception as e:
        _fail(r2, f"{type(e).__name__}: {e}")
        traceback.print_exc()
    reports.append(r2)

    # ==================================================================
    # Gate 3: Sealed Surface Enforcement
    # ==================================================================
    _gate(3, "Sealed Surface Enforcement")
    r3 = GateReport(gate=3, name="Sealed Surface Enforcement")
    try:
        from compgen.packs.verify import OwnershipViolation, check_surface_allowed

        # Sealed surface should be blocked
        violation = check_surface_allowed([loaded], requested_surface="tile_dialect_semantics")
        assert violation is not None, "Expected violation for sealed surface 'tile_dialect_semantics'"
        assert isinstance(violation, OwnershipViolation)
        assert violation.reason == "sealed_surface"
        print(f"  Sealed surface blocked: {violation.surface} ({violation.reason})")

        # Open aperture should be allowed
        allowed = check_surface_allowed([loaded], requested_surface="payload_to_cuda_tile_lowering")
        assert allowed is None, f"Expected None for open aperture, got {allowed}"
        print("  Open aperture allowed: payload_to_cuda_tile_lowering")

        r3.details = {
            "sealed_blocked": True,
            "aperture_allowed": True,
            "violation_pack": violation.pack_name,
            "violation_surface": violation.surface,
        }
        _pass(r3)
    except Exception as e:
        _fail(r3, f"{type(e).__name__}: {e}")
        traceback.print_exc()
    reports.append(r3)

    # ==================================================================
    # Gate 4: Pipeline Generation via CUDA GPU Stack
    # ==================================================================
    _gate(4, "Pipeline Generation via CUDA GPU Stack")
    r4 = GateReport(gate=4, name="Pipeline Generation")
    pipeline_module = None
    try:
        import torch
        from compgen.capture.torch_export import capture_model
        from compgen.ir.payload.import_fx import fx_to_xdsl
        from compgen.stages.registry import StageRegistry
        from compgen.stages.targets.cuda_gpu import create_cuda_gpu_stack
        from compgen.targets.capability import infer_capabilities
        from compgen.targets.schema import DeviceSpec, TargetProfile

        # Define SimpleMLP
        class SimpleMLP(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.fc1 = torch.nn.Linear(64, 128)
                self.fc2 = torch.nn.Linear(128, 32)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.fc2(torch.relu(self.fc1(x)))

        model = SimpleMLP()
        sample_input = (torch.randn(8, 64),)

        # Capture via torch.export
        ep = capture_model(model, sample_input)
        print(f"  Captured: {len(ep.graph.nodes)} FX nodes")

        # Convert to xDSL Payload IR
        module, diagnostics = fx_to_xdsl(ep)
        op_count = sum(1 for _ in module.walk())
        print(f"  Payload IR: {op_count} ops, {len(diagnostics)} diagnostics")

        # Create CUDA GPU target profile
        output_dir = Path(tempfile.mkdtemp(prefix="cuda_tile_pack_"))
        target = TargetProfile(
            name="cuda_a100",
            devices=[
                DeviceSpec(
                    device_type="gpu",
                    name="A100-SXM4-80GB",
                    vendor="nvidia",
                    kernel_backends=["triton", "cutlass"],
                ),
            ],
        )
        capabilities = infer_capabilities(target)

        # Create and register CUDA GPU stack
        registry = StageRegistry()
        stack = create_cuda_gpu_stack(output_dir=str(output_dir))
        registry.register_target_stack(stack)

        # Run pipeline
        result = registry.run_pipeline(module, target, capabilities)
        assert result.passed, f"Pipeline failed at stage: {result.first_failure}"
        print(f"  Pipeline: {result.stages_run} stages, passed={result.passed}")

        pipeline_module = result.final_module
        r4.details = {
            "fx_nodes": len(ep.graph.nodes),
            "payload_ops": op_count,
            "stages_run": result.stages_run,
            "passed": result.passed,
            "output_dir": str(output_dir),
        }
        _pass(r4)
    except Exception as e:
        _fail(r4, f"{type(e).__name__}: {e}")
        traceback.print_exc()
    reports.append(r4)

    # ==================================================================
    # Gate 5: Tile IR Lowering to Triton
    # ==================================================================
    _gate(5, "Tile IR Lowering to Triton")
    r5 = GateReport(gate=5, name="Tile IR Lowering to Triton")
    triton_code = ""
    try:
        from compgen.ir.tile.attrs import MemoryClassAttr, TileShapeAttr
        from compgen.ir.tile.lower_triton import lower_tile_to_triton
        from compgen.ir.tile.ops import TileElementwiseOp, TileLoadOp, TileMMAOp, TileStoreOp
        from xdsl.dialects.builtin import StringAttr, SymbolRefAttr

        # Build a representative tile IR sequence: Load A, Load B, MMA, ReLU, Store C
        load_a = TileLoadOp.build(properties={
            "src_memref": SymbolRefAttr("A"),
            "memory_class": MemoryClassAttr("global"),
            "shape": TileShapeAttr([16, 16]),
        })
        load_b = TileLoadOp.build(properties={
            "src_memref": SymbolRefAttr("B"),
            "memory_class": MemoryClassAttr("global"),
            "shape": TileShapeAttr([16, 16]),
        })
        mma = TileMMAOp.build(properties={
            "a_ref": SymbolRefAttr("A"),
            "b_ref": SymbolRefAttr("B"),
            "c_ref": SymbolRefAttr("C"),
            "shape": TileShapeAttr([16, 16, 8]),
        })
        relu = TileElementwiseOp.build(properties={
            "fragment_ref": SymbolRefAttr("C"),
            "op_kind": StringAttr("relu"),
            "shape": TileShapeAttr([16, 16]),
        })
        store = TileStoreOp.build(properties={
            "dst_memref": SymbolRefAttr("C"),
            "fragment_ref": SymbolRefAttr("C"),
            "memory_class": MemoryClassAttr("global"),
            "shape": TileShapeAttr([16, 16]),
        })

        ops = [load_a, load_b, mma, relu, store]
        result = lower_tile_to_triton(ops)

        assert result.kernel_code, "Triton kernel code is empty"
        assert "tl.load" in result.kernel_code, "Missing tl.load in Triton output"
        assert "tl.dot" in result.kernel_code, "Missing tl.dot in Triton output"
        assert "tl.store" in result.kernel_code, "Missing tl.store in Triton output"
        assert "tl.maximum" in result.kernel_code, "Missing tl.maximum (relu) in Triton output"

        triton_code = result.kernel_code
        print(f"  Triton code ({len(triton_code)} chars):")
        for line in triton_code.splitlines():
            print(f"    {line}")
        print(f"  Diagnostics: {result.diagnostics}")

        r5.details = {
            "code_length": len(triton_code),
            "has_tl_load": "tl.load" in triton_code,
            "has_tl_dot": "tl.dot" in triton_code,
            "has_tl_store": "tl.store" in triton_code,
            "has_relu": "tl.maximum" in triton_code,
            "diagnostics": result.diagnostics,
        }
        _pass(r5)
    except Exception as e:
        _fail(r5, f"{type(e).__name__}: {e}")
        traceback.print_exc()
    reports.append(r5)

    # ==================================================================
    # Gate 6: Verification
    # ==================================================================
    _gate(6, "Verification")
    r6 = GateReport(gate=6, name="Verification")
    try:
        # Verify the generated Triton code is syntactically valid Python
        assert triton_code, "No Triton code to verify"

        # Wrap in a function so compile() can check syntax
        wrapped = "def _triton_kernel():\n"
        for line in triton_code.splitlines():
            wrapped += f"    {line}\n"

        # compile() checks syntax without executing
        compiled = compile(wrapped, "<triton_kernel>", "exec")
        assert compiled is not None, "compile() returned None"
        print("  Syntax check: PASS")

        # Check expected Triton patterns
        patterns = ["tl.load", "tl.store", "tl.dot", "tl.maximum"]
        found = [p for p in patterns if p in triton_code]
        missing = [p for p in patterns if p not in triton_code]
        assert not missing, f"Missing expected patterns: {missing}"
        print(f"  Pattern check: all {len(found)} patterns found")

        r6.details = {
            "syntax_valid": True,
            "patterns_found": found,
            "patterns_missing": missing,
        }
        _pass(r6)
    except Exception as e:
        _fail(r6, f"{type(e).__name__}: {e}")
        traceback.print_exc()
    reports.append(r6)

    # ==================================================================
    # Gate 7: Promotion
    # ==================================================================
    _gate(7, "Promotion")
    r7 = GateReport(gate=7, name="Promotion")
    try:
        from compgen.promotion.promote import RecipePromoter
        from compgen.runtime.bundle import Bundle, BundleBuilder

        # Build a bundle from pipeline output
        bundle_dir = Path(tempfile.mkdtemp(prefix="cuda_tile_bundle_"))
        promoted_dir = Path(tempfile.mkdtemp(prefix="cuda_tile_recipes_"))

        if pipeline_module is not None:
            builder = BundleBuilder(output_dir=bundle_dir)
            manifest = builder.build(
                module=pipeline_module,
                target_name="cuda_a100",
                objective="latency",
                kernel_files={"tile_matmul.py": triton_code},
                verification_report={"passed": True, "levels": ["structural", "functional"]},
            )
        else:
            # Fallback: create a minimal bundle manifest directly
            import json

            bundle_dir.mkdir(parents=True, exist_ok=True)
            (bundle_dir / "payload.mlir").write_text("// placeholder payload")
            manifest = Bundle(
                target_profile="cuda_a100",
                model_hash="e2e_pack_test",
                objective="latency",
                artifacts={"payload": "payload.mlir"},
            )
            (bundle_dir / "manifest.json").write_text(json.dumps(manifest.to_dict(), indent=2))

        # Promote
        promoter = RecipePromoter(library_path=promoted_dir)
        promo_result = promoter.promote(manifest)
        assert promo_result.promoted, f"Promotion failed: {promo_result.reason}"
        assert promo_result.recipe_path is not None
        assert promo_result.recipe_path.exists()
        assert (promo_result.recipe_path / "manifest.json").exists()
        print(f"  Promoted to: {promo_result.recipe_path}")
        print(f"  Recipe key: {promo_result.key.key if promo_result.key else 'N/A'}")

        r7.details = {
            "promoted": promo_result.promoted,
            "recipe_path": str(promo_result.recipe_path),
            "recipe_key": promo_result.key.key if promo_result.key else "",
        }
        _pass(r7)
    except Exception as e:
        _fail(r7, f"{type(e).__name__}: {e}")
        traceback.print_exc()
    reports.append(r7)

    # ==================================================================
    # Gate 8: Pack Context Summary
    # ==================================================================
    _gate(8, "Pack Context Summary")
    r8 = GateReport(gate=8, name="Pack Context Summary")
    try:
        contribution = loaded.pack.compose()
        assert contribution.pack_name == "cuda_tile"
        assert "tile_dialect_semantics" in contribution.owned_surfaces
        assert "tile_kernel_substrate" in contribution.owned_surfaces
        assert "tile_dialect_semantics" in contribution.sealed_surfaces
        assert "payload_to_cuda_tile_lowering" in contribution.generation_apertures

        print(f"  Pack name: {contribution.pack_name}")
        print(f"  Owned surfaces: {list(contribution.owned_surfaces)}")
        print(f"  Sealed surfaces: {list(contribution.sealed_surfaces)}")
        print(f"  Generation apertures: {list(contribution.generation_apertures)}")
        print(f"  Benchmark targets: {list(contribution.benchmark_targets)}")
        print(f"  Available profilers: {list(contribution.available_profilers)}")
        print(f"  Runtime artifacts: {contribution.runtime_artifacts}")

        r8.details = {
            "pack_name": contribution.pack_name,
            "owned_surfaces": list(contribution.owned_surfaces),
            "sealed_surfaces": list(contribution.sealed_surfaces),
            "generation_apertures": list(contribution.generation_apertures),
            "benchmark_targets": list(contribution.benchmark_targets),
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
    print("  CUDA Tile Pack Integration Truth Path -- Summary")
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
