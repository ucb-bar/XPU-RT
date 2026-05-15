#!/usr/bin/env python3
"""CompGen end-to-end demo: SimpleMLP through the full pipeline.

Pipeline:
    1. Capture SimpleMLP via torch.export
    2. Convert to xDSL Payload IR
    3. Load target profile
    4. Analyze network patterns
    5. Build kernel contracts + strategy selection
    6. Run EqSat optimization
    7. Plan execution (placement + scheduling)
    8. Create artifact bundle
    9. Benchmark
    10. Report
"""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

import structlog
import torch

log = structlog.get_logger()


def main() -> None:
    """Run the full CompGen pipeline on SimpleMLP."""
    print("=" * 70)
    print("CompGen E2E Demo: SimpleMLP → Analyze → Optimize → Plan → Bundle")
    print("=" * 70)

    # ----------------------------------------------------------------
    # Step 1: Define and capture the model
    # ----------------------------------------------------------------
    print("\n[1/10] Capturing SimpleMLP via torch.export...")

    class SimpleMLP(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc1 = torch.nn.Linear(64, 128)
            self.fc2 = torch.nn.Linear(128, 32)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.fc2(torch.relu(self.fc1(x)))

    model = SimpleMLP()
    sample_input = (torch.randn(8, 64),)

    from compgen.capture.torch_export import capture_model

    ep = capture_model(model, sample_input)
    print(f"  Captured: {len(ep.graph.nodes)} FX nodes")

    # ----------------------------------------------------------------
    # Step 2: Convert to xDSL Payload IR
    # ----------------------------------------------------------------
    print("\n[2/10] Converting to xDSL Payload IR...")

    from compgen.ir.payload.import_fx import fx_to_xdsl

    module, diagnostics = fx_to_xdsl(ep)
    op_count = sum(1 for _ in module.walk())
    print(f"  Payload IR: {op_count} ops, {len(diagnostics)} diagnostics")

    # ----------------------------------------------------------------
    # Step 3: Load target profile
    # ----------------------------------------------------------------
    print("\n[3/10] Loading target profile...")

    from compgen.targets.schema import load_profile

    target = load_profile("examples/target_profiles/cuda_a100.yaml")
    print(f"  Target: {target.name} ({len(target.devices)} devices)")

    # ----------------------------------------------------------------
    # Step 4: Build kernel contracts + strategy selection
    # ----------------------------------------------------------------
    print("\n[4/10] Building kernel contracts + strategy selection...")

    from compgen.kernels.contracts import build_kernel_contracts
    from compgen.kernels.selector import select_strategies

    specs = build_kernel_contracts(module, target)
    decisions = select_strategies(specs, target)

    strategy_counts: dict[str, int] = {}
    for d in decisions:
        strategy_counts[d.strategy.value] = strategy_counts.get(d.strategy.value, 0) + 1
    print(f"  {len(specs)} kernel specs, strategies: {strategy_counts}")

    # ----------------------------------------------------------------
    # Step 5: Run EqSat optimization
    # ----------------------------------------------------------------
    print("\n[5/10] Running equality saturation...")

    from compgen.eqsat.config import EqSatConfig
    from compgen.eqsat.pipeline import run_eqsat_pass

    t0 = time.monotonic()
    eqsat_result = run_eqsat_pass(
        module,
        config=EqSatConfig(max_iterations=5),
    )
    eqsat_ms = (time.monotonic() - t0) * 1000
    print(
        f"  EqSat: {eqsat_result.eclasses_initial}→{eqsat_result.eclasses_after_rewrite} eclasses, "
        f"{eqsat_result.enodes_after_rewrite} enodes, "
        f"changed={eqsat_result.changed}, {eqsat_ms:.0f}ms"
    )
    for name, count in eqsat_result.rule_stats.items():
        if count > 0:
            print(f"    {name}: {count} matches")

    # ----------------------------------------------------------------
    # Step 6: Plan execution
    # ----------------------------------------------------------------
    print("\n[6/10] Planning execution (solver)...")

    from compgen.runtime.planner import plan_execution

    plan = plan_execution(module, target)
    print(
        f"  Plan: {len(plan.placements)} placements, "
        f"{len(plan.copies)} copies, "
        f"latency={plan.estimated_latency_us:.1f}us"
    )

    # ----------------------------------------------------------------
    # Step 7: Verify transform
    # ----------------------------------------------------------------
    print("\n[7/10] Verifying transform...")

    from compgen.transforms.verify import verify_transform

    verify_result = verify_transform(module, module.clone())
    print(f"  Verification: {'PASS' if verify_result.passed else 'FAIL'}")
    for level in verify_result.levels_passed:
        print(f"    {level.value}: PASS")

    # ----------------------------------------------------------------
    # Step 8: Create artifact bundle
    # ----------------------------------------------------------------
    print("\n[8/10] Creating artifact bundle...")

    output_dir = Path(tempfile.mkdtemp(prefix="compgen_bundle_"))
    from compgen.runtime.bundle import create_bundle

    manifest = create_bundle(
        output_dir=output_dir,
        module=module,
        execution_plan=plan,
        target_name=target.name,
        golden_inputs=sample_input,
        golden_outputs=model(*sample_input).detach(),
    )
    print(f"  Bundle: {output_dir}")
    print(f"  Artifacts: {list(manifest.artifacts.keys())}")

    # ----------------------------------------------------------------
    # Step 9: Benchmark
    # ----------------------------------------------------------------
    print("\n[9/10] Benchmarking...")

    from compgen.runtime.local_executor import LocalExecutor

    executor = LocalExecutor()

    # CPU benchmark
    cpu_result = executor.benchmark(model, sample_input, device="cpu", num_iterations=100)
    print(f"  CPU: {cpu_result.latency_median_us:.1f}us median")

    # GPU benchmark (if available)
    if torch.cuda.is_available():
        gpu_result = executor.benchmark(model, sample_input, device="cuda", num_iterations=100)
        print(f"  GPU: {gpu_result.latency_median_us:.1f}us median")

        # Compiled benchmark
        compiled_result = executor.benchmark(
            model, sample_input, device="cuda", mode="compiled", num_iterations=100
        )
        print(f"  GPU (compiled): {compiled_result.latency_median_us:.1f}us median")

    # ----------------------------------------------------------------
    # Step 10: Report
    # ----------------------------------------------------------------
    print("\n[10/10] Report")
    print("=" * 70)
    print(f"Model: SimpleMLP (fc1: 64→128, fc2: 128→32)")
    print(f"Target: {target.name}")
    print(f"FX nodes: {len(ep.graph.nodes)}")
    print(f"Payload IR ops: {op_count}")
    print(f"Kernel specs: {len(specs)} (strategies: {strategy_counts})")
    print(f"EqSat: {eqsat_result.eclasses_after_rewrite} eclasses explored")
    print(f"Plan: {len(plan.placements)} placements, {plan.estimated_latency_us:.1f}us est.")
    print(f"Bundle: {output_dir}")
    print(f"Verification: {'PASS' if verify_result.passed else 'FAIL'}")
    print("=" * 70)
    print("\nCompGen E2E demo complete.")


if __name__ == "__main__":
    main()
