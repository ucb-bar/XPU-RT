#!/usr/bin/env python3
"""TargetGen demo: hardware engineer workflow.

Shows the complete flow:
  1. Load a hardware spec YAML
  2. Classify the target
  3. Generate a support plan
  4. Generate the compilation pipeline
  5. Run the pipeline on a sample model
  6. Inspect the output
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

# Exemplar directory
EXEMPLAR_DIR = Path(__file__).parent.parent / "tests" / "targetgen" / "exemplars"


def main() -> None:
    """Run the targetgen demo."""
    print("=" * 70)
    print("CompGen TargetGen Demo: HW Spec → Generated Compiler Pipeline")
    print("=" * 70)

    # Pick an exemplar
    exemplars = sorted(EXEMPLAR_DIR.glob("*.yaml"))
    if not exemplars:
        print("No exemplar YAMLs found!")
        return

    for spec_path in exemplars:
        print(f"\n{'─' * 60}")
        print(f"Target: {spec_path.stem}")
        print(f"{'─' * 60}")

        # 1. Load + validate
        from compgen.targetgen.load import load_hardware_spec
        from compgen.targetgen.validate_spec import validate_hardware_spec

        spec = load_hardware_spec(spec_path)
        validation = validate_hardware_spec(spec)
        print(f"  Spec: {spec.name} (v{spec.schema_version})")
        print(f"  Valid: {validation.valid} ({len(validation.warnings)} warnings)")

        # 2. Classify
        from compgen.targetgen.classify import classify_hardware

        classification = classify_hardware(spec)
        print(f"  Family: {classification.family.value}")
        print(f"  Integration: {classification.integration_style.value}")
        print(f"  Lowering: {classification.lowering_surface.value}")
        print(f"  Confidence: {classification.confidence:.0%}")

        # 3. Plan
        from compgen.targetgen.plan import generate_support_plan

        plan = generate_support_plan(spec, classification)
        stage_names = [s.stage_name for s in plan.required_stages]
        print(f"  Stages ({len(stage_names)}): {' → '.join(stage_names)}")
        print(f"  Backend: {plan.kernel_backend}")
        if plan.llvm_patches_needed:
            print(f"  LLVM patches: NEEDED")
        if plan.needs_accel_dialect:
            print(f"  Accel dialect: NEEDED")

        # 4. Generate
        import tempfile

        from compgen.targetgen.generate import generate_target

        output_dir = Path(tempfile.mkdtemp(prefix=f"compgen_{spec.name}_"))
        gen = generate_target(spec_path, output_dir)
        print(f"  Output: {output_dir}")
        print(f"  Stack depth: {len(gen.dialect_stack.stages)} stages")
        print(f"  Verification tests: {len(gen.verification_manifest.tests)}")
        print(f"  Target maturity: {gen.verification_manifest.maturity.name}")

        # 5. Run pipeline on sample IR
        from xdsl.dialects import arith, func
        from xdsl.dialects.builtin import IndexType, ModuleOp
        from xdsl.ir import Block, Region

        from compgen.stages.registry import StageRegistry

        idx = IndexType()
        block = Block(arg_types=[idx, idx])
        a, b = block.args
        add = arith.AddiOp(a, b)
        block.add_op(add)
        block.add_op(func.ReturnOp(add.result))
        module = ModuleOp([func.FuncOp("test", ([idx, idx], [idx]), Region([block]))])

        registry = StageRegistry()
        gen.dialect_stack.target_name = gen.profile.name
        registry.register_target_stack(gen.dialect_stack)
        result = registry.run_pipeline(module, gen.profile, gen.capabilities)
        print(f"  Pipeline: {'PASS' if result.passed else 'FAIL'} ({result.stages_run} stages)")

    print(f"\n{'=' * 70}")
    print("TargetGen demo complete.")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
