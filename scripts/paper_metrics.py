#!/usr/bin/env python3
"""Paper metrics collection for CompGen MLSys submission.

Runs 7 experiments across 11+ models and 5 target families, collecting
structured JSON, markdown tables, and CSV for plots.

Usage:
    uv run python scripts/paper_metrics.py
    uv run python scripts/paper_metrics.py --experiments 1,2,3
    uv run python scripts/paper_metrics.py --models simple_mlp,llama31_decoder_block
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
import traceback

# Ensure repo root is on sys.path so `benchmarks.*` imports work
_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import structlog
import torch
import torch.nn as nn

log = structlog.get_logger()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_MODELS = [
    "simple_mlp",
    "transformer_block",
    "llama31_decoder_block",
    "llama31_8b_slice",
    "llama4_moe_router_expert_block",
    "dlrmv3_ranking_block",
    "mamba_block",
    "convnext_stage",
    "matmul_bias_gelu",
    "layernorm_chain",
    "softmax_elemwise",
]

TARGETGEN_SPECS = {
    "gpu_simt": "tests/targetgen/exemplars/test_gpu_simt.yaml",
    "rvv_cpu": "tests/targetgen/exemplars/test_rvv_cpu.yaml",
    "rocc_accel": "tests/targetgen/exemplars/test_rocc_accel.yaml",
    "matrix_ext": "tests/targetgen/exemplars/test_matrix_ext.yaml",
    "npu_text_isa": "tests/targetgen/exemplars/test_npu_text_isa.yaml",
}

ABLATION_CONFIGS = [
    {"ablation": "full", "label": "Full Pipeline"},
    {"ablation": "no_eqsat", "label": "No EqSat"},
    {"ablation": "no_verification", "label": "No Verification"},
    {"ablation": "no_solver", "label": "No Solver (Greedy)"},
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CompGen paper metrics")
    p.add_argument("--experiments", default="1,2,3,4,5,6,7",
                   help="Comma-separated experiment numbers")
    p.add_argument("--models", default="",
                   help="Comma-separated model names (default: all)")
    p.add_argument("--output-dir", default="artifacts/paper",
                   help="Output directory")
    p.add_argument("--iterations", type=int, default=50,
                   help="Benchmark iterations")
    p.add_argument("--agentic-budget", type=int, default=3,
                   help="Agentic loop iteration budget")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_model(name: str) -> tuple[nn.Module, tuple[Any, ...]]:
    """Load model + inputs from workloads registry."""
    from benchmarks.workloads import get_loader
    return get_loader(name)()


def run_safe(label: str, fn: Any, errors: dict[str, list[str]]) -> Any:
    """Run fn(), catch exceptions, return None on error."""
    try:
        return fn()
    except Exception as exc:
        tb = traceback.format_exc()
        errors.setdefault(label, []).append(f"{exc}\n{tb}")
        print(f"    [FAIL] {label}: {exc}")
        return None


def make_record(model_name: str, target_name: str = "cuda-a100", **kwargs: Any) -> Any:
    """Create a fresh RunRecord."""
    from benchmarks.record import RunRecord
    return RunRecord(model_name=model_name, target_name=target_name, **kwargs)


# ---------------------------------------------------------------------------
# Experiment 1: Pipeline Coverage
# ---------------------------------------------------------------------------

def run_exp1(models: list[str], target_path: str, out: Path, errors: dict[str, list[str]]) -> list[Any]:
    """Full pipeline coverage on each model."""
    from benchmarks.collector import collect_capture_metrics, collect_eqsat_metrics, collect_ir_metrics
    from compgen.capture.torch_export import capture_model
    from compgen.eqsat.config import EqSatConfig
    from compgen.eqsat.pipeline import run_eqsat_pass
    from compgen.ir.payload.import_fx import fx_to_xdsl
    from compgen.kernels.contracts import build_kernel_contracts
    from compgen.kernels.selector import select_strategies
    from compgen.runtime.planner import plan_execution
    from compgen.targets.schema import load_profile
    from compgen.transforms.verify import verify_transform

    target = load_profile(target_path)
    records = []

    for name in models:
        print(f"  [{name}]")
        rec = make_record(name)

        def _run(name=name, rec=rec) -> Any:
            model, inputs = load_model(name)
            model = model.eval()

            t0 = time.perf_counter()

            # Capture
            ep = capture_model(model, inputs)
            rec.capture = collect_capture_metrics(
                export_success=True,
                total_fx_nodes=len(ep.graph.nodes),
            )

            # IR
            module, diag = fx_to_xdsl(ep)
            rec.ir = collect_ir_metrics(module)
            original = module.clone()

            # EqSat
            t_eq = time.perf_counter()
            eqsat_result = run_eqsat_pass(module, config=EqSatConfig(max_iterations=5))
            rec.eqsat = collect_eqsat_metrics(eqsat_result, (time.perf_counter() - t_eq) * 1000)

            # Planning
            plan = plan_execution(module, target)
            rec.solver.placement_feasible = True
            rec.solver.schedule_makespan_us = plan.estimated_latency_us or 0.0
            rec.solver.copy_ops_count = len(plan.copies)

            # Kernel contracts
            specs = build_kernel_contracts(module, target)
            decisions = select_strategies(specs, target)
            rec.kernels.total_kernel_specs = len(specs)
            rec.kernels.strategy_histogram = {}
            for d in decisions:
                s = d.strategy.value
                rec.kernels.strategy_histogram[s] = rec.kernels.strategy_histogram.get(s, 0) + 1

            # Verification
            vr = verify_transform(module, original)
            rec.verification.structural_pass = vr.passed
            rec.verification.overall_status = "pass" if vr.passed else "fail"

            rec.total_compile_time_ms = (time.perf_counter() - t0) * 1000
            rec.status = "pass"
            return rec

        result = run_safe(f"exp1/{name}", _run, errors)
        if result is None:
            rec.status = "fail"
            rec.errors.append(errors.get(f"exp1/{name}", ["unknown"])[0].split("\n")[0])
        records.append(rec)

    return records


# ---------------------------------------------------------------------------
# Experiment 2: Performance Benchmarks
# ---------------------------------------------------------------------------

def run_exp2(models: list[str], num_iter: int, out: Path, errors: dict[str, list[str]]) -> list[Any]:
    """CPU/GPU benchmarks + numeric verification."""
    from compgen.runtime.local_executor import LocalExecutor
    from compgen.verify.harness import verify_callable_against_reference

    executor = LocalExecutor()
    has_gpu = torch.cuda.is_available()
    records = []

    for name in models:
        print(f"  [{name}]")
        rec = make_record(name)

        def _run(name=name, rec=rec) -> Any:
            model, inputs = load_model(name)
            model = model.eval()

            # Numeric verification FIRST (while model is on CPU)
            import copy
            with torch.no_grad():
                vr = verify_callable_against_reference(
                    name=f"verify_{name}",
                    ref_fn=lambda m=model, i=inputs: m(*i),
                    got_fn=lambda m=model, i=inputs: torch.compile(m, backend="eager")(*i),
                    out_dir=out / "exp2_verify" / name,
                )
            rec.verification.differential_pass = vr.passed
            rec.verification.differential_max_error = vr.comparisons[0].max_abs_error if vr.comparisons else 0.0
            rec.verification.overall_status = "pass" if vr.passed else "fail"

            # CPU benchmark
            cpu = executor.benchmark(model, inputs, device="cpu", num_iterations=num_iter)
            rec.baselines.eager_cpu_latency_us = cpu.latency_median_us
            rec.performance.latency_median_us = cpu.latency_median_us
            rec.performance.throughput_samples_per_sec = cpu.throughput_samples_per_sec

            if has_gpu:
                # Use fresh copies for GPU (benchmark moves model in-place)
                gpu_model = copy.deepcopy(model)
                gpu = executor.benchmark(gpu_model, inputs, device="cuda", num_iterations=num_iter)
                rec.baselines.eager_gpu_latency_us = gpu.latency_median_us
                rec.performance.peak_memory_bytes = gpu.peak_memory_bytes
                del gpu_model

                compiled_model = copy.deepcopy(model)
                compiled = executor.benchmark(compiled_model, inputs, device="cuda", mode="compiled", num_iterations=num_iter)
                rec.baselines.compiled_gpu_latency_us = compiled.latency_median_us
                del compiled_model

                if gpu.latency_median_us > 0:
                    rec.baselines.speedup_vs_eager_gpu = gpu.latency_median_us / compiled.latency_median_us

            rec.status = "pass"
            return rec

        result = run_safe(f"exp2/{name}", _run, errors)
        if result is None:
            rec.status = "fail"
            rec.errors.append(errors.get(f"exp2/{name}", ["unknown"])[0].split("\n")[0])
        records.append(rec)

        # Memory cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return records


# ---------------------------------------------------------------------------
# Experiment 3: Target Generation
# ---------------------------------------------------------------------------

@dataclass
class TargetGenRecord:
    family: str = ""
    spec_path: str = ""
    confidence: float = 0.0
    stages: int = 0
    verification_tests: int = 0
    kernel_backend: str = ""
    needs_accel_dialect: bool = False
    pipeline_pass: bool = False
    gen_time_ms: float = 0.0
    error: str = ""


def run_exp3(out: Path, errors: dict[str, list[str]]) -> list[TargetGenRecord]:
    """Target generation for 5 families."""
    from compgen.targetgen.generate import generate_target

    records = []
    for family_name, spec_path in TARGETGEN_SPECS.items():
        print(f"  [{family_name}]")
        rec = TargetGenRecord(family=family_name, spec_path=spec_path)

        def _run(rec=rec, spec_path=spec_path, family_name=family_name) -> TargetGenRecord:
            gen_dir = out / "exp3_targetgen" / family_name
            t0 = time.perf_counter()
            result = generate_target(spec_path=spec_path, output_dir=str(gen_dir))
            rec.gen_time_ms = (time.perf_counter() - t0) * 1000

            rec.confidence = result.classification.confidence
            rec.stages = len(result.dialect_stack.stages)
            rec.verification_tests = len(result.verification_manifest.tests)
            rec.kernel_backend = result.plan.kernel_backend
            rec.needs_accel_dialect = result.plan.needs_accel_dialect

            # Run pipeline on test IR
            from compgen.capture.torch_export import capture_model
            from compgen.ir.payload.import_fx import fx_to_xdsl
            from compgen.stages.registry import StageRegistry

            class TinyModel(nn.Module):
                def __init__(self) -> None:
                    super().__init__()
                    self.fc = nn.Linear(16, 8)

                def forward(self, x: torch.Tensor) -> torch.Tensor:
                    return self.fc(x)

            ep = capture_model(TinyModel(), (torch.randn(2, 16),))
            module, _ = fx_to_xdsl(ep)

            registry = StageRegistry()
            registry.register_target_stack(result.dialect_stack)
            pr = registry.run_pipeline(module, result.profile, result.capabilities)
            rec.pipeline_pass = pr.passed
            return rec

        result = run_safe(f"exp3/{family_name}", _run, errors)
        if result is None:
            rec.error = errors.get(f"exp3/{family_name}", ["unknown"])[0].split("\n")[0]
        records.append(rec)

    return records


# ---------------------------------------------------------------------------
# Experiment 4: Recipe IR Pipeline
# ---------------------------------------------------------------------------

def run_exp4(models: list[str], target_path: str, out: Path, errors: dict[str, list[str]]) -> list[Any]:
    """Recipe IR seed → validate → lower → guard synthesis."""
    from benchmarks.collector import collect_recipe_metrics
    from compgen.capture.torch_export import capture_model
    from compgen.ir.payload.import_fx import fx_to_xdsl
    from compgen.ir.recipe.lower import lower_recipe
    from compgen.ir.recipe.seed import generate_seed_recipe
    from compgen.ir.recipe.validate import validate_recipe_module
    from compgen.targets.schema import load_profile

    target = load_profile(target_path)
    records = []

    for name in models:
        print(f"  [{name}]")
        rec = make_record(name)

        def _run(name=name, rec=rec) -> Any:
            model, inputs = load_model(name)
            model = model.eval()

            ep = capture_model(model, inputs)
            module, _ = fx_to_xdsl(ep)

            # Seed recipe
            t0 = time.perf_counter()
            recipe = generate_seed_recipe(module, target_profile=target, objective="latency")
            rec.recipe.seed_generation_time_ms = (time.perf_counter() - t0) * 1000

            # Collect recipe metrics
            recipe_metrics = collect_recipe_metrics(recipe)
            rec.recipe.total_recipe_ops = recipe_metrics.total_recipe_ops
            rec.recipe.scope_ops = recipe_metrics.scope_ops
            rec.recipe.fact_ops = recipe_metrics.fact_ops
            rec.recipe.candidate_ops = recipe_metrics.candidate_ops
            rec.recipe.verify_ops = recipe_metrics.verify_ops

            # Validate
            validation = validate_recipe_module(recipe)
            rec.recipe.validation_passed = validation.valid
            rec.recipe.validation_errors = len(validation.errors)

            # Lower
            lowering = lower_recipe(recipe, target_class="gpu")
            rec.recipe.transform_scripts_count = len(lowering.transform_scripts)
            rec.recipe.kernel_jobs_count = len(lowering.kernel_jobs)
            rec.recipe.verification_obligations_count = len(lowering.verification_obligations)
            rec.recipe.lowering_diagnostics = len(lowering.diagnostics)

            rec.status = "pass" if validation.valid else "fail"
            return rec

        result = run_safe(f"exp4/{name}", _run, errors)
        if result is None:
            rec.status = "fail"
        records.append(rec)

    return records


# ---------------------------------------------------------------------------
# Experiment 5: Multi-Device Planning
# ---------------------------------------------------------------------------

def run_exp5(models: list[str], target_path: str, multi_path: str,
             out: Path, errors: dict[str, list[str]]) -> list[Any]:
    """Compare single-device vs multi-device planning."""
    from compgen.capture.torch_export import capture_model
    from compgen.ir.payload.import_fx import fx_to_xdsl
    from compgen.runtime.planner import plan_execution
    from compgen.targets.schema import load_profile

    single_target = load_profile(target_path)
    multi_target = load_profile(multi_path)
    records = []

    for name in models:
        for label, tgt in [("single_gpu", single_target), ("heterogeneous", multi_target)]:
            print(f"  [{name}/{label}]")
            rec = make_record(name, target_name=tgt.name)
            rec.config["planning"] = label

            def _run(name=name, rec=rec, tgt=tgt) -> Any:
                model, inputs = load_model(name)
                model = model.eval()
                ep = capture_model(model, inputs)
                module, _ = fx_to_xdsl(ep)

                t0 = time.perf_counter()
                plan = plan_execution(module, tgt)
                solver_ms = (time.perf_counter() - t0) * 1000

                rec.solver.placement_feasible = True
                rec.solver.schedule_makespan_us = plan.estimated_latency_us or 0.0
                rec.solver.copy_ops_count = len(plan.copies)
                rec.solver.placement_time_ms = solver_ms
                rec.solver.node_assignments = dict(plan.node_assignments) if plan.node_assignments else {}
                rec.status = "pass"
                return rec

            result = run_safe(f"exp5/{name}/{label}", _run, errors)
            if result is None:
                rec.status = "fail"
            records.append(rec)

    return records


# ---------------------------------------------------------------------------
# Experiment 6: Agentic Loop
# ---------------------------------------------------------------------------

def run_exp6(models: list[str], target_path: str, budget: int,
             out: Path, errors: dict[str, list[str]]) -> list[Any]:
    """Agentic compilation loop on select models."""
    from compgen.agent.compilation_loop import AgenticCompilationLoop
    from compgen.agent.env import CompilerEnv
    from compgen.capture.torch_export import capture_model
    from compgen.ir.payload.import_fx import fx_to_xdsl
    from compgen.targets.schema import load_profile

    target = load_profile(target_path)
    records = []

    # Try real LLM, fall back to mock
    try:
        from compgen.llm._env import resolve_api_key
        api_key = resolve_api_key("GOOGLE_API_KEY", "GEMINI_API_KEY", "GEMMINI_API")
        if api_key:
            from compgen.llm.gemini_client import GeminiClient
            llm = GeminiClient(model="gemini-2.5-pro", api_key=api_key)
            llm_label = "gemini-2.5-pro"
        else:
            from compgen.llm.mock_client import MockLLMClient
            llm = MockLLMClient(strict=False)
            llm_label = "mock"
    except Exception:
        from compgen.llm.mock_client import MockLLMClient
        llm = MockLLMClient(strict=False)
        llm_label = "mock"

    for name in models:
        print(f"  [{name}] (llm={llm_label}, budget={budget})")
        rec = make_record(name)

        def _run(name=name, rec=rec) -> Any:
            model, inputs = load_model(name)
            model = model.eval()
            ep = capture_model(model, inputs)
            module, _ = fx_to_xdsl(ep)

            env = CompilerEnv()
            env.reset(module=module, target=target, objective="latency", budget=budget,
                      exported_program=ep)

            loop = AgenticCompilationLoop(llm_client=llm, env=env, budget=budget)
            result = loop.run(target)

            rec.agentic.iterations_run = result.iterations_run
            rec.agentic.initial_cost_us = result.initial_cost_us
            rec.agentic.final_cost_us = result.final_cost_us
            rec.agentic.total_improvement_pct = result.total_improvement_pct
            rec.agentic.iterations_improved = result.iterations_improved
            rec.agentic.iteration_costs = [h.cost_after_us for h in result.history]
            rec.agentic.iteration_actions = [h.action_type for h in result.history]
            rec.llm.model_id = llm_label
            rec.status = "pass"
            return rec

        result = run_safe(f"exp6/{name}", _run, errors)
        if result is None:
            rec.status = "fail"
        records.append(rec)

    return records


# ---------------------------------------------------------------------------
# Experiment 7: Ablation Studies
# ---------------------------------------------------------------------------

def run_exp7(models: list[str], target_path: str, out: Path, errors: dict[str, list[str]]) -> list[Any]:
    """Ablation: full vs no-eqsat vs no-verify vs no-solver."""
    from compgen.capture.torch_export import capture_model
    from compgen.eqsat.config import EqSatConfig
    from compgen.eqsat.pipeline import run_eqsat_pass
    from compgen.ir.payload.import_fx import fx_to_xdsl
    from compgen.runtime.planner import plan_execution
    from compgen.targets.schema import load_profile
    from compgen.transforms.verify import verify_transform

    target = load_profile(target_path)
    records = []

    for name in models:
        for cfg in ABLATION_CONFIGS:
            label = cfg["ablation"]
            print(f"  [{name}/{label}]")
            rec = make_record(name)
            rec.config["ablation"] = label

            def _run(name=name, rec=rec, label=label) -> Any:
                model, inputs = load_model(name)
                model = model.eval()

                t0 = time.perf_counter()
                ep = capture_model(model, inputs)
                module, _ = fx_to_xdsl(ep)
                original = module.clone()

                # EqSat (skip if ablated)
                if label != "no_eqsat":
                    eqsat_result = run_eqsat_pass(module, config=EqSatConfig(max_iterations=5))
                    rec.eqsat.changed = eqsat_result.changed

                # Planning (skip if ablated — use trivial plan)
                if label != "no_solver":
                    plan = plan_execution(module, target)
                    rec.solver.placement_feasible = True
                    rec.solver.schedule_makespan_us = plan.estimated_latency_us or 0.0
                else:
                    rec.solver.placement_feasible = True
                    rec.solver.schedule_makespan_us = 0.0

                # Verification (skip if ablated)
                if label != "no_verification":
                    vr = verify_transform(module, original)
                    rec.verification.structural_pass = vr.passed
                    rec.verification.overall_status = "pass" if vr.passed else "fail"
                else:
                    rec.verification.overall_status = "skipped"

                rec.total_compile_time_ms = (time.perf_counter() - t0) * 1000
                rec.status = "pass"
                return rec

            result = run_safe(f"exp7/{name}/{label}", _run, errors)
            if result is None:
                rec.status = "fail"
            records.append(rec)

    return records


# ---------------------------------------------------------------------------
# Output: Tables + CSV + Summary
# ---------------------------------------------------------------------------

def write_coverage_table(records: list[Any], path: Path) -> None:
    """Write pipeline coverage table (Exp 1)."""
    lines = ["| Model | FX Nodes | IR Ops | EqSat Changed | Placements | Latency (us) | Kernels | Strategies | Verify | Status |",
             "|-------|----------|--------|---------------|------------|-------------|---------|------------|--------|--------|"]
    for r in records:
        strats = ", ".join(f"{k}:{v}" for k, v in sorted(r.kernels.strategy_histogram.items()))
        lines.append(
            f"| {r.model_name} | {r.capture.total_fx_nodes} | {r.ir.total_ops} | "
            f"{r.eqsat.changed} | {r.solver.schedule_makespan_us:.0f} | "
            f"{r.solver.schedule_makespan_us:.1f} | {r.kernels.total_kernel_specs} | "
            f"{strats} | {r.verification.overall_status} | {r.status} |"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_perf_table(records: list[Any], path: Path) -> None:
    """Write performance table (Exp 2)."""
    lines = ["| Model | CPU (us) | GPU (us) | Compiled (us) | Speedup | Max Abs Error | Mem (MB) | Status |",
             "|-------|----------|----------|---------------|---------|---------------|----------|--------|"]
    for r in records:
        mem_mb = r.performance.peak_memory_bytes / (1024 * 1024) if r.performance.peak_memory_bytes else 0
        lines.append(
            f"| {r.model_name} | {r.baselines.eager_cpu_latency_us:.1f} | "
            f"{r.baselines.eager_gpu_latency_us:.1f} | {r.baselines.compiled_gpu_latency_us:.1f} | "
            f"{r.baselines.speedup_vs_eager_gpu:.2f}x | {r.verification.differential_max_error:.2e} | "
            f"{mem_mb:.1f} | {r.status} |"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_targetgen_table(records: list[TargetGenRecord], path: Path) -> None:
    """Write target generation table (Exp 3)."""
    lines = ["| Family | Confidence | Stages | Backend | Accel Dialect | Verif Tests | Pipeline | Time (ms) |",
             "|--------|-----------|--------|---------|---------------|-------------|----------|-----------|"]
    for r in records:
        lines.append(
            f"| {r.family} | {r.confidence:.0%} | {r.stages} | {r.kernel_backend} | "
            f"{r.needs_accel_dialect} | {r.verification_tests} | "
            f"{'PASS' if r.pipeline_pass else 'FAIL'} | {r.gen_time_ms:.0f} |"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_recipe_table(records: list[Any], path: Path) -> None:
    """Write Recipe IR table (Exp 4)."""
    lines = ["| Model | Recipe Ops | Scopes | Facts | Candidates | Valid | Transforms | Kernel Jobs | Verif Obligs |",
             "|-------|-----------|--------|-------|------------|-------|------------|-------------|-------------|"]
    for r in records:
        lines.append(
            f"| {r.model_name} | {r.recipe.total_recipe_ops} | {r.recipe.scope_ops} | "
            f"{r.recipe.fact_ops} | {r.recipe.candidate_ops} | {r.recipe.validation_passed} | "
            f"{r.recipe.transform_scripts_count} | {r.recipe.kernel_jobs_count} | "
            f"{r.recipe.verification_obligations_count} |"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_multidevice_table(records: list[Any], path: Path) -> None:
    """Write multi-device table (Exp 5)."""
    lines = ["| Model | Planning | Latency (us) | Copies | Solver (ms) | Status |",
             "|-------|----------|-------------|--------|-------------|--------|"]
    for r in records:
        lines.append(
            f"| {r.model_name} | {r.config.get('planning', '?')} | "
            f"{r.solver.schedule_makespan_us:.1f} | {r.solver.copy_ops_count} | "
            f"{r.solver.placement_time_ms:.1f} | {r.status} |"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_agentic_table(records: list[Any], path: Path) -> None:
    """Write agentic loop table (Exp 6)."""
    lines = ["| Model | LLM | Iterations | Initial (us) | Final (us) | Improvement | Status |",
             "|-------|-----|-----------|-------------|-----------|-------------|--------|"]
    for r in records:
        lines.append(
            f"| {r.model_name} | {r.llm.model_id} | {r.agentic.iterations_run} | "
            f"{r.agentic.initial_cost_us:.1f} | {r.agentic.final_cost_us:.1f} | "
            f"{r.agentic.total_improvement_pct:+.1f}% | {r.status} |"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_ablation_table(records: list[Any], path: Path) -> None:
    """Write ablation table (Exp 7)."""
    lines = ["| Model | Ablation | Compile (ms) | EqSat Changed | Verify | Status |",
             "|-------|----------|-------------|---------------|--------|--------|"]
    for r in records:
        lines.append(
            f"| {r.model_name} | {r.config.get('ablation', '?')} | "
            f"{r.total_compile_time_ms:.1f} | {r.eqsat.changed} | "
            f"{r.verification.overall_status} | {r.status} |"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_csv(records: list[Any], path: Path, extra_fields: list[str] | None = None) -> None:
    """Write records to CSV."""
    if not records:
        return
    fields = ["model_name", "target_name", "status", "total_compile_time_ms"]
    fields += extra_fields or []
    lines = [",".join(fields)]
    for r in records:
        vals = []
        for f in fields:
            if "." in f:
                obj, attr = f.split(".", 1)
                vals.append(str(getattr(getattr(r, obj, r), attr, "")))
            else:
                vals.append(str(getattr(r, f, "")))
        lines.append(",".join(vals))
    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    experiments = set(int(x) for x in args.experiments.split(","))
    models = [m.strip() for m in args.models.split(",") if m.strip()] or DEFAULT_MODELS
    out = Path(args.output_dir)
    target_path = "examples/target_profiles/cuda_a100.yaml"
    multi_path = "examples/target_profiles/multi_device.yaml"

    # Create output dirs
    for subdir in ["raw", "tables", "csv", "plots"]:
        (out / subdir).mkdir(parents=True, exist_ok=True)

    errors: dict[str, list[str]] = {}
    all_results: dict[str, Any] = {}

    print("=" * 70)
    print(f"CompGen Paper Metrics — {datetime.now(UTC).isoformat()}")
    print(f"Models: {len(models)} | Experiments: {sorted(experiments)}")
    print(f"GPU: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'none'}")
    print(f"Output: {out}")
    print("=" * 70)

    # -- Experiment 1: Pipeline Coverage --
    if 1 in experiments:
        print("\n[EXP 1] Pipeline Coverage")
        records = run_exp1(models, target_path, out, errors)
        all_results["exp1"] = records
        write_coverage_table(records, out / "tables" / "table1_coverage.md")
        write_csv(records, out / "csv" / "coverage.csv",
                  ["capture.total_fx_nodes", "ir.total_ops", "eqsat.changed",
                   "kernels.total_kernel_specs", "verification.overall_status"])

    # -- Experiment 2: Performance --
    if 2 in experiments:
        print("\n[EXP 2] Performance Benchmarks")
        records = run_exp2(models, args.iterations, out, errors)
        all_results["exp2"] = records
        write_perf_table(records, out / "tables" / "table2_performance.md")
        write_csv(records, out / "csv" / "performance.csv",
                  ["baselines.eager_cpu_latency_us", "baselines.eager_gpu_latency_us",
                   "baselines.compiled_gpu_latency_us", "baselines.speedup_vs_eager_gpu",
                   "verification.differential_max_error"])

    # -- Experiment 3: Target Generation --
    if 3 in experiments:
        print("\n[EXP 3] Target Generation")
        tg_records = run_exp3(out, errors)
        all_results["exp3"] = tg_records
        write_targetgen_table(tg_records, out / "tables" / "table3_targetgen.md")

    # -- Experiment 4: Recipe IR --
    if 4 in experiments:
        print("\n[EXP 4] Recipe IR Pipeline")
        records = run_exp4(models, target_path, out, errors)
        all_results["exp4"] = records
        write_recipe_table(records, out / "tables" / "table4_recipe.md")
        write_csv(records, out / "csv" / "recipe.csv",
                  ["recipe.total_recipe_ops", "recipe.scope_ops", "recipe.fact_ops",
                   "recipe.candidate_ops", "recipe.validation_passed",
                   "recipe.transform_scripts_count", "recipe.verification_obligations_count"])

    # -- Experiment 5: Multi-Device --
    if 5 in experiments:
        print("\n[EXP 5] Multi-Device Planning")
        records = run_exp5(models, target_path, multi_path, out, errors)
        all_results["exp5"] = records
        write_multidevice_table(records, out / "tables" / "table5_multidevice.md")

    # -- Experiment 6: Agentic Loop --
    if 6 in experiments:
        agentic_models = [m for m in ["simple_mlp", "transformer_block", "llama31_decoder_block"] if m in models]
        print(f"\n[EXP 6] Agentic Loop ({len(agentic_models)} models)")
        records = run_exp6(agentic_models, target_path, args.agentic_budget, out, errors)
        all_results["exp6"] = records
        write_agentic_table(records, out / "tables" / "table6_agentic.md")

    # -- Experiment 7: Ablation --
    if 7 in experiments:
        ablation_models = models[:5]  # Top 5 for ablation
        print(f"\n[EXP 7] Ablation Studies ({len(ablation_models)} models x {len(ABLATION_CONFIGS)} configs)")
        records = run_exp7(ablation_models, target_path, out, errors)
        all_results["exp7"] = records
        write_ablation_table(records, out / "tables" / "table7_ablation.md")

    # -- Summary --
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    total_runs = sum(len(v) if isinstance(v, list) else 0 for v in all_results.values())
    total_pass = sum(
        1 for v in all_results.values() if isinstance(v, list)
        for r in v if getattr(r, "status", getattr(r, "pipeline_pass", None)) in ("pass", True)
    )
    total_fail = total_runs - total_pass

    summary = {
        "timestamp": datetime.now(UTC).isoformat(),
        "models": models,
        "experiments": sorted(experiments),
        "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else "none",
        "total_runs": total_runs,
        "passed": total_pass,
        "failed": total_fail,
        "errors": {k: [e.split("\n")[0] for e in v] for k, v in errors.items()},
    }

    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    # Combined markdown
    md_parts = [f"# CompGen Paper Metrics\n\nGenerated: {summary['timestamp']}\n"]
    md_parts.append(f"**Models:** {len(models)} | **Runs:** {total_runs} | **Pass:** {total_pass} | **Fail:** {total_fail}\n")
    for i in sorted(experiments):
        table_path = out / "tables" / f"table{i}_{'coverage performance targetgen recipe multidevice agentic ablation'.split()[i-1]}.md"
        if table_path.exists():
            md_parts.append(f"\n## Experiment {i}\n\n{table_path.read_text()}\n")
    if errors:
        md_parts.append("\n## Errors\n")
        for k, v in errors.items():
            md_parts.append(f"- **{k}:** {v[0].split(chr(10))[0]}")
    (out / "summary.md").write_text("\n".join(md_parts), encoding="utf-8")

    print(f"\nTotal: {total_runs} runs, {total_pass} passed, {total_fail} failed")
    print(f"Tables: {out / 'tables'}")
    print(f"CSV: {out / 'csv'}")
    print(f"Summary: {out / 'summary.md'}")

    if errors:
        print(f"\n{len(errors)} errors:")
        for k, v in errors.items():
            print(f"  {k}: {v[0].split(chr(10))[0]}")

    sys.exit(0 if total_fail == 0 else 1)


if __name__ == "__main__":
    main()
