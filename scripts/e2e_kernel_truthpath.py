#!/usr/bin/env python3
"""End-to-end kernel truth path: matmul_bias_gelu through Triton templates.

Pipeline
--------
1. Load ``matmul_bias_gelu`` workload from ``benchmarks/workloads.py``.
2. Capture via ``torch.export``.
3. Build kernel contracts from the IR.
4. Locate the matmul spec.
5. Run ``TritonTemplateProvider.search()``.
6. Validate with ``KernelValidator``.
7. Benchmark on GPU (when available).
8. Print gate report.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

# Ensure project root is on sys.path so local ``benchmarks`` package
# is found before the stale xDSL benchmarks installed in site-packages.
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import structlog
import torch

log = structlog.get_logger()


# ---------------------------------------------------------------------------
# Gate report
# ---------------------------------------------------------------------------

@dataclass
class Gate:
    """Single pass/fail gate."""

    name: str
    passed: bool
    detail: str = ""


def _print_report(gates: list[Gate]) -> None:
    width = 72
    print()
    print("=" * width)
    print("  Kernel Truth-Path Gate Report: matmul_bias_gelu")
    print("=" * width)
    for g in gates:
        status = "PASS" if g.passed else "FAIL"
        line = f"  [{status}] {g.name}"
        if g.detail:
            line += f"  -- {g.detail}"
        print(line)
    total = len(gates)
    passed = sum(1 for g in gates if g.passed)
    print("-" * width)
    print(f"  {passed}/{total} gates passed")
    print("=" * width)
    print()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> int:
    gates: list[Gate] = []

    # ------------------------------------------------------------------
    # 1. Load workload
    # ------------------------------------------------------------------
    print("\n[1/8] Loading matmul_bias_gelu workload...")
    from benchmarks.workloads import get_loader

    loader = get_loader("matmul_bias_gelu")
    model, sample_inputs = loader()
    model.eval()
    gates.append(Gate("workload_load", True, f"model={type(model).__name__}"))
    print(f"  Model: {type(model).__name__}, input shape: {sample_inputs[0].shape}")

    # ------------------------------------------------------------------
    # 2. Capture via torch.export
    # ------------------------------------------------------------------
    print("\n[2/8] Capturing via torch.export...")
    try:
        from compgen.capture.torch_export import capture_model

        ep = capture_model(model, sample_inputs)
        n_nodes = len(ep.graph.nodes)
        gates.append(Gate("torch_export", True, f"{n_nodes} FX nodes"))
        print(f"  Captured: {n_nodes} FX nodes")
    except Exception as exc:
        gates.append(Gate("torch_export", False, str(exc)))
        print(f"  SKIP (capture failed): {exc}")
        ep = None

    # ------------------------------------------------------------------
    # 3. Build kernel contracts
    # ------------------------------------------------------------------
    print("\n[3/8] Building kernel contracts...")
    specs: list = []
    if ep is not None:
        try:
            from compgen.ir.payload.import_fx import fx_to_xdsl
            from compgen.kernels.contracts import build_kernel_contracts
            from compgen.targets.schema import load_profile

            module, _ = fx_to_xdsl(ep)
            target = load_profile("examples/target_profiles/cuda_a100.yaml")
            specs = build_kernel_contracts(module, target)
            gates.append(Gate("kernel_contracts", len(specs) > 0, f"{len(specs)} specs"))
            print(f"  Built {len(specs)} kernel specs")
        except Exception as exc:
            gates.append(Gate("kernel_contracts", False, str(exc)))
            print(f"  SKIP (contracts failed): {exc}")
    else:
        gates.append(Gate("kernel_contracts", False, "no exported program"))

    # ------------------------------------------------------------------
    # 4. Identify matmul spec
    # ------------------------------------------------------------------
    print("\n[4/8] Locating matmul spec...")
    matmul_spec = None
    matmul_keywords = {"matmul", "dot", "mm", "linear", "gemm"}
    for s in specs:
        op_lower = s.contract.op_name.lower()
        if any(kw in op_lower for kw in matmul_keywords):
            matmul_spec = s
            break
    # Fallback: take the highest-FLOP spec (likely the matmul)
    if matmul_spec is None and specs:
        matmul_spec = max(specs, key=lambda s: s.contract.cost.flops)
    if matmul_spec:
        gates.append(Gate("matmul_spec_found", True, matmul_spec.contract.op_name))
        print(f"  Found: {matmul_spec.contract.op_name} (FLOPs={matmul_spec.contract.cost.flops})")
    else:
        gates.append(Gate("matmul_spec_found", False, "no specs available"))
        print("  No matmul spec found -- will use defaults for provider search")

    # ------------------------------------------------------------------
    # 5. Run TritonTemplateProvider
    # ------------------------------------------------------------------
    print("\n[5/8] Running TritonTemplateProvider.search()...")
    from compgen.kernels.provider import KernelContract, SearchBudget
    from compgen.kernels.providers.triton_templates import TritonTemplateProvider, triton_available

    M, K, N = model.m, model.k, model.weight.shape[1]
    contract = KernelContract(
        region_id="matmul_bias_gelu_0",
        op_family="matmul_bias_gelu",
        input_shapes=((M, K), (K, N)),
        output_shapes=((M, N),),
        dtypes=("float32",),
        target_name="cuda_a100",
    )
    provider = TritonTemplateProvider()
    budget = SearchBudget(max_iterations=1, max_time_ms=30_000)

    t0 = time.monotonic()
    result = provider.search(contract, budget)
    search_ms = (time.monotonic() - t0) * 1000

    gates.append(Gate("provider_search", result.found, f"{search_ms:.0f}ms, lang={result.language}"))
    if result.found:
        code_lines = result.kernel_code.count("\n") + 1
        print(f"  Found kernel: {code_lines} lines, language={result.language}")
    else:
        print("  Provider did not find a kernel")

    # ------------------------------------------------------------------
    # 6. Validate with KernelValidator
    # ------------------------------------------------------------------
    print("\n[6/8] Validating kernel...")
    validation_passed = False
    kernel_fn = None

    if result.found and triton_available() and torch.cuda.is_available():
        try:
            from compgen.kernels.providers.triton_templates import _import_kernel_from_source

            kernel_fn = _import_kernel_from_source(result.kernel_code)

            A = torch.randn(M, K, device="cuda", dtype=torch.float32)
            B = torch.randn(K, N, device="cuda", dtype=torch.float32)
            bias = torch.randn(N, device="cuda", dtype=torch.float32)
            ref_out = torch.nn.functional.gelu(A @ B + bias)

            actual = kernel_fn(A, B, bias)
            diff = actual.float() - ref_out.float()
            l2 = float(torch.norm(diff).item())
            max_abs = float(torch.max(torch.abs(diff)).item())
            tol = 1e-2
            validation_passed = max_abs <= tol
            status = "PASS" if validation_passed else "FAIL"
            print(f"    Correctness: {status} (l2={l2:.6f}, max_abs={max_abs:.6f}, tol={tol})")
            gates.append(Gate("validation", validation_passed, f"l2={l2:.6f} max_abs={max_abs:.6f}"))
        except Exception as exc:
            gates.append(Gate("validation", False, str(exc)))
            print(f"  Validation error: {exc}")
    elif not triton_available():
        gates.append(Gate("validation", False, "triton not importable"))
        print("  SKIP: triton not importable -- source code generated but cannot execute")
    elif not torch.cuda.is_available():
        gates.append(Gate("validation", False, "CUDA not available"))
        print("  SKIP: CUDA not available")
    else:
        gates.append(Gate("validation", False, "no kernel to validate"))
        print("  SKIP: no kernel code")

    # ------------------------------------------------------------------
    # 7. Benchmark
    # ------------------------------------------------------------------
    print("\n[7/8] Benchmarking...")
    latency_us = 0.0

    if validation_passed and kernel_fn is not None and torch.cuda.is_available():
        try:
            A = torch.randn(M, K, device="cuda", dtype=torch.float32)
            B = torch.randn(K, N, device="cuda", dtype=torch.float32)
            bias = torch.randn(N, device="cuda", dtype=torch.float32)

            # Warmup
            for _ in range(5):
                kernel_fn(A, B, bias)
            torch.cuda.synchronize()

            # Timed
            start_ev = torch.cuda.Event(enable_timing=True)
            end_ev = torch.cuda.Event(enable_timing=True)
            n_iters = 50
            start_ev.record()
            for _ in range(n_iters):
                kernel_fn(A, B, bias)
            end_ev.record()
            torch.cuda.synchronize()
            latency_us = start_ev.elapsed_time(end_ev) * 1000.0 / n_iters

            gates.append(Gate("benchmark", True, f"{latency_us:.1f} us/iter"))
            print(f"  Latency: {latency_us:.1f} us ({n_iters} iters)")
        except Exception as exc:
            gates.append(Gate("benchmark", False, str(exc)))
            print(f"  Benchmark error: {exc}")
    else:
        reason = "validation failed" if not validation_passed else "CUDA not available"
        gates.append(Gate("benchmark", False, reason))
        print(f"  SKIP: {reason}")

    # ------------------------------------------------------------------
    # 8. search_kernel bridge test
    # ------------------------------------------------------------------
    print("\n[8/8] Testing search_kernel() bridge...")
    try:
        from compgen.kernels.autocomp_adapter import search_kernel

        bridge_result = search_kernel(
            region_id="bridge_test_0",
            job={
                "op_family": "matmul_bias_gelu",
                "input_shapes": [(M, K), (K, N)],
                "output_shapes": [(M, N)],
                "dtypes": ["float32"],
                "target_name": "cuda_a100",
            },
            target=None,
        )
        bridge_ok = bridge_result["found"]
        gates.append(Gate("search_kernel_bridge", bridge_ok, f"found={bridge_ok}"))
        print(f"  Bridge result: found={bridge_ok}")
    except Exception as exc:
        gates.append(Gate("search_kernel_bridge", False, str(exc)))
        print(f"  Bridge error: {exc}")

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    _print_report(gates)

    all_passed = all(g.passed for g in gates)
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
