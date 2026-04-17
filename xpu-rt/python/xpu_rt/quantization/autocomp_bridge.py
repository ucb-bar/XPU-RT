"""Bridge from NPU kernel contracts to autocomp search format.

Converts ``NpuKernelContract`` into the format that autocomp's search
pipeline expects (``Prob`` + test harness + hardware config), and provides
utilities to load autocomp results back into CompGen.

This module does NOT duplicate autocomp's LLMClient or search infrastructure
(per CLAUDE.md rules). It only handles format translation.

Usage::

    from compgen.quantization.autocomp_bridge import (
        contract_to_autocomp_prob,
        load_autocomp_result,
    )

    prob = contract_to_autocomp_prob(contract, output_dir=Path("search/matmul_0"))
    # ... run autocomp search externally ...
    result = load_autocomp_result(contract.contract_id, results_dir)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from compgen.kernels.providers.npu_contracts import NpuKernelContract


@dataclass(frozen=True)
class AutocompKernelResult:
    """Result loaded from an autocomp search run.

    Attributes:
        contract_id: Which contract this result is for.
        kernel_code: Best kernel code found.
        language: Kernel language (``"python"``, ``"c"``, ``"npu_asm"``).
        latency_us: Measured latency in microseconds.
        correct: Whether the kernel passed correctness checks.
        iterations_used: How many search iterations were run.
        plan: The optimization plan that produced this kernel.
    """

    contract_id: str
    kernel_code: str
    language: str
    latency_us: float
    correct: bool
    iterations_used: int = 0
    plan: str = ""


def contract_to_autocomp_prob(
    contract: NpuKernelContract,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Convert an NpuKernelContract to an autocomp-compatible problem.

    Creates the directory structure that autocomp's search pipeline expects:
    - ``reference.py``: Reference PyTorch implementation
    - ``test.py``: Test harness with inputs and correctness check
    - ``prob.json``: Problem metadata

    Does NOT import autocomp directly — returns a dict that can be passed
    to ``autocomp.search.prob.Prob`` by the caller.

    Args:
        contract: The kernel contract to convert.
        output_dir: Directory to write the problem files.

    Returns:
        Dict with keys ``prob_type``, ``prob_id``, ``test_file``, ``sol_file``,
        ``context`` — matching ``autocomp.search.prob.Prob`` constructor args.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Write reference code
    sol_file = out / "reference.py"
    sol_file.write_text(contract.reference_pytorch)

    # Write test harness
    test_file = out / "test.py"
    test_code = _build_test_harness(contract)
    test_file.write_text(test_code)

    # Write problem metadata
    context = (
        f"NPU kernel for {contract.op_family} operation.\n"
        f"Input shapes: {contract.input_shapes}\n"
        f"Input dtypes: {contract.input_dtypes} -> {contract.output_dtype}\n"
        f"NPU unit: {contract.npu_unit} (tile: {contract.tile_shape})\n"
        f"Accumulation: {contract.accumulation_dtype}\n"
        f"Scale format: {contract.scale_format}\n"
        f"This kernel covers {contract.instance_count} ops in the SmolVLA graph.\n"
        f"ISA mnemonic: {contract.isa_mnemonic}\n"
    )

    prob_meta = {
        "prob_type": "compgen_npu",
        "prob_id": hash(contract.contract_id) % 10000,
        "test_file": str(test_file),
        "sol_file": str(sol_file),
        "context": context,
    }

    (out / "prob.json").write_text(json.dumps(prob_meta, indent=2))

    return prob_meta


def load_autocomp_result(
    contract_id: str,
    results_dir: str | Path,
) -> AutocompKernelResult | None:
    """Load the best kernel result from an autocomp search run.

    Expects autocomp's standard output structure::

        results_dir/
            results.json          # Contains best_score, best_code
            candidates-iter-N/    # Per-iteration candidates
            run_metadata.json     # Search metadata

    Args:
        contract_id: The contract this result is for.
        results_dir: Path to autocomp's output directory.

    Returns:
        ``AutocompKernelResult`` if results found, ``None`` otherwise.
    """
    rdir = Path(results_dir)

    # Try autocomp's results.json format
    results_file = rdir / "results.json"
    if results_file.exists():
        data = json.loads(results_file.read_text())
        return AutocompKernelResult(
            contract_id=contract_id,
            kernel_code=data.get("best_code", ""),
            language=data.get("language", "python"),
            latency_us=data.get("best_score", 0.0),
            correct=data.get("correct", False),
            iterations_used=data.get("iterations", 0),
            plan=data.get("best_plan", ""),
        )

    # Try reading best candidate directly
    for candidate_dir in sorted(rdir.glob("candidates-iter-*"), reverse=True):
        for candidate_file in sorted(candidate_dir.glob("candidate_*.txt"), reverse=True):
            code = candidate_file.read_text()
            if code.strip():
                return AutocompKernelResult(
                    contract_id=contract_id,
                    kernel_code=code,
                    language="python",
                    latency_us=0.0,
                    correct=False,  # Needs validation
                    iterations_used=0,
                )

    return None


def validate_kernel_against_contract(
    kernel_code: str,
    contract: NpuKernelContract,
    atol: float = 1e-2,
    rtol: float = 1e-2,
) -> dict[str, Any]:
    """Validate a generated kernel against its contract's reference.

    Runs both the reference PyTorch code and the kernel, comparing outputs.

    Args:
        kernel_code: The generated kernel code.
        contract: The kernel contract with reference implementation.
        atol: Absolute tolerance for correctness.
        rtol: Relative tolerance for correctness.

    Returns:
        Dict with ``correct``, ``max_error``, ``latency_us`` keys.
    """
    import torch
    import time

    # Execute reference
    ref_ns: dict[str, Any] = {}
    exec(contract.reference_pytorch, {"torch": torch, "F": torch.nn.functional}, ref_ns)

    # Execute kernel
    kernel_ns: dict[str, Any] = {}
    try:
        exec(kernel_code, {"torch": torch, "F": torch.nn.functional}, kernel_ns)
    except Exception as e:
        return {"correct": False, "max_error": float("inf"), "latency_us": 0.0, "error": str(e)}

    # Compare outputs
    ref_out = ref_ns.get("C") or ref_ns.get("out")
    kernel_out = kernel_ns.get("C") or kernel_ns.get("out")

    if ref_out is None or kernel_out is None:
        return {"correct": False, "max_error": float("inf"), "latency_us": 0.0, "error": "no output variable"}

    try:
        correct = torch.allclose(ref_out.float(), kernel_out.float(), atol=atol, rtol=rtol)
        max_error = (ref_out.float() - kernel_out.float()).abs().max().item()
    except Exception as e:
        return {"correct": False, "max_error": float("inf"), "latency_us": 0.0, "error": str(e)}

    return {"correct": correct, "max_error": max_error, "latency_us": 0.0}


def _build_test_harness(contract: NpuKernelContract) -> str:
    """Build autocomp-compatible test harness code."""
    dtype_map = {"bf16": "torch.bfloat16", "f32": "torch.float32", "fp8_e4m3": "torch.bfloat16"}
    py_dtype = dtype_map.get(contract.input_dtypes[0], "torch.bfloat16") if contract.input_dtypes else "torch.bfloat16"

    init_lines = []
    for i, shape in enumerate(contract.input_shapes):
        init_lines.append(f"input_{i} = torch.randn({list(shape)}, dtype={py_dtype})")

    inputs_init = "\n    ".join(init_lines)
    input_args = ", ".join(f"input_{i}" for i in range(len(contract.input_shapes)))

    return f'''"""Autocomp test harness — {contract.contract_id}
Op: {contract.op_family} | Unit: {contract.npu_unit} | Instances: {contract.instance_count}
"""
import torch
import time

def get_inputs():
    torch.manual_seed(42)
    {inputs_init}
    return ({input_args},)

def reference({input_args}):
    """Reference implementation (correctness oracle)."""
{_indent_block(contract.reference_pytorch, 4)}

def check_correctness(candidate_output, reference_output):
    return torch.allclose(candidate_output.float(), reference_output.float(), atol=1e-2, rtol=1e-2)

if __name__ == "__main__":
    inputs = get_inputs()
    ref_out = reference(*inputs)
    print(f"Reference output shape: {{ref_out.shape}}, dtype: {{ref_out.dtype}}")
'''


def _indent_block(code: str, spaces: int) -> str:
    prefix = " " * spaces
    lines = code.strip().split("\n")
    return "\n".join(prefix + line for line in lines)


__all__ = [
    "AutocompKernelResult",
    "contract_to_autocomp_prob",
    "load_autocomp_result",
    "validate_kernel_against_contract",
]
