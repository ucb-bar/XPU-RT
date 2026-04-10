"""NPU kernel contract generation with shape deduplication.

Analyzes captured FX graph partitions to produce the **minimum set of unique
kernel contracts** that cover all compute ops.  Many ops in a transformer
model share identical (shape, dtype) signatures (e.g., all Q/K/V projections
in the same layer have the same dimensions), so deduplication drastically
reduces the number of kernels that need to be generated.

Each contract specifies exactly what a kernel must implement:
- Input/output shapes and dtypes
- NPU execution unit (MXU for matmuls, VPU for vector ops)
- Accumulation dtype and tile geometry
- Reference PyTorch code for correctness testing
- Priority based on total FLOPs across all instances

Usage::

    from compgen.quantization.kernel_contracts import (
        generate_npu_kernel_contracts,
        export_contracts_yaml,
    )

    contracts = generate_npu_kernel_contracts(captured_graphs)
    export_contracts_yaml(contracts, Path("artifacts/kernel_contracts"))
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import yaml

from compgen.quantization.npu_op_map import (
    NpuOpCategory,
    NpuQuantDecision,
    _OP_TABLE,
)
from compgen.quantization.graph_analyzer import _FN_TO_ATEN, _normalize_fn_target


@dataclass
class NpuKernelContract:
    """Specification for a single NPU kernel that needs to be generated.

    One contract may cover many graph ops that share the same signature.

    Attributes:
        contract_id: Unique identifier (e.g., ``"matmul_fp8_768x3072x768"``).
        op_family: Operation family (``"matmul"``, ``"conv2d"``, ``"softmax"``, etc.).
        npu_unit: Target execution unit (``"mxu"``, ``"vpu"``, ``"xlu"``).
        input_shapes: Concrete input tensor shapes.
        output_shapes: Concrete output tensor shapes.
        input_dtypes: Input dtype per operand (``"fp8_e4m3"``, ``"bf16"``).
        output_dtype: Output dtype (``"bf16"`` for MXU accumulation).
        accumulation_dtype: Internal accumulation dtype (``"bf16"`` for NPU).
        scale_format: Scale register format (``"e8m0"`` for po2, ``None`` for unscaled).
        tile_shape: NPU tile geometry (``(32, 32)`` for MXU).
        reference_pytorch: Reference PyTorch code for correctness testing.
        instance_count: Number of graph ops covered by this contract.
        source_ops: FX node names of ops covered by this contract.
        estimated_flops: FLOPs per invocation.
        total_flops: FLOPs summed across all instances.
        priority: Sort key for generation order (higher = generate first).
        isa_mnemonic: NPU ISA instruction (if directly mappable).
    """

    contract_id: str = ""
    op_family: str = ""
    npu_unit: str = ""
    input_shapes: list[tuple[int, ...]] = field(default_factory=list)
    output_shapes: list[tuple[int, ...]] = field(default_factory=list)
    input_dtypes: list[str] = field(default_factory=list)
    output_dtype: str = "bf16"
    accumulation_dtype: str = "bf16"
    scale_format: str | None = "e8m0"
    tile_shape: tuple[int, int] = (32, 32)
    reference_pytorch: str = ""
    instance_count: int = 0
    source_ops: list[str] = field(default_factory=list)
    estimated_flops: int = 0
    total_flops: int = 0
    priority: int = 0
    isa_mnemonic: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a YAML/JSON-compatible dict."""
        return {
            "contract_id": self.contract_id,
            "op_family": self.op_family,
            "npu_unit": self.npu_unit,
            "input_shapes": [list(s) for s in self.input_shapes],
            "output_shapes": [list(s) for s in self.output_shapes],
            "input_dtypes": self.input_dtypes,
            "output_dtype": self.output_dtype,
            "accumulation_dtype": self.accumulation_dtype,
            "scale_format": self.scale_format,
            "tile_shape": list(self.tile_shape),
            "reference_pytorch": self.reference_pytorch,
            "instance_count": self.instance_count,
            "source_ops": self.source_ops[:10],  # Truncate for readability
            "estimated_flops": self.estimated_flops,
            "total_flops": self.total_flops,
            "priority": self.priority,
            "isa_mnemonic": self.isa_mnemonic,
        }


# ---------------------------------------------------------------------------
# Op family classification
# ---------------------------------------------------------------------------

_MATMUL_OPS = {"aten.linear.default", "aten.mm.default", "aten.addmm.default", "aten.bmm.default"}
_CONV_OPS = {"aten.conv2d", "aten.convolution.default"}
_SOFTMAX_OPS = {"aten._softmax.default", "aten.softmax.int"}
_ELEMENTWISE_BINARY = {"aten.add.Tensor", "aten.sub.Tensor", "aten.mul.Tensor", "aten.div.Tensor"}
_ELEMENTWISE_UNARY = {
    "aten.exp.default", "aten.exp2.default", "aten.log2.default",
    "aten.sin.default", "aten.cos.default", "aten.tanh.default",
    "aten.sqrt.default", "aten.reciprocal.default",
    "aten.relu.default", "aten.gelu.default", "aten.silu.default",
    "aten.pow.Tensor_Scalar", "aten.clamp.default", "aten.abs.default",
    "aten.neg.default",
}
_REDUCTION_OPS = {"aten.sum.default", "aten.sum.dim_IntList", "aten.amax.default", "aten.mean.dim"}


def _classify_op_family(aten_target: str) -> str | None:
    """Classify an ATen target into an op family. Returns None for passthrough."""
    if aten_target in _MATMUL_OPS:
        return "matmul"
    if aten_target in _CONV_OPS:
        return "conv2d"
    if aten_target in _SOFTMAX_OPS:
        return "softmax"
    if aten_target in _ELEMENTWISE_BINARY:
        return "elementwise_binary"
    if aten_target in _ELEMENTWISE_UNARY:
        return "elementwise_unary"
    if aten_target in _REDUCTION_OPS:
        return "reduction"
    if aten_target.startswith("passthrough."):
        return None
    return None


def _npu_unit_for_family(family: str) -> str:
    """Map op family to NPU execution unit."""
    if family in ("matmul", "conv2d"):
        return "mxu"
    if family in ("reduction",):
        return "xlu"
    return "vpu"


# ---------------------------------------------------------------------------
# Shape extraction from FX nodes
# ---------------------------------------------------------------------------

def _extract_shape(val: Any) -> tuple[int, ...] | None:
    """Extract shape from an FX node's meta value."""
    if val is None:
        return None
    if hasattr(val, "shape"):
        return tuple(int(d) for d in val.shape)
    return None


def _extract_dtype(val: Any) -> str:
    """Extract dtype string from an FX node's meta value."""
    if val is None:
        return "bf16"
    if hasattr(val, "dtype"):
        dtype = val.dtype
        if dtype == torch.bfloat16:
            return "bf16"
        if dtype == torch.float32:
            return "f32"
        if dtype == torch.float16:
            return "f16"
        if hasattr(torch, "float8_e4m3fn") and dtype == torch.float8_e4m3fn:
            return "fp8_e4m3"
    return "bf16"


def _estimate_matmul_flops(input_shapes: list[tuple[int, ...]]) -> int:
    """Estimate FLOPs for a matmul from input shapes."""
    if len(input_shapes) < 2:
        return 0
    a, b = input_shapes[0], input_shapes[1]
    if len(a) >= 2 and len(b) >= 2:
        m, k = a[-2], a[-1]
        n = b[-1]
        batch = 1
        for d in a[:-2]:
            batch *= d
        return 2 * batch * m * k * n
    return 0


def _normalize_target(node: Any) -> str:
    """Normalize an FX node target to ATen string form."""
    raw = node.target
    if hasattr(raw, "name"):
        name = raw.name() if callable(getattr(raw, "name", None)) else str(raw)
        return f"aten.{name}" if not name.startswith("aten.") else name
    if hasattr(raw, "__module__") and hasattr(raw, "__name__"):
        return _normalize_fn_target(raw)
    return str(raw)


# ---------------------------------------------------------------------------
# Contract generation with deduplication
# ---------------------------------------------------------------------------

# Signature = (op_family, input_shapes_tuple, input_dtypes_tuple)
_SignatureKey = tuple[str, tuple[tuple[int, ...], ...], tuple[str, ...]]


def generate_npu_kernel_contracts(
    graphs: list[torch.fx.GraphModule],
    model: torch.nn.Module | None = None,
) -> list[NpuKernelContract]:
    """Generate deduplicated NPU kernel contracts from captured FX graphs.

    Walks all ``call_function`` nodes, extracts concrete shapes from node
    metadata, groups ops by (family, shapes, dtypes) signature, and produces
    one contract per unique signature.

    Args:
        graphs: Captured FX graph partitions.
        model: Optional model for additional metadata extraction.

    Returns:
        List of ``NpuKernelContract``, sorted by priority (highest first).
    """
    # Accumulate ops by signature
    signature_groups: dict[_SignatureKey, list[dict[str, Any]]] = defaultdict(list)

    for gi, graph in enumerate(graphs):
        for node in graph.graph.nodes:
            if node.op != "call_function":
                continue

            aten_target = _normalize_target(node)
            family = _classify_op_family(aten_target)
            if family is None:
                continue

            # Extract shapes from node metadata.
            # Dynamo graphs use "example_value"; torch.export uses "val".
            input_shapes: list[tuple[int, ...]] = []
            input_dtypes: list[str] = []
            for arg in node.args:
                if hasattr(arg, "meta"):
                    meta_val = arg.meta.get("val") or arg.meta.get("example_value")
                    if meta_val is not None:
                        shape = _extract_shape(meta_val)
                        dtype = _extract_dtype(meta_val)
                        if shape is not None:
                            input_shapes.append(shape)
                            input_dtypes.append(dtype)

            output_shape: tuple[int, ...] | None = None
            output_dtype = "bf16"
            node_val = node.meta.get("val") or node.meta.get("example_value")
            if node_val is not None:
                output_shape = _extract_shape(node_val)
                output_dtype = _extract_dtype(node_val)

            if not input_shapes:
                continue

            # Build signature key for deduplication
            sig_key: _SignatureKey = (
                family,
                tuple(input_shapes),
                tuple(input_dtypes),
            )

            signature_groups[sig_key].append({
                "aten_target": aten_target,
                "node_name": node.name,
                "graph_idx": gi,
                "input_shapes": input_shapes,
                "input_dtypes": input_dtypes,
                "output_shape": output_shape,
                "output_dtype": output_dtype,
            })

    # Build contracts from grouped signatures
    contracts: list[NpuKernelContract] = []

    for sig_key, ops in signature_groups.items():
        family, input_shapes_tuple, input_dtypes_tuple = sig_key
        npu_unit = _npu_unit_for_family(family)

        # Determine dtypes for NPU
        if npu_unit == "mxu":
            # MXU: FP8 inputs, BF16 accumulation
            npu_input_dtypes = ["fp8_e4m3"] * len(input_shapes_tuple)
            npu_output_dtype = "bf16"
            accum_dtype = "bf16"
            scale_fmt: str | None = "e8m0"
        else:
            # VPU/XLU: BF16 throughout
            npu_input_dtypes = ["bf16"] * len(input_shapes_tuple)
            npu_output_dtype = "bf16"
            accum_dtype = "bf16"
            scale_fmt = None

        # Estimate FLOPs
        flops = 0
        if family == "matmul":
            flops = _estimate_matmul_flops(list(input_shapes_tuple))
        elif family == "conv2d" and len(input_shapes_tuple) >= 2:
            # Rough estimate: N*C_out*H*W*C_in*kH*kW*2
            pass
        else:
            # Elementwise: product of output shape
            out_shape = ops[0].get("output_shape")
            if out_shape:
                flops = 1
                for d in out_shape:
                    flops *= d

        # Build shape string for contract ID
        shape_str = "x".join(
            "x".join(str(d) for d in s) for s in input_shapes_tuple
        )
        contract_id = f"{family}_{npu_input_dtypes[0]}_{shape_str}"

        # ISA mnemonic
        isa = None
        first_target = ops[0]["aten_target"]
        decision = _OP_TABLE.get(first_target)
        if decision is not None:
            isa = decision.isa_mnemonic

        # Reference PyTorch code
        ref_code = _generate_reference_pytorch(family, list(input_shapes_tuple), list(input_dtypes_tuple))

        output_shapes: list[tuple[int, ...]] = []
        for op in ops:
            if op.get("output_shape") and op["output_shape"] not in output_shapes:
                output_shapes.append(op["output_shape"])
                break

        contracts.append(NpuKernelContract(
            contract_id=contract_id,
            op_family=family,
            npu_unit=npu_unit,
            input_shapes=list(input_shapes_tuple),
            output_shapes=output_shapes,
            input_dtypes=npu_input_dtypes,
            output_dtype=npu_output_dtype,
            accumulation_dtype=accum_dtype,
            scale_format=scale_fmt,
            tile_shape=(32, 32),
            reference_pytorch=ref_code,
            instance_count=len(ops),
            source_ops=[op["node_name"] for op in ops[:20]],
            estimated_flops=flops,
            total_flops=flops * len(ops),
            priority=flops * len(ops),
            isa_mnemonic=isa,
        ))

    # Sort by priority (highest first)
    contracts.sort(key=lambda c: c.priority, reverse=True)
    return contracts


# ---------------------------------------------------------------------------
# Reference PyTorch code generation
# ---------------------------------------------------------------------------

def _generate_reference_pytorch(
    family: str,
    input_shapes: list[tuple[int, ...]],
    input_dtypes: list[str],
) -> str:
    """Generate minimal reference PyTorch code for correctness testing."""
    dtype_map = {"bf16": "torch.bfloat16", "f32": "torch.float32", "fp8_e4m3": "torch.bfloat16"}
    py_dtype = dtype_map.get(input_dtypes[0], "torch.bfloat16") if input_dtypes else "torch.bfloat16"

    if family == "matmul" and len(input_shapes) >= 2:
        a_shape = list(input_shapes[0])
        b_shape = list(input_shapes[1])
        # Many matmuls come from F.linear where weight is [out, in].
        # Detect incompatible shapes and use F.linear instead of mm.
        a_last = a_shape[-1] if a_shape else 0
        b_first = b_shape[-2] if len(b_shape) >= 2 else (b_shape[0] if b_shape else 0)
        if a_last == b_first:
            # Compatible for matmul: A @ B
            return (
                f"import torch\n"
                f"# Matmul: {a_shape} @ {b_shape}\n"
                f"A = torch.randn({a_shape}, dtype={py_dtype})\n"
                f"B = torch.randn({b_shape}, dtype={py_dtype})\n"
                f"C = torch.mm(A, B) if A.dim() == 2 else torch.matmul(A, B)\n"
            )
        else:
            # Likely F.linear: A @ B^T where B is [out_features, in_features]
            return (
                f"import torch\n"
                f"import torch.nn.functional as F\n"
                f"# Linear (A @ B^T): input={a_shape}, weight={b_shape}\n"
                f"A = torch.randn({a_shape}, dtype={py_dtype})\n"
                f"B = torch.randn({b_shape}, dtype={py_dtype})\n"
                f"C = F.linear(A, B)  # Computes A @ B^T\n"
            )
    elif family == "conv2d" and len(input_shapes) >= 2:
        x_shape = list(input_shapes[0])
        w_shape = list(input_shapes[1])
        return (
            f"import torch\n"
            f"import torch.nn.functional as F\n"
            f"# Conv2d: input={x_shape}, weight={w_shape}\n"
            f"x = torch.randn({x_shape}, dtype={py_dtype})\n"
            f"w = torch.randn({w_shape}, dtype={py_dtype})\n"
            f"out = F.conv2d(x, w, padding=1)\n"
        )
    elif family == "softmax" and input_shapes:
        shape = list(input_shapes[0])
        return (
            f"import torch\n"
            f"# Softmax over last dimension\n"
            f"x = torch.randn({shape}, dtype={py_dtype})\n"
            f"out = torch.softmax(x, dim=-1)\n"
        )
    elif family in ("elementwise_binary", "elementwise_unary") and input_shapes:
        shape = list(input_shapes[0])
        if family == "elementwise_binary" and len(input_shapes) >= 2:
            b_shape = list(input_shapes[1])
            return (
                f"import torch\n"
                f"# Elementwise binary: {shape} op {b_shape}\n"
                f"A = torch.randn({shape}, dtype={py_dtype})\n"
                f"B = torch.randn({b_shape}, dtype={py_dtype})\n"
                f"out = A + B  # Replace with target op\n"
            )
        return (
            f"import torch\n"
            f"# Elementwise unary: {shape}\n"
            f"x = torch.randn({shape}, dtype={py_dtype})\n"
            f"out = torch.exp(x)  # Replace with target op\n"
        )
    elif family == "reduction" and input_shapes:
        shape = list(input_shapes[0])
        return (
            f"import torch\n"
            f"# Reduction over last dimension\n"
            f"x = torch.randn({shape}, dtype={py_dtype})\n"
            f"out = x.sum(dim=-1)\n"
        )

    return f"# Reference code for {family} with shapes {input_shapes}\nimport torch\n"


# ---------------------------------------------------------------------------
# Export functions
# ---------------------------------------------------------------------------

def export_contracts_yaml(
    contracts: list[NpuKernelContract],
    output_dir: str | Path,
) -> Path:
    """Export kernel contracts to YAML files.

    Creates one YAML file per contract plus a ``summary.yaml`` index.

    Args:
        contracts: List of kernel contracts.
        output_dir: Directory to write YAML files.

    Returns:
        Path to the output directory.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Individual contract files
    for i, contract in enumerate(contracts):
        filename = f"{i:03d}_{contract.contract_id[:60]}.yaml"
        (out / filename).write_text(yaml.dump(contract.to_dict(), default_flow_style=False, sort_keys=False))

    # Summary index
    summary = {
        "total_contracts": len(contracts),
        "total_ops_covered": sum(c.instance_count for c in contracts),
        "contracts_by_unit": {},
        "contracts_by_family": {},
        "contracts": [],
    }

    for c in contracts:
        summary["contracts_by_unit"][c.npu_unit] = summary["contracts_by_unit"].get(c.npu_unit, 0) + 1
        summary["contracts_by_family"][c.op_family] = summary["contracts_by_family"].get(c.op_family, 0) + 1
        summary["contracts"].append({
            "contract_id": c.contract_id,
            "op_family": c.op_family,
            "npu_unit": c.npu_unit,
            "instance_count": c.instance_count,
            "total_flops": c.total_flops,
            "priority": c.priority,
        })

    (out / "summary.yaml").write_text(yaml.dump(summary, default_flow_style=False, sort_keys=False))
    return out


def export_contracts_autocomp(
    contracts: list[NpuKernelContract],
    output_dir: str | Path,
) -> Path:
    """Export kernel contracts in autocomp-ready format.

    For each contract, creates:
    - ``reference.py``: Reference PyTorch implementation
    - ``test.py``: Autocomp-compatible test harness
    - ``contract.yaml``: Full contract metadata

    Args:
        contracts: List of kernel contracts.
        output_dir: Directory to write autocomp packages.

    Returns:
        Path to the output directory.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for contract in contracts:
        pkg_dir = out / contract.contract_id[:60]
        pkg_dir.mkdir(exist_ok=True)

        # Reference code
        (pkg_dir / "reference.py").write_text(contract.reference_pytorch)

        # Test harness
        test_code = _generate_test_harness(contract)
        (pkg_dir / "test.py").write_text(test_code)

        # Contract metadata
        (pkg_dir / "contract.yaml").write_text(
            yaml.dump(contract.to_dict(), default_flow_style=False, sort_keys=False)
        )

    # Index file
    index = {
        "contracts": [c.contract_id for c in contracts],
        "total": len(contracts),
        "mxu_contracts": [c.contract_id for c in contracts if c.npu_unit == "mxu"],
        "vpu_contracts": [c.contract_id for c in contracts if c.npu_unit == "vpu"],
    }
    (out / "index.yaml").write_text(yaml.dump(index, default_flow_style=False))
    return out


def _generate_test_harness(contract: NpuKernelContract) -> str:
    """Generate an autocomp-compatible test harness for a contract."""
    dtype_map = {"bf16": "torch.bfloat16", "f32": "torch.float32", "fp8_e4m3": "torch.bfloat16"}
    py_dtype = dtype_map.get(contract.input_dtypes[0], "torch.bfloat16") if contract.input_dtypes else "torch.bfloat16"

    input_init = []
    for i, shape in enumerate(contract.input_shapes):
        input_init.append(f"    input_{i} = torch.randn({list(shape)}, dtype={py_dtype}, device='cpu')")

    inputs_str = "\n".join(input_init)
    input_args = ", ".join(f"input_{i}" for i in range(len(contract.input_shapes)))

    return f'''"""Autocomp test harness for contract: {contract.contract_id}

Op family: {contract.op_family}
NPU unit:  {contract.npu_unit}
Shapes:    {contract.input_shapes}
Dtypes:    {contract.input_dtypes} -> {contract.output_dtype}
Instances: {contract.instance_count} ops in the SmolVLA graph use this kernel
"""
import torch

def get_test_inputs():
    """Generate test inputs matching the contract shapes."""
    torch.manual_seed(42)
{inputs_str}
    return ({input_args},)

def reference_forward({input_args}):
    """Reference PyTorch implementation (golden output)."""
{_indent(contract.reference_pytorch.split("\\n")[-1] if "out = " in contract.reference_pytorch else "    return " + input_args.split(",")[0], 4)}

def check_correctness(kernel_output, reference_output, atol=1e-2, rtol=1e-2):
    """Check kernel output against reference."""
    return torch.allclose(kernel_output.float(), reference_output.float(), atol=atol, rtol=rtol)
'''


def _indent(code: str, spaces: int) -> str:
    """Add indentation to a code string."""
    prefix = " " * spaces
    return "\n".join(prefix + line if line.strip() else line for line in code.split("\n"))


def format_contracts_report(contracts: list[NpuKernelContract]) -> str:
    """Format a human-readable kernel contracts report."""
    lines = [
        "=" * 70,
        "  NPU Kernel Contracts — Deduplicated Summary",
        "=" * 70,
        "",
        f"  Total unique contracts: {len(contracts)}",
        f"  Total graph ops covered: {sum(c.instance_count for c in contracts)}",
        "",
        "  By NPU unit:",
    ]

    by_unit: dict[str, list[NpuKernelContract]] = defaultdict(list)
    for c in contracts:
        by_unit[c.npu_unit].append(c)

    for unit, unit_contracts in sorted(by_unit.items()):
        total_ops = sum(c.instance_count for c in unit_contracts)
        lines.append(f"    {unit.upper()}: {len(unit_contracts)} contracts covering {total_ops} ops")

    lines.append("")
    lines.append("  Top 20 contracts by priority:")
    for c in contracts[:20]:
        lines.append(
            f"    [{c.npu_unit.upper():3s}] {c.contract_id[:50]:50s}  "
            f"x{c.instance_count:3d} ops  {c.estimated_flops:>12,} FLOPs"
        )

    lines.append("")
    lines.append("=" * 70)
    return "\n".join(lines)


__all__ = [
    "NpuKernelContract",
    "export_contracts_autocomp",
    "export_contracts_yaml",
    "format_contracts_report",
    "generate_npu_kernel_contracts",
]
