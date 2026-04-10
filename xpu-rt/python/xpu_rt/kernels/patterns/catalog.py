"""Kernel pattern catalog -- the minimum set of reusable kernel patterns.

Collapses per-shape kernel contracts into parameterized patterns. Multiple
graph ops with different shapes but the same computation share one pattern.

Example: 150 ``aten.linear`` ops with varying (M, K, N) dimensions all
belong to a single ``"matmul"`` pattern parameterized by M, K, N.

Framework-agnostic: works with any model, any target.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from compgen.kernels.patterns.detection import DetectedPattern


@dataclass
class KernelPattern:
    """A reusable, parameterized kernel pattern.

    Represents a class of computations (e.g., ``"matmul"``) rather than a
    specific shape.  Multiple graph ops share one pattern.

    Attributes:
        pattern_id: Unique identifier (``"matmul"``, ``"softmax"``, etc.).
        op_family: Broad category (``"matmul"``, ``"elementwise"``, ``"reduction"``).
        description: Human-readable description of what this pattern computes.
        fused_ops: PyTorch ops this pattern fuses (e.g., ``["linear", "silu"]``).
        parameter_schema: What varies across instances (``{"M": "int", ...}``).
        shape_variants: All concrete parameter instances from the model.
        instance_count: Total graph ops covered by this pattern.
        total_flops: Total FLOPs across all instances.
        compute_fraction: Fraction of total model FLOPs.
        reference_fn: Python code implementing the computation.
        input_generator_fn: Python code generating test inputs for any params.
        priority_shapes: Top shapes by FLOPs (implement these first).
    """

    pattern_id: str = ""
    op_family: str = ""
    description: str = ""
    fused_ops: list[str] = field(default_factory=list)
    parameter_schema: dict[str, str] = field(default_factory=dict)
    shape_variants: list[dict[str, Any]] = field(default_factory=list)
    instance_count: int = 0
    total_flops: int = 0
    compute_fraction: float = 0.0
    reference_fn: str = ""
    input_generator_fn: str = ""
    priority_shapes: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to YAML/JSON-compatible dict."""
        return {
            "pattern_id": self.pattern_id,
            "op_family": self.op_family,
            "description": self.description,
            "fused_ops": self.fused_ops,
            "parameter_schema": self.parameter_schema,
            "instance_count": self.instance_count,
            "total_flops": self.total_flops,
            "compute_fraction": round(self.compute_fraction, 4),
            "shape_variants_count": len(self.shape_variants),
            "priority_shapes": self.priority_shapes[:5],
        }


# ---------------------------------------------------------------------------
# Reference code generators per pattern type
# ---------------------------------------------------------------------------

_REFERENCE_CODE: dict[str, str] = {
    "matmul": '''\
import torch
import torch.nn.functional as F

def compute(activation, weight):
    """Matrix multiply: activation @ weight^T (F.linear semantics).

    For NPU: FP8 inputs, BF16 accumulation, 32x32 tile.
    """
    return F.linear(activation.to(torch.bfloat16), weight.to(torch.bfloat16))
''',
    "batch_matmul": '''\
import torch

def compute(A, B):
    """Batched matrix multiply: A @ B.

    Used for attention score (Q @ K^T) and attention output (weights @ V).
    """
    return torch.matmul(A.to(torch.bfloat16), B.to(torch.bfloat16))
''',
    "fused_linear_silu": '''\
import torch
import torch.nn.functional as F

def compute(activation, weight):
    """Linear + SiLU activation (gate projection in Gemma MLP)."""
    return F.silu(F.linear(activation.to(torch.bfloat16), weight.to(torch.bfloat16)))
''',
    "fused_linear_gelu": '''\
import torch
import torch.nn.functional as F

def compute(activation, weight):
    """Linear + GELU activation (SigLIP MLP)."""
    return F.gelu(F.linear(activation.to(torch.bfloat16), weight.to(torch.bfloat16)))
''',
    "softmax": '''\
import torch

def compute(x):
    """Softmax over last dimension. Always BF16 (never quantized)."""
    return torch.softmax(x.to(torch.bfloat16), dim=-1)
''',
    "silu": '''\
import torch
import torch.nn.functional as F

def compute(x):
    """SiLU (Swish) activation."""
    return F.silu(x.to(torch.bfloat16))
''',
    "gelu": '''\
import torch
import torch.nn.functional as F

def compute(x):
    """GELU activation (tanh approximation)."""
    return F.gelu(x.to(torch.bfloat16), approximate="tanh")
''',
    "elementwise_binary": '''\
import torch

def compute(a, b):
    """Generic elementwise binary operation (add/sub/mul/div)."""
    return (a.to(torch.bfloat16) + b.to(torch.bfloat16))  # Replace op as needed
''',
    "elementwise_unary": '''\
import torch

def compute(x):
    """Generic elementwise unary operation (exp/sin/cos/tanh/etc)."""
    return torch.exp(x.to(torch.bfloat16))  # Replace op as needed
''',
    "reduction": '''\
import torch

def compute(x):
    """Reduction over last dimension (sum/mean/amax)."""
    return x.to(torch.bfloat16).sum(dim=-1)
''',
    "conv2d": '''\
import torch
import torch.nn.functional as F

def compute(x, weight):
    """2D convolution (patch embedding)."""
    return F.conv2d(x.to(torch.bfloat16), weight.to(torch.bfloat16), padding=0, stride=16)
''',
}

_INPUT_GENERATORS: dict[str, str] = {
    "matmul": '''\
import torch

def generate_inputs(M=64, K=768, N=3072, dtype=torch.bfloat16):
    """Generate test inputs for matmul pattern."""
    torch.manual_seed(42)
    activation = torch.randn(1, M, K, dtype=dtype)
    weight = torch.randn(N, K, dtype=dtype)  # F.linear weight shape: [out, in]
    return {"activation": activation, "weight": weight}
''',
    "batch_matmul": '''\
import torch

def generate_inputs(batch=1, heads=15, seq_q=291, seq_k=64, dtype=torch.bfloat16):
    """Generate test inputs for batch matmul pattern."""
    torch.manual_seed(42)
    A = torch.randn(batch, heads, seq_q, seq_k, dtype=dtype)
    B = torch.randn(batch, heads, seq_k, seq_q, dtype=dtype)
    return {"A": A, "B": B}
''',
    "softmax": '''\
import torch

def generate_inputs(batch=1, heads=15, seq=291, dtype=torch.bfloat16):
    torch.manual_seed(42)
    x = torch.randn(batch, heads, seq, seq, dtype=dtype)
    return {"x": x}
''',
}


# ---------------------------------------------------------------------------
# Catalog builder
# ---------------------------------------------------------------------------

def build_pattern_catalog(
    detected: list[DetectedPattern],
) -> list[KernelPattern]:
    """Build a pattern catalog from detected FX graph patterns.

    Groups detected patterns by type, collects unique shape variants,
    computes aggregate statistics, and sorts by total FLOPs.

    Args:
        detected: List of detected patterns from ``detect_patterns_in_graphs()``.

    Returns:
        List of ``KernelPattern``, sorted by total FLOPs (highest first).
    """
    # Group by pattern_type
    groups: dict[str, list[DetectedPattern]] = defaultdict(list)
    for p in detected:
        groups[p.pattern_type].append(p)

    total_flops_all = sum(p.flops for p in detected)
    patterns: list[KernelPattern] = []

    for ptype, instances in groups.items():
        # Collect unique shape variants
        seen_shapes: set[str] = set()
        shape_variants: list[dict[str, Any]] = []
        for inst in instances:
            key = str(inst.params) if inst.params else str(inst.input_shapes)
            if key not in seen_shapes:
                seen_shapes.add(key)
                variant = dict(inst.params) if inst.params else {}
                variant["input_shapes"] = inst.input_shapes
                variant["output_shape"] = inst.output_shape
                variant["flops"] = inst.flops
                shape_variants.append(variant)

        # Sort variants by FLOPs
        shape_variants.sort(key=lambda v: v.get("flops", 0), reverse=True)

        total_flops = sum(inst.flops for inst in instances)
        compute_frac = total_flops / total_flops_all if total_flops_all > 0 else 0.0

        # Determine op family
        if ptype in ("matmul", "fused_linear_silu", "fused_linear_gelu", "fused_linear_relu"):
            op_family = "matmul"
        elif ptype == "batch_matmul":
            op_family = "matmul"
        elif ptype in ("elementwise_binary", "elementwise_unary", "silu", "gelu", "softmax"):
            op_family = "elementwise"
        elif ptype == "reduction":
            op_family = "reduction"
        elif ptype == "conv2d":
            op_family = "conv"
        else:
            op_family = ptype

        # Parameter schema
        if shape_variants and shape_variants[0]:
            schema = {k: type(v).__name__ for k, v in shape_variants[0].items()
                      if k not in ("input_shapes", "output_shape", "flops")}
        else:
            schema = {}

        # Description
        fused_ops = [ptype]
        if "fused_linear" in ptype:
            parts = ptype.replace("fused_linear_", "").split("_")
            fused_ops = ["linear", *parts]

        desc_map = {
            "matmul": "Matrix multiply (F.linear): activation @ weight^T",
            "batch_matmul": "Batched matrix multiply (attention scores/output)",
            "fused_linear_silu": "Linear + SiLU activation (Gemma MLP gate)",
            "fused_linear_gelu": "Linear + GELU activation (SigLIP MLP)",
            "softmax": "Softmax over last dimension (always BF16)",
            "silu": "SiLU (Swish) activation",
            "gelu": "GELU activation (tanh approximation)",
            "elementwise_binary": "Generic binary op (add/sub/mul/div)",
            "elementwise_unary": "Generic unary op (exp/sin/cos/tanh/rsqrt/...)",
            "reduction": "Reduction (sum/mean/amax)",
            "conv2d": "2D convolution (vision patch embedding)",
        }

        patterns.append(KernelPattern(
            pattern_id=ptype,
            op_family=op_family,
            description=desc_map.get(ptype, f"Compute pattern: {ptype}"),
            fused_ops=fused_ops,
            parameter_schema=schema,
            shape_variants=shape_variants,
            instance_count=len(instances),
            total_flops=total_flops,
            compute_fraction=compute_frac,
            reference_fn=_REFERENCE_CODE.get(ptype, f"# Reference for {ptype}\nimport torch\n"),
            input_generator_fn=_INPUT_GENERATORS.get(ptype, ""),
            priority_shapes=shape_variants[:5],
        ))

    patterns.sort(key=lambda p: p.total_flops, reverse=True)
    return patterns


# ---------------------------------------------------------------------------
# Export and reporting
# ---------------------------------------------------------------------------

def export_pattern_catalog(
    patterns: list[KernelPattern],
    output_dir: str | Path,
) -> Path:
    """Export pattern catalog to YAML files.

    Creates one directory per pattern with metadata, reference code, and
    input generator.

    Args:
        patterns: List of kernel patterns.
        output_dir: Root directory for pattern output.

    Returns:
        Path to the output directory.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Catalog index
    catalog = {
        "total_patterns": len(patterns),
        "total_instances": sum(p.instance_count for p in patterns),
        "patterns": [p.to_dict() for p in patterns],
    }
    (out / "catalog.yaml").write_text(yaml.dump(catalog, default_flow_style=False, sort_keys=False))

    # Per-pattern directories
    for pattern in patterns:
        pdir = out / pattern.pattern_id
        pdir.mkdir(exist_ok=True)

        # Pattern metadata
        (pdir / "pattern.yaml").write_text(yaml.dump(pattern.to_dict(), default_flow_style=False, sort_keys=False))

        # Reference implementation
        (pdir / "reference.py").write_text(pattern.reference_fn)

        # Input generator
        if pattern.input_generator_fn:
            (pdir / "input_generator.py").write_text(pattern.input_generator_fn)

        # All shape variants
        (pdir / "all_variants.yaml").write_text(yaml.dump(
            {"variants": pattern.shape_variants}, default_flow_style=False,
        ))

    return out


def format_pattern_report(patterns: list[KernelPattern]) -> str:
    """Format a human-readable pattern catalog report."""
    lines = [
        "=" * 70,
        "  Kernel Pattern Catalog — Reusable Patterns",
        "=" * 70,
        "",
        f"  Total patterns:    {len(patterns)}",
        f"  Total op instances: {sum(p.instance_count for p in patterns)}",
        "",
        "  Patterns by priority (total FLOPs):",
        "",
    ]

    for i, p in enumerate(patterns):
        pct = f"{p.compute_fraction * 100:.1f}%" if p.compute_fraction > 0 else "—"
        lines.append(
            f"  {i + 1:3d}. {p.pattern_id:30s}  "
            f"x{p.instance_count:4d} ops  "
            f"{p.total_flops:>15,} FLOPs  "
            f"({pct:>6s})  "
            f"[{len(p.shape_variants)} shapes]"
        )

    lines.extend(["", "=" * 70])
    return "\n".join(lines)


__all__ = [
    "KernelPattern",
    "build_pattern_catalog",
    "export_pattern_catalog",
    "format_pattern_report",
]
