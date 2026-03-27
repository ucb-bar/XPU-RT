"""Kernel strategy selection (Stage 2 gap analysis).

For each kernel specification, determines the best strategy:
- NATIVE: Can be lowered natively by the target's compiler (no custom kernel needed).
- LIBRARY: A vendor library kernel exists (cuBLAS, cuDNN, MKL, etc.).
- AUTOCOMP: Needs LLM-driven search via autocomp.
- FALLBACK: Use a generic but correct fallback implementation.
- UNSUPPORTED: Cannot be compiled for this target.

Invariants:
    - Every op must get a strategy (no silent drops).
    - UNSUPPORTED ops produce a clear diagnostic.
    - Strategy selection is deterministic given the same profile and contracts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from compgen.kernels.contracts import KernelSpec
from compgen.targets.schema import TargetProfile


class KernelStrategy(Enum):
    """Strategy for handling a kernel."""

    NATIVE = "native"
    LIBRARY = "library"
    AUTOCOMP = "autocomp"
    EXO = "exo"
    FALLBACK = "fallback"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True)
class StrategyDecision:
    """A strategy decision for a single kernel.

    Attributes:
        spec: The kernel specification.
        strategy: The chosen strategy.
        reason: Why this strategy was chosen.
        library_name: Library name (if strategy is LIBRARY).
    """

    spec: KernelSpec
    strategy: KernelStrategy
    reason: str
    library_name: str | None = None


# Ops that GPU vendor libraries handle well
_GPU_LIBRARY_OPS: dict[str, str] = {
    "linalg.matmul": "cublas",
    "linalg.batch_matmul": "cublas",
    "linalg.conv_2d_nhwc_hwcf": "cudnn",
    "linalg.softmax": "cudnn",
    "linalg.batch_norm": "cudnn",
    "linalg.layer_norm": "cudnn",
}

# Ops that are cheaply lowered natively (elementwise, reshape, etc.)
_NATIVE_OPS: frozenset[str] = frozenset({
    "arith.addi", "arith.subi", "arith.muli",
    "arith.addf", "arith.subf", "arith.mulf", "arith.divf",
    "arith.negf", "arith.maximumf", "arith.minimumf",
    "arith.constant", "arith.select",
    "arith.extf", "arith.truncf", "arith.sitofp", "arith.fptosi",
    "arith.cmpf", "arith.cmpi", "arith.index_cast",
    "linalg.fill", "linalg.transpose",
    "func.call",
})

# Minimum FLOPs to justify autocomp search (below this, use fallback)
_MIN_AUTOCOMP_FLOPS = 1000


@dataclass
class KernelSelector:
    """Selects kernel strategies based on target profile and contracts.

    Attributes:
        target: The target profile.
        native_ops: Set of op names with native lowering support.
        library_ops: Dict mapping op names to library names.
    """

    target: TargetProfile
    native_ops: set[str] = field(default_factory=set)
    library_ops: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Populate native_ops and library_ops from target profile."""
        self.native_ops = set(_NATIVE_OPS)

        # Check if target has GPU devices
        has_gpu = any(
            d.device_type == "gpu" for d in self.target.devices
        )
        if has_gpu:
            self.library_ops = dict(_GPU_LIBRARY_OPS)

        # Add any ops from device supported_ops
        for device in self.target.devices:
            if hasattr(device, "supported_ops") and device.supported_ops:
                self.native_ops.update(device.supported_ops)

    def select(self, specs: list[KernelSpec]) -> list[StrategyDecision]:
        """Select strategies for a list of kernel specifications.

        Args:
            specs: Kernel specifications from gap analysis.

        Returns:
            List of StrategyDecision, one per spec.
        """
        decisions: list[StrategyDecision] = []

        for spec in specs:
            op_name = spec.contract.op_name

            # 1. Check native lowering
            if op_name in self.native_ops:
                decisions.append(StrategyDecision(
                    spec=spec,
                    strategy=KernelStrategy.NATIVE,
                    reason=f"{op_name} has native lowering support",
                ))
                continue

            # 2. Check library coverage
            if op_name in self.library_ops:
                decisions.append(StrategyDecision(
                    spec=spec,
                    strategy=KernelStrategy.LIBRARY,
                    reason=f"{op_name} available in {self.library_ops[op_name]}",
                    library_name=self.library_ops[op_name],
                ))
                continue

            # 3. Check if worth autocomp search
            if spec.contract.cost.flops >= _MIN_AUTOCOMP_FLOPS:
                decisions.append(StrategyDecision(
                    spec=spec,
                    strategy=KernelStrategy.AUTOCOMP,
                    reason=f"{op_name} has {spec.contract.cost.flops} FLOPs, worth searching",
                ))
                continue

            # 4. Fallback for small ops
            decisions.append(StrategyDecision(
                spec=spec,
                strategy=KernelStrategy.FALLBACK,
                reason=f"{op_name} too small for custom kernel ({spec.contract.cost.flops} FLOPs)",
            ))

        return decisions


def select_strategies(
    specs: list[KernelSpec], target: TargetProfile
) -> list[StrategyDecision]:
    """Convenience function: select strategies with defaults."""
    selector = KernelSelector(target=target)
    return selector.select(specs)


__all__ = ["KernelSelector", "KernelStrategy", "StrategyDecision", "select_strategies"]
