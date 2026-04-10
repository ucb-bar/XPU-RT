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

from typing import Any

from compgen.kernels.contracts import KernelSpec
from compgen.targets.schema import TargetProfile

# Lazy import to avoid circular dependencies
_ukernel_registry = None


def _get_ukernel_registry():
    """Get or build the default ukernel registry (lazy singleton)."""
    global _ukernel_registry
    if _ukernel_registry is None:
        from compgen.ir.ukernel.builtins import build_default_registry
        _ukernel_registry = build_default_registry()
    return _ukernel_registry


class KernelStrategy(Enum):
    """Strategy for handling a kernel."""

    NATIVE = "native"
    LIBRARY = "library"
    UKERNEL = "ukernel"
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
    llm_client: Any = None  # Optional CompGenLLMProtocol for LLM-guided decisions

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

    def _check_ukernel(self, spec: KernelSpec) -> bool:
        """Check if a registered ukernel can handle this spec."""
        from compgen.ir.ukernel.constraints import ConstraintContext

        registry = _get_ukernel_registry()

        # Map op_name to op_family for ukernel matching
        op_name = spec.contract.op_name
        op_family = op_name.split(".")[-1] if "." in op_name else op_name

        # Build constraint context from spec and target
        shapes: dict[str, int] = {}
        if spec.input_shapes:
            first = spec.input_shapes[0]
            if len(first) >= 2:
                shapes["M"] = first[0]
                shapes["K"] = first[1]
            if len(spec.input_shapes) > 1 and len(spec.input_shapes[1]) >= 2:
                shapes["N"] = spec.input_shapes[1][1]

        # Fallback: estimate shapes from cost if no explicit shapes
        if not shapes and spec.contract.cost.flops > 0:
            # For matmul, flops ≈ 2*M*N*K, use flops as a proxy for "has dimensions"
            shapes["M"] = 1  # Minimal valid shape
            shapes["N"] = 1
            shapes["K"] = 1

        features: set[str] = set()
        for device in self.target.devices:
            for cu in device.compute_units:
                features.add(f"has_{cu.name}")
            for feat in getattr(device, "features", []):
                features.add(f"has_{feat}")

        dtypes = tuple(spec.contract.supported_dtypes) if spec.contract.supported_dtypes else ("float32",)

        context = ConstraintContext(
            shapes=shapes,
            dtypes=dtypes,
            target_features=frozenset(features),
            device_type=self.target.devices[0].device_type if self.target.devices else "",
        )

        return registry.select_ukernel(op_family, context) is not None

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

            # 2.5. Check ukernel registry
            if self._check_ukernel(spec):
                decisions.append(StrategyDecision(
                    spec=spec,
                    strategy=KernelStrategy.UKERNEL,
                    reason=f"{op_name} matched a registered ukernel",
                ))
                continue

            # 3. LLM-guided strategy selection (Unit 6)
            if self.llm_client is not None:
                llm_decision = self._ask_llm_for_strategy(spec, op_name)
                if llm_decision is not None:
                    decisions.append(llm_decision)
                    continue

            # 3b. Check if worth autocomp search (heuristic fallback)
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


    def _ask_llm_for_strategy(self, spec: KernelSpec, op_name: str) -> StrategyDecision | None:
        """Ask LLM to select strategy for a borderline op."""
        try:
            from compgen.agent.prompts.kernel_strategy import KERNEL_STRATEGY_SCHEMA, KernelStrategyContext
            from compgen.agent.prompts.kernel_strategy import format_prompt as fmt_ks
            from compgen.agent.prompts.kernel_strategy import parse_response as parse_ks
            from compgen.llm.base import GenerationRequest, LLMConfig

            op_family = op_name.split(".")[-1] if "." in op_name else op_name
            has_gpu = any(d.device_type == "gpu" for d in self.target.devices)
            ctx = KernelStrategyContext(
                op_name=op_name,
                op_family=op_family,
                flops=spec.contract.cost.flops,
                input_shapes=str(spec.input_shapes),
                output_shapes=str(spec.output_shapes),
                dtype=spec.contract.supported_dtypes[0] if spec.contract.supported_dtypes else "float32",
                target_name=self.target.name,
                has_gpu=has_gpu,
                available_strategies=["native", "library", "ukernel", "autocomp", "fallback"],
            )
            prompt = fmt_ks(ctx)

            from compgen.llm.base import Objective, PromptContext
            model_id = "default"
            try:
                m = getattr(self.llm_client, "model", None)
                if m is not None and isinstance(m, str):
                    model_id = m
            except Exception:
                pass
            request = GenerationRequest(
                prompt_template=prompt,
                context=PromptContext(
                    model_ir_summary="",
                    target_profile_summary=str(self.target.name),
                    available_transforms=[],
                    kernel_contracts=[],
                    objective=Objective.LATENCY,
                ),
                config=LLMConfig(model=model_id, temperature=0.1, max_tokens=600),
            )
            response = self.llm_client.generate_structured(request, KERNEL_STRATEGY_SCHEMA)
            result = parse_ks(response.raw_text)
            if result and result.get("strategy"):
                strategy_name = result["strategy"].upper()
                strategy = KernelStrategy[strategy_name]
                return StrategyDecision(
                    spec=spec,
                    strategy=strategy,
                    reason=f"LLM: {result.get('reason', 'no reason')}",
                )
        except Exception:
            pass
        return None


def select_strategies(
    specs: list[KernelSpec], target: TargetProfile
) -> list[StrategyDecision]:
    """Convenience function: select strategies with defaults."""
    selector = KernelSelector(target=target)
    return selector.select(specs)


__all__ = ["KernelSelector", "KernelStrategy", "StrategyDecision", "select_strategies"]
