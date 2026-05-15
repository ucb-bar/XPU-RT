"""Autocomp kernel provider — wraps autocomp_adapter.py as a KernelProvider.

Implements the two-phase plan→code loop from the Autocomp paper,
with knowledge export and contract feedback for CompGen's memory.

Honest scope (Phase 4 production-grade): the autocomp search requires
three environmental preconditions — a working ``autocomp`` install, a
Google API key (for the CUDA LLM agent), and a compatible GPU runner.
When any of these is missing, this provider raises
:class:`~compgen.kernels.errors.UnmeasurableKernelError` with a
specific reason rather than silently returning ``found=False``. The
previous placeholder ``return ProviderResult(found=False)`` at
the end of ``_search_with_adapter`` is a known bug: it let selectors
believe autocomp was "tried and nothing found" when in reality
autocomp was never invoked. That path has been replaced with a typed
error so the escalating router can fall through to the next provider
on a real signal.
"""

from __future__ import annotations

from typing import Any

import structlog

from compgen.kernels.errors import UnmeasurableKernelError
from compgen.kernels.provider import (
    KernelContract,
    KnowledgeExport,
    ProviderResult,
    SearchBudget,
)

log = structlog.get_logger()


class AutocompProvider:
    """Wraps the Autocomp adapter as a KernelProvider.

    Delegates to ``compgen.kernels.autocomp_adapter`` for the actual
    search, then extracts knowledge and contract feedback from
    results. Unavailable-environment paths raise
    :class:`UnmeasurableKernelError` — the escalating router's
    ``accepts_contract`` check gates whether this provider is tried
    at all, so raising on unavailability is the right signal.
    """

    # Provider preference (M-91a follow-up, 2026-05-15): autocomp is
    # the fallback CUDA kernel-search provider. KernelBlaster
    # (priority=90) wins applicability/auction ordering when both
    # accept a contract — see KernelBlasterProvider for the rationale.
    priority: int = 80

    def __init__(self) -> None:
        self._accumulated_knowledge: list[KnowledgeExport] = []

    @property
    def name(self) -> str:
        return "autocomp"

    def accepts_contract(self, contract: KernelContract) -> bool:
        """Autocomp handles GPU kernels (Triton/CUDA).

        Returns False when:
        - The target is not GPU-ish (accelerator, ukernel-runtime).
        - The autocomp library is not installed.
        - No Google API key is available.
        - No CUDA device is available.

        Returning False here is the clean way to keep this provider
        out of the escalation chain when it can't work; actually
        *running* search on an unsupported contract raises
        :class:`UnmeasurableKernelError`.
        """
        target = contract.target_name.lower()
        hardware = contract.hardware_key.lower()
        gpu_ish = any(
            kw in target or kw in hardware for kw in ["gpu", "cuda", "triton", "h100", "a100", "hopper", "ampere"]
        )
        if not gpu_ish:
            return False
        return self._environment_ready() is None

    def search(self, contract: KernelContract, budget: SearchBudget) -> ProviderResult:
        """Search using Autocomp's beam search.

        Wraps the existing AutocompAdapter.search_kernel() and
        translates results into the provider protocol. Raises
        :class:`UnmeasurableKernelError` when the environment is not
        set up for autocomp; returns a real :class:`ProviderResult`
        on success with measured latency recorded by the adapter.
        """
        env_problem = self._environment_ready()
        if env_problem is not None:
            raise UnmeasurableKernelError(f"autocomp unavailable: {env_problem}")
        from compgen.kernels.autocomp_adapter import AutocompAdapter

        adapter = AutocompAdapter(max_iterations=budget.max_iterations)
        return self._search_with_adapter(adapter, contract, budget)

    def _environment_ready(self) -> str | None:
        """Return None when autocomp can run, else a reason string."""
        # Library import is the first precondition.
        try:
            import autocomp  # noqa: F401
        except ImportError:
            return "autocomp library not installed"

        # API key — CudaLLMAgent needs GOOGLE_API_KEY.
        import os

        if not os.environ.get("GOOGLE_API_KEY"):
            return "GOOGLE_API_KEY not set (required by CudaLLMAgent)"

        # CUDA.
        try:
            import torch

            if not torch.cuda.is_available():
                return "no CUDA device available"
        except Exception as exc:
            return f"torch.cuda probe raised: {exc!r}"

        return None

    def _search_with_adapter(
        self,
        adapter: Any,
        contract: KernelContract,
        budget: SearchBudget,
    ) -> ProviderResult:
        """Run the real autocomp adapter and translate to ProviderResult.

        Requires a :class:`PatternCluster` built from the contract.
        When we can't build one, raise — don't pretend the search
        found nothing.
        """
        from compgen.agent.analyzer import PatternCluster
        from compgen.targets.schema import TargetProfile

        # Derive total_flops + total_bytes honestly from the contract
        # shapes; the adapter uses these for cost-context strings.
        def _shape_elems(shape: tuple[int, ...]) -> int:
            n = 1
            for d in shape:
                n *= max(int(d), 1)
            return n

        in_elems = sum(_shape_elems(s) for s in contract.input_shapes)
        out_elems = sum(_shape_elems(s) for s in contract.output_shapes)
        bytes_per_elem = 2 if "f16" in contract.dtypes or "bf16" in contract.dtypes else 4
        total_bytes = (in_elems + out_elems) * bytes_per_elem
        # Conservative flops estimate: 2 * largest matmul-like dim^3.
        largest = max(
            (_shape_elems(s) for s in contract.input_shapes + contract.output_shapes),
            default=0,
        )
        total_flops = 2 * largest

        try:
            cluster = PatternCluster(
                cluster_id=contract.region_id or contract.op_family or "autocomp_search",
                pattern_type=contract.op_family or "custom",
                node_names=tuple(),
                total_flops=total_flops,
                total_bytes=total_bytes,
                arithmetic_intensity=(total_flops / total_bytes) if total_bytes else 0.0,
                estimated_latency_per_device={contract.target_name or "cuda-default": 0.0},
                best_device=contract.target_name or "cuda-default",
                is_bottleneck=True,
                kernel_opportunity=contract.op_family or "custom",
                input_shapes={f"in_{i}": tuple(s) for i, s in enumerate(contract.input_shapes)},
                output_shapes={f"out_{i}": tuple(s) for i, s in enumerate(contract.output_shapes)},
            )
        except Exception as exc:
            raise UnmeasurableKernelError(f"autocomp cannot build PatternCluster from contract: {exc!r}") from exc

        target = TargetProfile(name=contract.target_name or "cuda-default")
        result = adapter.search_kernel(cluster, target, budget=budget.max_iterations)

        knowledge = [
            KnowledgeExport(
                kind="autocomp_search_result",
                scope="operator_family",
                scope_key=contract.op_family or "",
                content=f"autocomp found kernel for {contract.op_family!r}",
                metadata={
                    "target": contract.target_name,
                    "correct": bool(result.correct),
                    "latency_us": float(result.latency_us),
                    "speedup_vs_baseline": float(result.speedup_vs_baseline),
                },
                confidence=0.9 if result.correct else 0.4,
            )
        ]
        self._accumulated_knowledge.extend(knowledge)

        return ProviderResult(
            found=bool(result.correct),
            kernel_code=result.kernel_code or "",
            language=result.language or "cuda",
            latency_us=float(result.latency_us),
            correct=bool(result.correct),
            plan=result.plan or "",
            speedup=float(result.speedup_vs_baseline),
            iterations_used=int(result.iterations_used),
            total_candidates=int(result.total_candidates),
            knowledge_exports=knowledge,
            contract_feedback=[],
            metadata={
                "provider": "autocomp",
                "cost_source": "measured_gpu" if result.correct else "unmeasured",
            },
        )

    def export_knowledge(self) -> list[KnowledgeExport]:
        """Export accumulated knowledge from all searches."""
        exports = list(self._accumulated_knowledge)
        self._accumulated_knowledge.clear()
        return exports


class ExoProvider:
    """Wraps the Exo schedule agent as a KernelProvider.

    Accepts accelerator-class contracts and dispatches via
    :class:`~compgen.kernels.exo_adapter.ExoAdapter`. Like autocomp,
    raises :class:`UnmeasurableKernelError` when Exo isn't available
    rather than silently returning ``found=False``.
    """

    def __init__(self) -> None:
        self._accumulated_knowledge: list[KnowledgeExport] = []

    @property
    def name(self) -> str:
        return "exo"

    def accepts_contract(self, contract: KernelContract) -> bool:
        target = contract.target_name.lower()
        hardware = contract.hardware_key.lower()
        if not any(kw in target or kw in hardware for kw in ["gemmini", "exo", "snax", "accel"]):
            return False
        # Don't offer escalation if we can't run.
        try:
            import exo  # noqa: F401
        except ImportError:
            return False
        return True

    def search(self, contract: KernelContract, budget: SearchBudget) -> ProviderResult:
        try:
            import exo  # noqa: F401
        except ImportError as exc:
            raise UnmeasurableKernelError(f"Exo library not installed: {exc!r}") from exc

        from compgen.kernels.exo_adapter import ExoAdapter

        adapter = ExoAdapter(target_name=contract.target_name or "generic")
        # Only the 2D-input `matmul` / `conv2d` style contracts currently
        # have a seed generator. The adapter returns None when it
        # doesn't know how to seed; surface that as a typed error.
        result = adapter.search_kernel(
            op_name=contract.op_family or "unknown",
            input_shapes=list(contract.input_shapes),
            output_shapes=list(contract.output_shapes),
            dtype=(contract.dtypes[0] if contract.dtypes else "f32"),
            search_budget=budget.max_iterations,
        )
        if result is None:
            raise UnmeasurableKernelError(f"Exo has no seed generator for op_family={contract.op_family!r}")
        import math

        return ProviderResult(
            found=bool(result.correct),
            kernel_code=result.scheduled_code or result.proc_code,
            language="exo",
            latency_us=float(result.latency_us) if math.isfinite(result.latency_us) else float("nan"),
            correct=bool(result.correct),
            metadata={
                "provider": "exo",
                "proc_code": result.proc_code,
                "c_code": result.c_code,
                "schedule_ops_applied": int(result.schedule_ops_applied),
                "cost_source": ("measured_cpu" if math.isfinite(result.latency_us) else "unmeasured"),
            },
        )

    def export_knowledge(self) -> list[KnowledgeExport]:
        exports = list(self._accumulated_knowledge)
        self._accumulated_knowledge.clear()
        return exports


__all__ = ["AutocompProvider", "ExoProvider"]
