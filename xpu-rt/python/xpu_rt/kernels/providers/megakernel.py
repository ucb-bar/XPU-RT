"""Megakernel kernel provider.

Adapts the persistent-Triton emitter
(:mod:`compgen.ir.tile.lower_megakernel`) to the
:class:`compgen.kernels.provider.KernelProvider` protocol.

Unlike autocomp / Exo / Triton-template providers, the megakernel provider
does not search a kernel space at the per-op level.  Instead it expects an
``event.graph`` already annotated with a static schedule (via
:class:`compgen.ir.payload.passes.megakernel_static_schedule.StaticMegakernelSchedule`)
and emits the resulting fused persistent kernel as its single search
result.

This keeps ETC integration consistent with CompGen's pluggable-provider
contract while staying faithful to the paper's compiler-driven (not
search-driven) megakernel synthesis model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from compgen.ir.event.ops import GraphOp
from compgen.ir.tile.lower_megakernel import (
    MegakernelLoweringResult,
    lower_megakernel,
)
from compgen.kernels.provider import (
    ContractFeedback,
    KernelContract,
    KnowledgeExport,
    ProviderResult,
    SearchBudget,
)


@dataclass
class MegakernelProvider:
    """Provider that emits persistent megakernel Triton source.

    Attributes:
        graph_lookup: callable mapping a contract's ``region_id`` to the
                      ``event.graph`` op it should lower.  Defaults to
                      always returning ``None`` (no graph found), which
                      makes ``accepts_contract`` answer ``False``.
    """

    graph_lookup: Any = None  # callable[[str], GraphOp | None]
    _last_result: MegakernelLoweringResult | None = field(default=None, init=False)

    @property
    def name(self) -> str:
        return "megakernel"

    def accepts_contract(self, contract: KernelContract) -> bool:
        if self.graph_lookup is None:
            return False
        graph = self.graph_lookup(contract.region_id)
        return isinstance(graph, GraphOp)

    def search(
        self,
        contract: KernelContract,
        budget: SearchBudget,
    ) -> ProviderResult:
        graph: GraphOp | None = self.graph_lookup(contract.region_id) if self.graph_lookup is not None else None
        if graph is None:
            return ProviderResult(
                found=False,
                language="triton",
                metadata={"reason": "no event.graph for region"},
            )
        try:
            lowered = lower_megakernel(graph)
        except ValueError as e:
            return ProviderResult(
                found=False,
                language="triton",
                metadata={"reason": f"lower_megakernel: {e}"},
            )
        self._last_result = lowered
        if not lowered.kernel_source:
            return ProviderResult(
                found=False,
                language="triton",
                metadata={"reason": "lowering returned empty source"},
            )
        plan = (
            f"persistent megakernel '{lowered.kernel_name}' on "
            f"{lowered.launch_config.get('grid', '?')} SMs; "
            f"{sum(len(q) for q in lowered.task_queue.values())} tasks"
        )
        return ProviderResult(
            found=True,
            kernel_code=lowered.kernel_source,
            language="triton",
            correct=False,  # caller must run the differential gate to confirm
            plan=plan,
            iterations_used=1,
            total_candidates=1,
            metadata={
                "kernel_name": lowered.kernel_name,
                "launch_config": lowered.launch_config,
                "event_layout": lowered.event_layout,
                "task_queue": {str(k): v for k, v in lowered.task_queue.items()},
            },
            contract_feedback=[
                ContractFeedback(
                    field="objective",
                    current_value=contract.objective,
                    suggested_value="latency",
                    reason="megakernel synthesis is latency-optimal",
                    measured_gain=0.0,
                ),
            ],
        )

    def export_knowledge(self) -> list[KnowledgeExport]:
        if self._last_result is None or not self._last_result.kernel_source:
            return []
        return [
            KnowledgeExport(
                kind="megakernel_layout",
                scope="region",
                scope_key=self._last_result.kernel_name,
                content=str(self._last_result.event_layout),
                metadata={
                    "task_count": sum(len(q) for q in self._last_result.task_queue.values()),
                    "grid": self._last_result.launch_config.get("grid"),
                },
                confidence=0.8,
            ),
        ]


__all__ = ["MegakernelProvider"]
