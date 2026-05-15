"""Verification gates for LLM invent-slots.

Every gate is a function matching :class:`xpu_rt.llm.registry.InventSlot.gate_impl`:

    (proposal: dict, **ctx) -> dict[str, Any] with "status" in
    {"accepted", "rejected", "deferred"}

Gates compose via :func:`composite` which short-circuits on first
rejection. Per the approved P7/P8 plan, default gate for ported-pass
invent-slots is ``composite(structural, differential)``; SMT
refinement is opt-in via ``ctx["require_smt"] = True``.
"""

from __future__ import annotations

from xpu_rt.agent.gates.composite import composite_gate
from xpu_rt.agent.gates.cost_model import cost_model_gate
from xpu_rt.agent.gates.differential import differential_gate
from xpu_rt.agent.gates.liveness import liveness_gate
from xpu_rt.agent.gates.megakernel import megakernel_persistent_kernel_gate
from xpu_rt.agent.gates.smt_refinement import smt_refinement_gate
from xpu_rt.agent.gates.structural import structural_gate

__all__ = [
    "composite_gate",
    "cost_model_gate",
    "differential_gate",
    "liveness_gate",
    "megakernel_persistent_kernel_gate",
    "smt_refinement_gate",
    "structural_gate",
]
