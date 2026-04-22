"""End-to-end MCP-driven optimisation — Wave 7.

Wires every W2-W6 primitive together so that *the agent* (Claude Code,
or any MCP client) is the entity actually driving the loop:

  * Dispatch decision      → routed via ``McpDispatchLLM``
  * Per-region kernel codegen → routed via ``McpCodegenFn``
                               (request_kernel_codegen + lookup_cached_kernel)
  * Per-region bench        → routed via ``McpBenchFn``
                               (request_kernel_bench + register_bench_result)
  * Cache + persistence    → ``KernelStore`` + ``KernelDB`` (write-through
                              already lives inside the MCP register tools)
  * Knowledge accumulation  → ``record_lesson`` / ``query_knowledge``
                              MCP tools (W7.2)

There is no Python-level "magic" left — every decision is an MCP tool
call. Tests pre-populate the MCP caches as if Claude Code already
fulfilled the requests; in production Claude Code reads the pending
queues between optimisation passes and calls the register tools.

Public surface:

  * ``optimize_via_mcp(model_fn, target, contracts, sm, session_id, ...)``
    — the headless entry point a pipeline calls.
  * ``request_model_optimization``  /  ``register_optimization_progress``
    — MCP tools so an agent can kick off + monitor an optimisation
    without going through Python at all.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from compgen.agent.hw_aware_dispatch import TargetDispatchDecision
from compgen.agent.kernel_optimizer import (
    CodegenResult,
    OptimizedModel,
    optimize_model,
    optimize_model_multi_target,
)
from compgen.kernels.contract_v3 import (
    HardwareEnvelope,
    KernelContractV3,
)
from compgen.llm.base import Objective
from compgen.mcp.session import SessionManager

# ---------------------------------------------------------------------------
# CodegenFn that round-trips through the MCP kernel tools
# ---------------------------------------------------------------------------


@dataclass
class McpCodegenFn:
    """Routes per-region codegen through MCP.

    Cache hit → returns the agent-supplied source as a callable
    placeholder + the language tag.

    Cache miss → fires ``request_kernel_codegen`` (which queues a
    pending entry the agent can pick up via
    ``list_pending_kernel_requests``) and returns a placeholder
    callable. Subsequent passes hit the cache once the agent has
    fulfilled the request via ``register_kernel_result``.
    """

    sm: SessionManager
    session_id: str

    def __call__(
        self,
        contract: KernelContractV3,
        decision: TargetDispatchDecision,
    ) -> CodegenResult:
        from compgen.agent.kernel_optimizer import _v3_to_fingerprint_dict
        from compgen.mcp.tools.kernel import (
            lookup_cached_kernel,
            request_kernel_codegen,
        )

        contract_dict = _v3_to_fingerprint_dict(contract)
        # Cache lookup first (cheap, no queueing).
        lk = lookup_cached_kernel(
            self.sm,
            session_id=self.session_id,
            contract_v3=contract_dict,
        )
        if lk.get("found"):
            kernel_source = lk.get("kernel_code", "")
            language = lk.get("language", "unknown")
            return CodegenResult(
                callable_kernel=_make_python_callable(kernel_source),
                provider_name=f"mcp_cache:{language}",
                source=kernel_source,
                language=language,
            )

        # Miss → queue + return placeholder.
        request_kernel_codegen(
            self.sm,
            session_id=self.session_id,
            contract_v3=contract_dict,
        )
        return CodegenResult(
            callable_kernel=lambda *a, **kw: None,
            provider_name="mcp_pending",
            source=f"# pending MCP codegen for {contract.op_name}",
            language="python",
        )


def _make_python_callable(source: str) -> Callable[..., Any]:
    """If the kernel source is plain Python that defines a function
    named ``kernel`` or ``main``, exec it and return the function.
    Otherwise return a placeholder no-op callable. Tests + smoke-runs
    use this; production wiring uses the real provider.
    """
    if not source.strip():
        return lambda *a, **kw: None
    try:
        scope: dict[str, Any] = {}
        compiled = compile(source, "<mcp_kernel>", "exec")
        exec(compiled, scope)  # noqa: S102 — agent-supplied kernel
        for name in ("kernel", "main", "run"):
            fn = scope.get(name)
            if callable(fn):
                return fn
    except Exception:  # noqa: BLE001
        pass
    return lambda *a, **kw: None


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def optimize_via_mcp(
    model_fn: Callable[..., Any] | None,
    target: str,
    *,
    contracts: Sequence[KernelContractV3],
    sm: SessionManager,
    session_id: str,
    envelope: HardwareEnvelope | None = None,
    perf_budget_us: float | None = None,
    objective: Objective = Objective.LATENCY,
    capture_graph: bool = True,
    sample_inputs: tuple = (),
) -> OptimizedModel:
    """End-to-end optimisation with every callback routed through MCP.

    Identical signature shape to ``optimize_model`` but with the
    callback slots pre-wired: dispatch decisions, codegen, and bench
    all round-trip through the in-session MCP tools.
    """
    from compgen.mcp.tools.bench import McpBenchFn
    from compgen.mcp.tools.dispatch import McpDispatchLLM

    llm = McpDispatchLLM(
        sm=sm,
        session_id=session_id,
        perf_budget_us=perf_budget_us,
        objective=objective,
    )
    codegen_fn = McpCodegenFn(sm=sm, session_id=session_id)
    bench_fn = McpBenchFn(sm=sm, session_id=session_id)
    return optimize_model(
        model_fn=model_fn,
        target=target,
        contracts=contracts,
        envelope=envelope,
        perf_budget_us=perf_budget_us,
        objective=objective,
        llm=llm,
        codegen_fn=codegen_fn,
        bench_fn=bench_fn,
        capture_graph=capture_graph,
        sample_inputs=sample_inputs,
    )


def optimize_via_mcp_multi_target(
    model_fn: Callable[..., Any] | None,
    targets: Sequence[str],
    *,
    contracts: Sequence[KernelContractV3],
    sm: SessionManager,
    session_id: str,
    envelopes: Sequence[HardwareEnvelope] | None = None,
    perf_budget_us: float | None = None,
    objective: Objective = Objective.LATENCY,
    capture_graph: bool = True,
    sample_inputs: tuple = (),
) -> dict[str, OptimizedModel]:
    """W7.3 — multi-target version of ``optimize_via_mcp``."""
    from compgen.mcp.tools.bench import McpBenchFn
    from compgen.mcp.tools.dispatch import McpDispatchLLM

    llm = McpDispatchLLM(
        sm=sm,
        session_id=session_id,
        perf_budget_us=perf_budget_us,
        objective=objective,
    )
    codegen_fn = McpCodegenFn(sm=sm, session_id=session_id)
    bench_fn = McpBenchFn(sm=sm, session_id=session_id)
    return optimize_model_multi_target(
        model_fn=model_fn,
        targets=targets,
        contracts=contracts,
        envelopes=envelopes,
        perf_budget_us=perf_budget_us,
        objective=objective,
        llm=llm,
        codegen_fn=codegen_fn,
        bench_fn=bench_fn,
        capture_graph=capture_graph,
        sample_inputs=sample_inputs,
    )


# ---------------------------------------------------------------------------
# MCP tool surface — agent kicks off optimisation
# ---------------------------------------------------------------------------


@dataclass
class _OptimizationProgress:
    """Per-session state for ``request_model_optimization``."""

    contracts_seen: list[str] = field(default_factory=list)
    completed_passes: int = 0
    last_summary: str = ""


def _progress(session) -> _OptimizationProgress:
    cur: _OptimizationProgress | None = getattr(session, "optim_progress", None)
    if cur is None:
        cur = _OptimizationProgress()
        session.optim_progress = cur  # type: ignore[attr-defined]
    return cur


def request_model_optimization(
    sm: SessionManager,
    *,
    session_id: str,
    target: str,
    contract_fingerprints: list[str],
    perf_budget_us: float | None = None,
    objective: str = "latency",
) -> dict[str, Any]:
    """Kick off / advance an optimisation pass.

    The actual contracts must already have been registered with the
    session (typically via ``open_target`` → ``load_model``). This tool
    is a thin progress-tracker — it records the contract fingerprints
    the orchestration cares about and returns the count of pending
    codegen / bench / dispatch requests so the agent knows what to
    fulfil next.
    """
    session = sm.get(session_id)
    progress = _progress(session)
    for fp in contract_fingerprints:
        if fp not in progress.contracts_seen:
            progress.contracts_seen.append(fp)
    # Surface the queues the agent may want to drain (deferred imports
    # to avoid import-cycle with compgen.mcp.tools).
    from compgen.mcp.tools.bench import list_pending_bench_requests
    from compgen.mcp.tools.dispatch import list_pending_dispatch_decisions
    from compgen.mcp.tools.kernel import list_pending_kernel_requests

    pending_codegen = list_pending_kernel_requests(sm, session_id=session_id)
    pending_dispatch = list_pending_dispatch_decisions(sm, session_id=session_id)
    pending_bench = list_pending_bench_requests(sm, session_id=session_id)
    return {
        "ok": True,
        "session_id": session_id,
        "target": target,
        "objective": objective,
        "perf_budget_us": perf_budget_us,
        "contracts_tracked": len(progress.contracts_seen),
        "pending": {
            "codegen": pending_codegen.get("pending_count", 0),
            "dispatch": pending_dispatch.get("pending_count", 0),
            "bench": pending_bench.get("pending_count", 0),
        },
        "next_pass_hint": (
            "Drain pending queues (request → register loops), then call "
            "request_model_optimization again to continue. When all "
            "queues are empty AND every contract has a cached kernel + "
            "perf, the optimisation has converged."
        ),
    }


def register_optimization_progress(
    sm: SessionManager,
    *,
    session_id: str,
    summary: str,
    completed_passes: int | None = None,
) -> dict[str, Any]:
    """Agent reports a summary line + (optionally) the pass count."""
    session = sm.get(session_id)
    progress = _progress(session)
    progress.last_summary = summary
    if completed_passes is not None:
        progress.completed_passes = int(completed_passes)
    return {
        "ok": True,
        "session_id": session_id,
        "completed_passes": progress.completed_passes,
        "last_summary": progress.last_summary,
    }


OPTIMIZE_TOOLS: list[dict[str, Any]] = [
    {
        "name": "request_model_optimization",
        "description": (
            "Kick off / advance the W6 optimisation loop. Records the "
            "contracts being tracked and surfaces the count of pending "
            "codegen/dispatch/bench requests for the agent to drain."
        ),
        "phase": "transform",
        "handler": request_model_optimization,
        "input_schema": {
            "type": "object",
            "properties": {
                "session_id": {"type": "string"},
                "target": {"type": "string"},
                "contract_fingerprints": {"type": "array", "items": {"type": "string"}},
                "perf_budget_us": {"type": ["number", "null"]},
                "objective": {"type": "string"},
            },
            "required": ["session_id", "target", "contract_fingerprints"],
        },
    },
    {
        "name": "register_optimization_progress",
        "description": "Agent posts a short summary of optimisation progress.",
        "phase": "transform",
        "handler": register_optimization_progress,
        "input_schema": {
            "type": "object",
            "properties": {
                "session_id": {"type": "string"},
                "summary": {"type": "string"},
                "completed_passes": {"type": ["integer", "null"]},
            },
            "required": ["session_id", "summary"],
        },
    },
]


__all__ = [
    "McpCodegenFn",
    "OPTIMIZE_TOOLS",
    "optimize_via_mcp",
    "optimize_via_mcp_multi_target",
    "register_optimization_progress",
    "request_model_optimization",
]
