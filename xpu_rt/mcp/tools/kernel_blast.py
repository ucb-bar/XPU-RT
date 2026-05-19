"""MCP tools for KernelBlaster v2.

Three workflows:

* **Inspect** (``xpu_rt_blast_lessons_for_region``,
  ``xpu_rt_blast_strategies_for_target``) — read-only queries against
  the target card's lessons.jsonl and strategies.json so the agent can
  surface "what we know about this state" to the user before proposing.
* **Agent-in-loop propose** (``xpu_rt_blast_prepare_propose`` +
  ``xpu_rt_blast_apply_response``) — the in-loop Claude Code session
  produces the kernel itself; the tool builds the prompt bundle (with
  card context, lessons, strategies, prior attempts) and folds the
  agent's response back into the loop.
* **Headless run** (``xpu_rt_blast_run_headless``) — drives the full
  agent loop with the Gemini generator and the mock evaluator. Gated
  by the $100 Gemini budget cap.
"""

from __future__ import annotations

import logging
from dataclasses import asdict
from typing import Any

from xpu_rt.kernels.kernelblaster_v2 import (
    AgentLoopConfig,
    KernelBlasterV2,
    KernelGeneratorLLM,
    KernelGeneratorMock,
    ProposeResponse,
    StateVector,
    derive_state,
)
from xpu_rt.kernels.kernelblaster_v2.evaluators import EvaluationReport, MockEvaluator
from xpu_rt.kernels.kernelblaster_v2.prompt_builder import (
    GENERATOR_RESPONSE_SCHEMA,
    GENERATOR_SYSTEM_PROMPT,
    PromptBuilder,
)
from xpu_rt.kernels.kernelblaster_v2.strategy_db import StrategyDB
from xpu_rt.kernels.provider import KernelContract
from xpu_rt.memory import target_knowledge as tk

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Contract helpers
# ---------------------------------------------------------------------------


def _contract_from_payload(payload: dict[str, Any]) -> KernelContract:
    """Build a KernelContract from a JSON-friendly dict.

    The MCP envelope passes shapes as nested lists; tuples are restored
    here so the contract is immutable and consistent with the rest of
    the codebase.
    """
    return KernelContract(
        region_id=str(payload.get("region_id", "")),
        op_family=str(payload.get("op_family", "")),
        input_shapes=tuple(tuple(s) for s in payload.get("input_shapes", ())),
        output_shapes=tuple(tuple(s) for s in payload.get("output_shapes", ())),
        dtypes=tuple(payload.get("dtypes", ())),
        layout=str(payload.get("layout", "row_major")),
        target_name=str(payload.get("target_name", "")),
        hardware_key=str(payload.get("hardware_key", "")),
        objective=str(payload.get("objective", "latency")),
        constraints=dict(payload.get("constraints", {})),
        provider_hints=dict(payload.get("provider_hints", {})),
    )


# ---------------------------------------------------------------------------
# Read-only inspectors
# ---------------------------------------------------------------------------


def xpu_rt_blast_lessons_for_region(
    sm: Any,
    *,
    contract: dict[str, Any],
    limit: int = 10,
    **kwargs: Any,
) -> dict[str, Any]:
    """Return lessons matching the contract's derived state."""
    contract_obj = _contract_from_payload(contract)
    if not tk.exists(contract_obj.target_name):
        return {"ok": False, "error": f"no card for target_id={contract_obj.target_name!r}"}
    card = tk.load(contract_obj.target_name)
    state = derive_state(contract_obj, target_id=card.target_id)
    rows = []
    for lesson in tk.iter_lessons(card):
        if (
            lesson.op_family == state.op_family
            and lesson.dtype_class == state.dtype_class
            and lesson.layout_kind == state.layout_kind
            and lesson.archetype == state.archetype
        ):
            rows.append(asdict(lesson))
    rows.sort(key=lambda r: r["timestamp"], reverse=True)
    return {
        "ok": True,
        "state": state.to_dict(),
        "lessons": rows[: max(limit, 1)],
    }


def xpu_rt_blast_strategies_for_target(
    sm: Any,
    *,
    target_id: str,
    op_family: str = "",
    limit: int = 20,
    **kwargs: Any,
) -> dict[str, Any]:
    """Return the strategy DB for ``target_id``, optionally filtered."""
    if not tk.exists(target_id):
        return {"ok": False, "error": f"no card for target_id={target_id!r}"}
    card = tk.load(target_id)
    db = StrategyDB.for_card(card)
    rows = []
    for row in db.rows.values():
        if op_family and op_family not in row.state_key:
            continue
        rows.append(row.to_dict())
    rows.sort(key=lambda r: (-r["mean_speedup"], r["state_key"]))
    return {"ok": True, "target_id": target_id, "rows": rows[: max(limit, 1)]}


# ---------------------------------------------------------------------------
# Agent-in-loop propose / apply
# ---------------------------------------------------------------------------


def xpu_rt_blast_prepare_propose(
    sm: Any,
    *,
    contract: dict[str, Any],
    prior_attempts: list[dict[str, Any]] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Build the propose-request prompt bundle for one attempt.

    The agent picks up the returned ``system`` + ``user`` + ``schema``,
    produces a JSON response matching the schema, and then calls
    :func:`xpu_rt_blast_apply_response` to feed the response back along
    with the evaluator's score.
    """
    contract_obj = _contract_from_payload(contract)
    if not tk.exists(contract_obj.target_name):
        return {
            "ok": False,
            "error": f"no card for target_id={contract_obj.target_name!r}; "
            f"run /xpu-rt-target first",
        }
    card = tk.load(contract_obj.target_name)
    state = derive_state(contract_obj, target_id=card.target_id)
    db = StrategyDB.for_card(card)
    bundle = PromptBuilder(card=card, strategy_db=db).build(
        contract=contract_obj,
        state=state,
        prior_attempts=tuple(prior_attempts or ()),
    )
    return {
        "ok": True,
        "state": state.to_dict(),
        "card_target_id": card.target_id,
        "system": bundle.system,
        "user": bundle.user,
        "schema": bundle.schema,
        "metadata": bundle.metadata,
    }


def xpu_rt_blast_apply_response(
    sm: Any,
    *,
    contract: dict[str, Any],
    response: dict[str, Any],
    evaluation: dict[str, Any],
    **kwargs: Any,
) -> dict[str, Any]:
    """Fold one agent-produced (response, evaluation) pair into the card.

    Records the strategy outcome and writes a lesson row when the
    evaluation reports ``correct=True``. Returns the updated state so
    the agent can decide whether to iterate.
    """
    contract_obj = _contract_from_payload(contract)
    if not tk.exists(contract_obj.target_name):
        return {"ok": False, "error": f"no card for target_id={contract_obj.target_name!r}"}
    card = tk.load(contract_obj.target_name)
    state = derive_state(contract_obj, target_id=card.target_id)

    action = str(response.get("action") or "(unknown)")
    correct = bool(evaluation.get("correct", False))
    speedup = float(evaluation.get("score", 0.0))
    notes = str(evaluation.get("diff_summary", ""))

    db = StrategyDB.for_card(card)
    entry = db.record(state=state, action=action, accepted=correct, speedup=speedup, notes=notes)

    lesson_written = False
    if correct and action and action != "(unknown)":
        from xpu_rt.kernels.kernelblaster_v2.lesson_writer import LessonWriter

        LessonWriter(card=card).write(
            state=state,
            action=action,
            measured_gain=speedup,
            notes=notes,
        )
        lesson_written = True

    return {
        "ok": True,
        "state": state.to_dict(),
        "strategy_row": entry.to_dict(),
        "lesson_written": lesson_written,
    }


# ---------------------------------------------------------------------------
# Headless run
# ---------------------------------------------------------------------------


def xpu_rt_blast_run_headless(
    sm: Any,
    *,
    contract: dict[str, Any],
    model: str = "gemini-2.5-flash",
    max_iterations: int = 4,
    accept_threshold: float = 1.0,
    **kwargs: Any,
) -> dict[str, Any]:
    """Drive the full agent loop synchronously via Gemini.

    Uses :class:`MockEvaluator` until real evaluators land in task
    #10 — the goal of this tool today is to surface the prompt /
    response flow end-to-end so users can see KB v2 working without
    needing Claude Code in the loop.
    """
    contract_obj = _contract_from_payload(contract)
    if not tk.exists(contract_obj.target_name):
        return {"ok": False, "error": f"no card for target_id={contract_obj.target_name!r}"}
    card = tk.load(contract_obj.target_name)
    loop = KernelBlasterV2(
        card=card,
        generator=KernelGeneratorLLM(model=model),
        evaluator=MockEvaluator(table=lambda c: EvaluationReport(correct=True, score=1.0)),
        config=AgentLoopConfig(max_iterations=max_iterations, accept_threshold=accept_threshold),
    )
    try:
        result = loop.run(contract_obj)
    except Exception as exc:  # noqa: BLE001
        logger.exception("xpu_rt_blast_run_headless failed")
        return {"ok": False, "error": f"{exc.__class__.__name__}: {exc}"}
    provider = result.to_provider_result()
    return {
        "ok": True,
        "found": provider.found,
        "plan": provider.plan,
        "kernel_code": provider.kernel_code,
        "iterations_used": provider.iterations_used,
        "metadata": provider.metadata,
    }


# ---------------------------------------------------------------------------
# Tool registration
# ---------------------------------------------------------------------------


KERNEL_BLAST_TOOLS: list[dict[str, Any]] = [
    {
        "name": "xpu_rt_blast_lessons_for_region",
        "description": "Return lessons that match a contract's derived state.",
        "phase": "inspect",
        "handler": xpu_rt_blast_lessons_for_region,
        "input_schema": {
            "type": "object",
            "properties": {
                "contract": {"type": "object"},
                "limit": {"type": "integer", "default": 10},
            },
            "required": ["contract"],
        },
    },
    {
        "name": "xpu_rt_blast_strategies_for_target",
        "description": "Return the strategy DB for one target, optionally filtered by op family.",
        "phase": "inspect",
        "handler": xpu_rt_blast_strategies_for_target,
        "input_schema": {
            "type": "object",
            "properties": {
                "target_id": {"type": "string"},
                "op_family": {"type": "string"},
                "limit": {"type": "integer", "default": 20},
            },
            "required": ["target_id"],
        },
    },
    {
        "name": "xpu_rt_blast_prepare_propose",
        "description": (
            "Build the propose-request prompt bundle (system + user + JSON "
            "schema) for one KB v2 attempt, given a contract and the prior "
            "attempts in this run."
        ),
        "phase": "lifecycle",
        "handler": xpu_rt_blast_prepare_propose,
        "input_schema": {
            "type": "object",
            "properties": {
                "contract": {"type": "object"},
                "prior_attempts": {"type": "array"},
            },
            "required": ["contract"],
        },
    },
    {
        "name": "xpu_rt_blast_apply_response",
        "description": (
            "Fold one agent-produced (response, evaluation) pair into the "
            "target card: record strategy stats, append a lesson on accept."
        ),
        "phase": "lifecycle",
        "handler": xpu_rt_blast_apply_response,
        "input_schema": {
            "type": "object",
            "properties": {
                "contract": {"type": "object"},
                "response": {"type": "object"},
                "evaluation": {"type": "object"},
            },
            "required": ["contract", "response", "evaluation"],
        },
    },
    {
        "name": "xpu_rt_blast_run_headless",
        "description": (
            "Drive the full KB v2 agent loop synchronously via Gemini. "
            "Respects the configured cumulative-USD cap. Mock evaluator "
            "today; real evaluators land with task #10."
        ),
        "phase": "lifecycle",
        "handler": xpu_rt_blast_run_headless,
        "input_schema": {
            "type": "object",
            "properties": {
                "contract": {"type": "object"},
                "model": {"type": "string", "default": "gemini-2.5-flash"},
                "max_iterations": {"type": "integer", "default": 4},
                "accept_threshold": {"type": "number", "default": 1.0},
            },
            "required": ["contract"],
        },
    },
]


__all__ = ["KERNEL_BLAST_TOOLS"]
