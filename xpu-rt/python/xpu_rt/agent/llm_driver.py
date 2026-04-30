"""Thin LLM-driven compilation orchestrator.

``LLMDrivenCompiler`` is the shared backbone that sits behind both the
Python ``compile_with_llm`` entry (see :mod:`compgen.api_llm`) and the
MCP server tools (see :mod:`compgen.mcp`). It delegates real work to
:class:`~compgen.agent.loop.AgenticCompilationLoop` (``_proposal_to_action``,
``_run_per_step_verification``) and :class:`~compgen.agent.env.CompilerEnv`;
it does NOT reimplement the loop.

Responsibilities:

* Own one :class:`CompilerEnv` + recipe-tracking IR per session.
* Wire an :class:`LLMRecorder` + :class:`ToolCallRecorder` so every
  invocation is auditable / replayable.
* Dispatch LLM tool / invent-slot calls by looking them up in the
  registry (:func:`compgen.llm.registry.get_registry`) and, when they
  map to an env action, translating via
  :meth:`AgenticCompilationLoop._proposal_to_action`.
* Expose a token-efficient ``current_view()`` (Recipe-IR) and
  ``diff_since(checkpoint_id)`` so LLM callers can reason without
  re-sending the full IR.
* Keep a small checkpoint history so LLM turns can branch or revert.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog

from compgen.agent.env import (
    Action,
    CompilerEnv,
    NoopAction,
)
from compgen.agent.gates import composite_gate, differential_gate, structural_gate
from compgen.agent.loop import AgenticCompilationLoop
from compgen.agent.prompts.analyze import ProposedOptimization
from compgen.ir.recipe.llm_view import diff_views, recipe_to_llm_view
from compgen.llm.base import CompGenLLMProtocol
from compgen.llm.recorder import LLMRecorder, ToolCallRecorder
from compgen.llm.registry import InventSlot, Registry, Tool, get_registry
from compgen.targets.schema import TargetProfile

log = structlog.get_logger()


# Every non-accepted invocation is recorded; the driver is intentionally
# permissive about what the LLM can try — the gates / env enforce safety.


@dataclass
class DriverCheckpoint:
    """A snapshot of the Recipe-IR view at a moment in time.

    Stored in :attr:`LLMDrivenCompiler._checkpoints`, keyed by a short
    id we hand back to the LLM so it can name a prior state (e.g.
    ``diff_since("ckpt_3")``).
    """

    ckpt_id: str
    view: dict[str, Any]
    step_index: int


@dataclass
class DriverStepResult:
    """What a single :meth:`LLMDrivenCompiler.step` call produced.

    The shape is intentionally JSON-friendly so the MCP server can
    return it verbatim. ``ir_view_after`` is *only* present when the
    call was successfully applied — otherwise callers should use
    ``current_view()``.
    """

    status: str  # accepted | rejected | deferred | applied | failed
    kind: str  # tool | invent | proposal
    name: str
    ir_hash_before: str
    ir_hash_after: str
    elapsed_ms: int
    summary: str = ""
    gate_result: dict[str, Any] | None = None
    tool_result: dict[str, Any] | None = None
    diagnostics: tuple[str, ...] = ()
    remediation_hint: str | None = None
    ir_view_after: dict[str, Any] | None = None  # filled only on mutation


@dataclass
class LLMDrivenCompiler:
    """Per-session orchestrator for LLM-driven compilation.

    The caller is responsible for preparing a :class:`CompilerEnv` that
    has already been reset with a module + target (usually via
    :meth:`compgen.api.CompiledModel.create_agent_env`). After
    construction callers drive the session with :meth:`step_tool`,
    :meth:`step_invent`, :meth:`step_proposal`, and
    :meth:`run_per_step_verification`.

    Attributes:
        env: The live :class:`CompilerEnv`.
        target: The hardware target profile.
        llm_client: The backend LLM — usually already wrapped in
            :class:`LLMRecorder`. The driver will wrap it if not.
        transcript_dir: Where to write the ToolCallRecorder JSONL.
            Defaults to ``~/.compgen/transcripts``.
        budget: Soft cap on accepted LLM-driven steps.
        registry: The :class:`Registry` to dispatch tools + slots
            against. Defaults to the process-wide registry.
        max_view_ops: Cap on ops in Recipe-IR views returned to the LLM.
    """

    env: CompilerEnv
    target: TargetProfile
    llm_client: CompGenLLMProtocol
    transcript_dir: Path | None = None
    budget: int = 10
    registry: Registry | None = None
    max_view_ops: int = 80

    # internal state
    _session_id: str = field(default="", init=False)
    _llm_recorder: LLMRecorder | None = field(default=None, init=False)
    _tool_recorder: Any | None = field(default=None, init=False)
    _loop: AgenticCompilationLoop | None = field(default=None, init=False)
    _checkpoints: dict[str, DriverCheckpoint] = field(default_factory=dict, init=False)
    _step_index: int = field(default=0, init=False)
    _accepted_steps: int = field(default=0, init=False)
    _last_view: dict[str, Any] | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        # Pull model/target from the env so the session id is
        # descriptive when available. The env may not have been reset
        # yet; both lookups are best-effort.
        _env = getattr(self, "env", None)
        _model = getattr(_env, "_pytorch_model", None) if _env is not None else None
        _target = getattr(_env, "_target", None) if _env is not None else None
        from compgen.trace.session_id import build_session_id as _build_sid

        self._session_id = _build_sid(
            model=_model,
            # env stores a TargetProfile directly — wrap it in a
            # structural proxy so build_session_id's profile lookup
            # still works.
            target_device=type("_TD", (), {"profile": _target})() if _target is not None else None,
            prefix="drv",
        )
        if self.registry is None:
            self.registry = get_registry()
        # Ensure canonical invent slots are registered so an MCP-driven
        # agent that calls propose_invent_slot fresh actually finds them.
        # Idempotent — second + later calls return early on each slot.
        try:
            from compgen.agent.invent_slots.registrar import register_invent_slots

            register_invent_slots(self.registry)
        except Exception:  # noqa: BLE001
            # Never block driver init on slot registration; the agent
            # will see status=unknown if it tries an unregistered slot
            # and can read the remediation hint we surface below.
            pass

        # Resolve transcript dir. Honour COMPGEN_SESSION_DIR env var so
        # tests can redirect writes away from ~/.compgen.
        if self.transcript_dir is None:
            import os

            env = os.environ.get("COMPGEN_SESSION_DIR")
            if env:
                self.transcript_dir = Path(env).expanduser()
            else:
                self.transcript_dir = Path("~/.compgen/transcripts").expanduser()
        self.transcript_dir.mkdir(parents=True, exist_ok=True)

        # Install a trace bus for this session so every LLM call, tool call,
        # pass, analysis and decision lands in a single correlated JSONL.
        # If a bus has already been installed (e.g. by ``api.compile_model``)
        # we reuse it; otherwise the trace lives under the session dir.
        # Local imports to avoid a circular dep: ``compgen.trace.adapters``
        # imports :class:`McpTranscriptRecorder`, whose package transitively
        # imports this module.
        from compgen.trace import (
            TracingLLMRecorder,
            TracingToolCallRecorder,
            get_active_bus,
            install_bus,
        )

        bus = get_active_bus()
        if bus is None:
            bus = install_bus(
                output_dir=self.transcript_dir / self._session_id,
                session_id=self._session_id,
            )

        # Wire recorders. If the caller passed a bare client, wrap it.
        if isinstance(self.llm_client, LLMRecorder):
            self._llm_recorder = self.llm_client
        else:
            self._llm_recorder = LLMRecorder(
                wrapped=self.llm_client,
                log_dir=self.transcript_dir / self._session_id / "llm",
                enabled=True,
            )

        # Route env + loop through the tracing wrapper so every call
        # emits an llm_prompt / llm_response trace event. ``wrap`` is
        # idempotent — a second call on an already-wrapped recorder
        # returns it unchanged.
        self.llm_client = TracingLLMRecorder.wrap(self._llm_recorder, bus)  # type: ignore[assignment]

        raw_tool_recorder = ToolCallRecorder(
            log_path=self.transcript_dir / self._session_id / "tools.jsonl",
            enabled=True,
        )
        self._tool_recorder = TracingToolCallRecorder.wrap(raw_tool_recorder, bus)

        # Turn on recipe tracking so we can serve LLM views + diffs.
        if self.env.recipe is None:
            try:
                self.env.enable_recipe_tracking()
            except Exception as e:  # noqa: BLE001
                log.debug("llm_driver.recipe_tracking_unavailable", error=str(e))

        # Create the backing loop (we only call its translation + per-step
        # verification helpers; we never run its top-level run()).
        self._loop = AgenticCompilationLoop(
            llm_client=self.llm_client,
            env=self.env,
            budget=self.budget,
        )
        self.env.attach_llm_client(self.llm_client)

        # Seed checkpoint 0 so diff_since("ckpt_0") is always valid.
        # Capture at full breadth so later diffs see every op even when
        # the recipe is much larger than the default max_view_ops cap.
        self._last_view = self._compute_view()
        self._checkpoints["ckpt_0"] = DriverCheckpoint(
            ckpt_id="ckpt_0",
            view=self._compute_view(max_ops=100_000),
            step_index=0,
        )

    # ------------------------------------------------------------------
    # Views + introspection
    # ------------------------------------------------------------------

    def current_view(
        self,
        *,
        focus: str | None = None,
        max_ops: int | None = None,
    ) -> dict[str, Any]:
        """Return the token-efficient Recipe-IR view of the current session.

        Args:
            focus: Optional ``op_id`` from a prior view — triggers a
                verbatim inline of the named op plus its neighbours.
            max_ops: Override the session cap.
        """
        return self._compute_view(focus=focus, max_ops=max_ops)

    def diff_since(self, ckpt_id: str) -> dict[str, Any]:
        """Diff the current Recipe-IR view against a named checkpoint.

        Returns an empty diff with ``status="unknown_checkpoint"`` if
        ``ckpt_id`` doesn't exist, which is stable for MCP responses.

        Both views are computed with an effectively unlimited cap
        (``max_ops=100_000``) so the diff always sees every op — even
        an agent-appended propose op buried at the tail of a 1000-op
        recipe stays visible. The agent that wants a token-limited
        diff can re-request a smaller view directly.
        """
        ckpt = self._checkpoints.get(ckpt_id)
        if ckpt is None:
            return {
                "status": "unknown_checkpoint",
                "ckpt_id": ckpt_id,
                "available": sorted(self._checkpoints.keys()),
            }
        # Re-compute the checkpoint's view at full breadth too, since
        # it may have been stored with a smaller max_ops.
        ckpt_full = ckpt.view
        if ckpt_full.get("total_ops", 0) > len(ckpt_full.get("banner", []) + ckpt_full.get("middle", [])):
            # Original ckpt was truncated; we still diff against it but
            # surface the limitation in the result.
            pass
        now = self._compute_view(max_ops=100_000)
        diff = diff_views(ckpt_full, now)
        diff["status"] = "ok"
        diff["ckpt_id"] = ckpt_id
        return diff

    def checkpoint(self, label: str | None = None) -> str:
        """Freeze the current view as a named checkpoint. Returns its id.

        The stored view is full-breadth (not truncated by the default
        ``max_view_ops`` cap) so later ``diff_since(label)`` calls
        always see everything between the snapshot and the present.
        """
        ckpt_id = label or f"ckpt_{len(self._checkpoints)}"
        if ckpt_id in self._checkpoints:
            # Auto-dedupe by appending step index.
            ckpt_id = f"{ckpt_id}_{self._step_index}"
        full_view = self._compute_view(max_ops=100_000)
        self._checkpoints[ckpt_id] = DriverCheckpoint(
            ckpt_id=ckpt_id,
            view=full_view,
            step_index=self._step_index,
        )
        # Track the (possibly default-bounded) view for cheap re-reads.
        self._last_view = self._compute_view()
        return ckpt_id

    def summary(self) -> dict[str, Any]:
        """Return a small snapshot of session state for the MCP/Python callers."""
        assert self._loop is not None
        return {
            "session_id": self._session_id,
            "target": self.target.name,
            "step_index": self._step_index,
            "accepted_steps": self._accepted_steps,
            "budget": self.budget,
            "checkpoints": sorted(self._checkpoints.keys()),
            "recipe_hash": (self._last_view or {}).get("hash"),
            "cost_before_best_us": self.env._best_cost,
            "llm_calls": (self._llm_recorder.total_calls if self._llm_recorder is not None else 0),
            "tool_calls": (self._tool_recorder.total_calls if self._tool_recorder is not None else 0),
        }

    # ------------------------------------------------------------------
    # Tool / slot / proposal dispatch
    # ------------------------------------------------------------------

    def step_tool(
        self, tool_name: str, args: dict[str, Any] | None = None, *, phase: int | None = None, llm_turn_id: str = ""
    ) -> DriverStepResult:
        """Invoke a registered :class:`Tool` by name.

        Returns a :class:`DriverStepResult` whose ``status`` is one of:

        - ``"applied"`` — the tool ran successfully.
        - ``"failed"`` — the tool raised or returned a non-OK status.
        - ``"unknown"`` — no tool with that name in the registry.
        - ``"no_impl"`` — the tool is a stub without a real impl.
        """
        assert self.registry is not None
        assert self._tool_recorder is not None
        args = args or {}
        tool: Tool | None = self.registry.lookup_tool(tool_name, phase=phase)

        t0 = time.perf_counter()
        before_hash = self._current_ir_hash()
        view_before = self._compute_view()

        if tool is None:
            assert self.registry is not None
            available = sorted(t.name for t in self.registry.list_tools())
            nearest = self._nearest_slot_names(tool_name, available)
            hint = (
                f"No tool named {tool_name!r} in registry. "
                f"Available ({len(available)}): {available[:10] if available else '<none>'}. "
                + (f"Did you mean: {nearest}?" if nearest else "")
            )
            self._record_tool(
                name=tool_name,
                phase=phase or -1,
                kind="tool_call",
                args=args,
                result={"status": "unknown", "available_tools": available},
                before=view_before,
                after=view_before,
                gate_result=None,
                elapsed_ms=int((time.perf_counter() - t0) * 1000),
                llm_turn_id=llm_turn_id,
            )
            return DriverStepResult(
                status="unknown",
                kind="tool",
                name=tool_name,
                ir_hash_before=before_hash,
                ir_hash_after=before_hash,
                elapsed_ms=int((time.perf_counter() - t0) * 1000),
                summary=f"No tool named {tool_name!r} in registry.",
                remediation_hint=hint,
                tool_result={"available_tools": available, "nearest": nearest},
            )

        try:
            tool_result = tool.invoke(**args)
            raw_status = str(tool_result.get("status", "applied")).lower()
            status = "applied" if raw_status not in {"no_impl", "error", "failed"} else raw_status
            if tool.is_stub:
                status = "no_impl"
        except Exception as e:  # noqa: BLE001
            tool_result = {"status": "error", "error": f"{type(e).__name__}: {e}"}
            status = "failed"

        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        view_after = self._compute_view()
        after_hash = view_after.get("hash", before_hash)

        if status in {"applied"}:
            self._accepted_steps += 1
        self._step_index += 1

        self._record_tool(
            name=tool.name,
            phase=tool.phase,
            kind="tool_call",
            args=args,
            result=tool_result,
            before=view_before,
            after=view_after,
            gate_result=None,
            elapsed_ms=elapsed_ms,
            llm_turn_id=llm_turn_id,
        )

        self._last_view = view_after
        return DriverStepResult(
            status=status,
            kind="tool",
            name=tool.name,
            ir_hash_before=before_hash,
            ir_hash_after=after_hash,
            elapsed_ms=elapsed_ms,
            summary=f"tool {tool.name} status={status}",
            tool_result=tool_result,
            ir_view_after=(view_after if status == "applied" and after_hash != before_hash else None),
        )

    def step_invent(
        self,
        slot_name: str,
        proposal: dict[str, Any],
        *,
        phase: int | None = None,
        gate_ctx: dict[str, Any] | None = None,
        gates: list[Any] | None = None,
        llm_turn_id: str = "",
    ) -> DriverStepResult:
        """Submit a proposal to a registered :class:`InventSlot`'s gate.

        Runs the slot's registered ``gate_impl`` (or the default
        ``composite(structural, differential)`` when no gates are
        supplied) and returns an aggregated result carrying the
        remediation hint when the gate rejects.
        """
        assert self.registry is not None
        assert self._tool_recorder is not None
        slot: InventSlot | None = self.registry.lookup_invent_slot(
            slot_name,
            phase=phase,
        )

        t0 = time.perf_counter()
        before_hash = self._current_ir_hash()
        view_before = self._compute_view()
        ctx: dict[str, Any] = dict(gate_ctx or {})

        if slot is None:
            return self._unknown_slot_result(
                slot_name,
                before_hash,
                t0,
                phase,
                llm_turn_id,
                view=view_before,
            )

        if gates is None:
            # Default ladder: structural first, then differential IF ctx
            # has ref_fn/got_fn. Saves wasted work when the caller only
            # wants a syntactic check.
            gates = [structural_gate]
            if "ref_fn" in ctx and "got_fn" in ctx:
                gates.append(differential_gate)

        try:
            if slot.gate_impl is not None and (gates is None or len(gates) == 0):
                gate_result = slot.verify(proposal, **ctx)
            else:
                gate_result = composite_gate(
                    proposal,
                    gates=gates,
                    slot_name=slot.name,
                    **ctx,
                )
        except Exception as e:  # noqa: BLE001
            gate_result = {
                "status": "rejected",
                "details": {
                    "reason": "gate_raised",
                    "error": f"{type(e).__name__}: {e}",
                },
            }

        status = str(gate_result.get("status", "deferred"))
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        appended_op_name: str | None = None

        if status == "accepted":
            self._accepted_steps += 1
            # Append the corresponding Recipe-IR propose-op to the live
            # recipe module so downstream stages (lower_recipe →
            # RecipeExecutor) actually see it. Without this, accepted
            # proposals are dead letters. Bridge is scoped to slots we
            # know how to encode; unknown slots fall back to side-log only.
            if self.env.recipe is not None:
                try:
                    from compgen.agent.recipe_bridge_invent import (
                        proposal_to_recipe_op,
                    )

                    op = proposal_to_recipe_op(
                        slot.name,
                        proposal,
                        iteration=self._step_index,
                        llm_turn_id=llm_turn_id or self._session_id,
                    )
                    if op is not None:
                        self.env.recipe.body.block.add_op(op)
                        appended_op_name = op.name
                except ValueError as exc:
                    # Malformed proposal — demote to rejected with a
                    # remediation hint so the LLM retries.
                    status = "rejected"
                    gate_result = {
                        "status": "rejected",
                        "details": {
                            "reason": "proposal_schema_error",
                            "error": str(exc),
                            "remediation_hint": (
                                f"The {slot.name} builder could not construct "
                                "its Recipe-IR op from your 'chosen' block. "
                                f"Fix: {exc}"
                            ),
                        },
                    }
                    self._accepted_steps -= 1
                except Exception:  # noqa: BLE001
                    # Surfacing op-construction failures as rejections
                    # keeps the loop advancing; log silently.
                    pass

            # Remember an abbreviated invocation for later graduation /
            # contribution drafting.
            try:
                from compgen.agent.extensions.local_loader import record_accepted_invocation

                record_accepted_invocation(
                    None,
                    slot.name,
                    {
                        "proposal_digest": str(sorted(proposal.items()))[:400],
                        "session_id": self._session_id,
                        "step_index": self._step_index,
                    },
                )
            except Exception:  # noqa: BLE001
                pass

        # Recompute the view AFTER we've had a chance to mutate the
        # recipe — so ir_hash_after reflects the appended op when one
        # landed, and diff_since(...) surfaces it.
        view_after = self._compute_view()
        after_hash = view_after.get("hash", before_hash)

        self._step_index += 1

        self._record_tool(
            name=slot.name,
            phase=slot.phase,
            kind="invent_proposal",
            args={"proposal_keys": sorted(proposal.keys())},
            result={"status": status, "appended_op": appended_op_name},
            before=view_before,
            after=view_after,
            gate_result=gate_result,
            elapsed_ms=elapsed_ms,
            llm_turn_id=llm_turn_id,
            select_vs_invent=proposal.get("select_vs_invent", "invent"),
        )

        remediation = None
        details = gate_result.get("details") or {}
        if isinstance(details, dict):
            remediation = details.get("remediation_hint")

        self._last_view = view_after
        summary = f"invent slot {slot.name} gate={status}"
        if appended_op_name:
            summary += f" appended={appended_op_name}"
        return DriverStepResult(
            status=status,
            kind="invent",
            name=slot.name,
            ir_hash_before=before_hash,
            ir_hash_after=after_hash,
            elapsed_ms=elapsed_ms,
            summary=summary,
            gate_result=gate_result,
            remediation_hint=remediation,
            ir_view_after=(view_after if status == "accepted" and after_hash != before_hash else None),
            tool_result={"appended_op": appended_op_name} if appended_op_name else None,
        )

    def step_invent_many(
        self,
        proposals: list[dict[str, Any]],
        *,
        atomic: bool = False,
        gate_ctx: dict[str, Any] | None = None,
        llm_turn_id: str = "",
    ) -> tuple[list[DriverStepResult], bool]:
        """Run a batch of invent-slot proposals; return (results, rolled_back).

        Each entry in ``proposals`` is ``{slot_name, proposal, [phase], [gate_ctx]}``.
        On ``atomic=True``, the recipe + payload are deep-cloned BEFORE the
        first proposal; if any proposal in the batch is rejected (or the
        slot is unknown), the snapshots are rebound and the batch is
        considered rolled back. Returns the per-step results regardless,
        so the agent can read them all + decide.
        """
        snapshot_recipe = None
        snapshot_payload = None
        if atomic and self.env.recipe is not None:
            snapshot_recipe = self.env.recipe.clone()
        if atomic and self.env.payload_module is not None:
            snapshot_payload = self.env.payload_module.clone()

        results: list[DriverStepResult] = []
        rolled_back = False
        for entry in proposals:
            slot_name = entry.get("slot_name") or ""
            proposal = entry.get("proposal") or {}
            phase = entry.get("phase")
            entry_ctx = dict(gate_ctx or {})
            entry_ctx.update(entry.get("gate_ctx") or {})
            r = self.step_invent(
                slot_name,
                proposal,
                phase=phase,
                gate_ctx=entry_ctx,
                llm_turn_id=llm_turn_id,
            )
            results.append(r)
            if atomic and r.status not in {"accepted"}:
                # Rollback: rebind both recipe + payload to the
                # snapshots we took before the batch started.
                if snapshot_recipe is not None:
                    self.env._recipe_module = snapshot_recipe
                if snapshot_payload is not None:
                    self.env.set_payload_module(snapshot_payload)
                self._last_view = self._compute_view()
                rolled_back = True
                break
        return results, rolled_back

    def step_proposal(
        self,
        action_type: str,
        *,
        target: str = "",
        reason: str = "",
        llm_turn_id: str = "",
    ) -> DriverStepResult:
        """Convert an LLM action proposal into a concrete env step.

        Translates via :meth:`AgenticCompilationLoop._proposal_to_action`
        (delegated, not reimplemented), applies it to
        :meth:`CompilerEnv.step`, and optionally triggers per-step
        verification via :meth:`AgenticCompilationLoop._run_per_step_verification`.
        """
        assert self._loop is not None
        assert self._tool_recorder is not None

        t0 = time.perf_counter()
        before_hash = self._current_ir_hash()
        view_before = self._compute_view()

        proposal = ProposedOptimization(
            action_type=action_type,
            target=target,
            reason=reason,
            expected_improvement=0.0,
        )
        action: Action = self._loop._proposal_to_action(proposal)

        if isinstance(action, NoopAction):
            elapsed_ms = int((time.perf_counter() - t0) * 1000)
            self._record_tool(
                name=f"proposal:{action_type}",
                phase=-1,
                kind="tool_call",
                args={"target": target, "reason": reason},
                result={"status": "noop"},
                before=view_before,
                after=view_before,
                gate_result=None,
                elapsed_ms=elapsed_ms,
                llm_turn_id=llm_turn_id,
            )
            return DriverStepResult(
                status="noop",
                kind="proposal",
                name=action_type,
                ir_hash_before=before_hash,
                ir_hash_after=before_hash,
                elapsed_ms=elapsed_ms,
                summary=f"proposal {action_type!r} mapped to noop",
            )

        # Apply the action.
        try:
            step_result = self.env.step(action)
        except Exception as e:  # noqa: BLE001
            elapsed_ms = int((time.perf_counter() - t0) * 1000)
            return DriverStepResult(
                status="failed",
                kind="proposal",
                name=action_type,
                ir_hash_before=before_hash,
                ir_hash_after=before_hash,
                elapsed_ms=elapsed_ms,
                summary=f"env.step raised: {type(e).__name__}: {e}",
            )

        applied = bool(step_result.info.action_applied)
        if applied:
            self._accepted_steps += 1
        self._step_index += 1

        view_after = self._compute_view()
        after_hash = view_after.get("hash", before_hash)

        self._record_tool(
            name=f"proposal:{action_type}",
            phase=-1,
            kind="tool_call",
            args={"target": target, "reason": reason},
            result={
                "action_applied": applied,
                "cost_after_us": step_result.info.cost_after_us,
                "error": step_result.info.error,
            },
            before=view_before,
            after=view_after,
            gate_result=None,
            elapsed_ms=int((time.perf_counter() - t0) * 1000),
            llm_turn_id=llm_turn_id,
        )

        self._last_view = view_after
        return DriverStepResult(
            status="applied" if applied else "failed",
            kind="proposal",
            name=action_type,
            ir_hash_before=before_hash,
            ir_hash_after=after_hash,
            elapsed_ms=int((time.perf_counter() - t0) * 1000),
            summary=(f"proposal {action_type!r} applied={applied} cost={step_result.info.cost_after_us:.1f}"),
            diagnostics=step_result.info.diagnostics,
            tool_result={
                "cost_before_us": step_result.info.cost_before_us,
                "cost_after_us": step_result.info.cost_after_us,
                "improvement_pct": step_result.info.improvement_pct,
                "error": step_result.info.error,
            },
            ir_view_after=(view_after if applied and after_hash != before_hash else None),
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _compute_view(
        self,
        *,
        focus: str | None = None,
        max_ops: int | None = None,
    ) -> dict[str, Any]:
        """Compute the current Recipe-IR view, or a minimal stub view if tracking is off."""
        module = self.env.recipe
        if module is None:
            return {
                "hash": "sha256:no-recipe",
                "counts": {},
                "total_ops": 0,
                "banner": [],
                "middle": [],
            }
        return recipe_to_llm_view(
            module,
            max_ops=max_ops if max_ops is not None else self.max_view_ops,
            focus=focus,
        )

    def _current_ir_hash(self) -> str:
        if self._last_view is not None:
            return str(self._last_view.get("hash", "sha256:unknown"))
        return self._compute_view().get("hash", "sha256:unknown")

    def _record_tool(
        self,
        *,
        name: str,
        phase: int,
        kind: str,
        args: dict[str, Any],
        result: dict[str, Any],
        before: Any,
        after: Any,
        gate_result: dict[str, Any] | None,
        elapsed_ms: int,
        llm_turn_id: str,
        select_vs_invent: str = "na",
    ) -> None:
        assert self._tool_recorder is not None
        self._tool_recorder.record(
            phase=phase,
            name=name,
            kind=kind,
            args=args,
            result=result,
            select_vs_invent=select_vs_invent,
            before=before,
            after=after,
            gate_result=gate_result,
            elapsed_ms=elapsed_ms,
            llm_turn_id=llm_turn_id or self._session_id,
        )

    def _unknown_slot_result(
        self,
        slot_name: str,
        before_hash: str,
        t0: float,
        phase: int | None,
        llm_turn_id: str,
        view: dict[str, Any],
    ) -> DriverStepResult:
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        # Show the agent what slots ARE registered so it can pick a real
        # one on the next turn. Cap the list so the response stays small.
        assert self.registry is not None
        available = sorted(s.name for s in self.registry.list_invent_slots())
        nearest = self._nearest_slot_names(slot_name, available)
        hint = (
            f"No invent-slot named {slot_name!r} in registry. "
            f"Available: {available[:10] if available else '<none>'}. "
            + (f"Did you mean: {nearest}?" if nearest else "")
        )
        self._record_tool(
            name=slot_name,
            phase=phase or -1,
            kind="invent_proposal",
            args={},
            result={"status": "unknown", "available_slots": available},
            before=view,
            after=view,
            gate_result=None,
            elapsed_ms=elapsed_ms,
            llm_turn_id=llm_turn_id,
        )
        return DriverStepResult(
            status="unknown",
            kind="invent",
            name=slot_name,
            ir_hash_before=before_hash,
            ir_hash_after=before_hash,
            elapsed_ms=elapsed_ms,
            summary=f"No invent-slot named {slot_name!r} in registry.",
            remediation_hint=hint,
            tool_result={"available_slots": available, "nearest": nearest},
        )

    @staticmethod
    def _nearest_slot_names(query: str, available: list[str], k: int = 3) -> list[str]:
        """Cheap edit-distance ranking so the agent can recover from typos."""
        import difflib

        return difflib.get_close_matches(query, available, n=k, cutoff=0.4)


__all__ = [
    "DriverCheckpoint",
    "DriverStepResult",
    "LLMDrivenCompiler",
]
