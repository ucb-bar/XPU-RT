"""Per-client session state for the MCP server.

Each MCP client is identified by a string ``session_id``. A session
owns:

* The resolved :class:`~compgen.api.CompGenDevice` (from ``open_target``).
* The loaded model + sample inputs (from ``load_model``).
* The live :class:`~compgen.agent.llm_driver.LLMDrivenCompiler`
  (created once the model is loaded).
* A small scratch directory for bundle exports.

The session manager is intentionally in-process and non-persistent —
restarting ``compgen-mcp`` drops all sessions. Cross-session knowledge
lives in ``~/.compgen/transcripts`` + ``~/.compgen/memory.sqlite`` via
the existing :class:`~compgen.llm.recorder.ToolCallRecorder` +
:class:`~compgen.memory.store.CompilerMemory`.
"""

from __future__ import annotations

import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog

from compgen.agent.llm_driver import LLMDrivenCompiler
from compgen.api import CompGenDevice, CompiledModel
from compgen.llm.base import CompGenLLMProtocol

log = structlog.get_logger()


@dataclass
class McpSession:
    """One MCP client session."""

    session_id: str
    scratch_dir: Path
    device: CompGenDevice | None = None
    compiled: CompiledModel | None = None
    driver: LLMDrivenCompiler | None = None
    llm_client: CompGenLLMProtocol | None = None
    model_hint: str = ""
    provider: str = ""
    spec_path: Path | None = None
    packs: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    # In-session kernel-codegen state. Populated by the
    # ``request_kernel_codegen`` / ``register_kernel_result`` MCP tools so
    # ``ClaudeCodeKernelProvider`` can fulfill kernel requests by reading
    # this cache instead of calling an external Anthropic API.
    kernel_cache: Any = None  # set lazily; see compgen.mcp.tools.kernel._kernel_cache
    # In-session HW-aware-dispatch state (W6.4). Populated by the
    # ``request_dispatch_decision`` / ``register_dispatch_decision`` tools
    # so the ``McpDispatchLLM`` adapter can route W6's ``decide_dispatch``
    # through Claude Code instead of an external API.
    dispatch_cache: Any = None
    # In-session bench cache (W7.1) — see compgen.mcp.tools.bench._bench_cache
    bench_cache: Any = None
    # Per-session optimisation progress tracker (W7.3)
    optim_progress: Any = None
    # In-session refinement cache (W8.2) — see compgen.mcp.tools.refinement
    refinement_cache: Any = None
    # In-session autotune cache (W8.3) — see compgen.mcp.tools.autotune
    autotune_cache: Any = None
    # In-session decision-site registry. Stage plugins enqueue sites
    # here via :mod:`compgen.agent.decisions`; MCP tools read/write
    # them via :mod:`compgen.mcp.tools.decisions`.
    decision_registry: Any = None
    # H2 — Section 11 Dream 1: phase-scoped discovery. ``None`` means
    # legacy callers see the full tool catalogue (backwards-compat
    # default); strict mode is opt-in via ``enter_phase`` + the env
    # flag ``COMPGEN_STRICT_PHASE_GATING=1``.
    current_phase: str | None = None
    # H3 — Section 11 Dream 3: capability tokens. Empty set means no
    # high-risk tool is callable; populated at session open by the
    # operator who passes a list to ``open_target``.
    capabilities: frozenset[str] = field(default_factory=frozenset)
    # H3 — closed enum of the caller's role for ``caller_must_be``
    # gating. ``agent`` is the default (this Claude Code session);
    # other valid values: ``operator``, ``kernel_provider``.
    caller_role: str = "agent"

    def require_decision_registry(self):
        """Lazy-create and return this session's :class:`DecisionRegistry`.

        Also installs it as the active registry so stage plugins in the
        current context see it.
        """
        if self.decision_registry is None:
            from compgen.agent.decisions import (
                DecisionRegistry,
                install_registry,
            )

            self.decision_registry = DecisionRegistry()
            install_registry(self.decision_registry)
        return self.decision_registry

    def require_device(self) -> CompGenDevice:
        if self.device is None:
            raise RuntimeError("No target open in this session. Call open_target first.")
        return self.device

    def require_driver(self) -> LLMDrivenCompiler:
        if self.driver is None:
            raise RuntimeError("No model loaded in this session. Call load_model first.")
        return self.driver

    def require_compiled(self) -> CompiledModel:
        if self.compiled is None:
            raise RuntimeError("No compiled model in this session. Call load_model first.")
        return self.compiled


class SessionManager:
    """Process-wide map of ``session_id -> :class:`McpSession```.

    Thread safety is delegated to the MCP server framework — handlers
    run sequentially on stdio transport. If the transport is ever
    swapped for something concurrent, add a lock here.
    """

    def __init__(self, scratch_root: Path | None = None) -> None:
        self._sessions: dict[str, McpSession] = {}
        self._scratch_root = scratch_root or Path(tempfile.gettempdir()) / "compgen-mcp"
        self._scratch_root.mkdir(parents=True, exist_ok=True)

    def open(self, session_id: str | None = None) -> McpSession:
        sid = session_id or f"mcp_{uuid.uuid4().hex[:10]}"
        if sid in self._sessions:
            return self._sessions[sid]
        scratch = self._scratch_root / sid
        scratch.mkdir(parents=True, exist_ok=True)
        session = McpSession(session_id=sid, scratch_dir=scratch)
        self._sessions[sid] = session
        log.info("mcp.session.open", session_id=sid)
        return session

    def get(self, session_id: str) -> McpSession:
        if session_id not in self._sessions:
            raise KeyError(f"Unknown session id: {session_id}")
        return self._sessions[session_id]

    def close(self, session_id: str) -> None:
        session = self._sessions.pop(session_id, None)
        if session is None:
            return
        # We intentionally leave the scratch dir on disk; it's small
        # (bundle manifests, recipe exports) and useful for post-mortem.
        log.info("mcp.session.close", session_id=session_id)

    def list_sessions(self) -> list[str]:
        return sorted(self._sessions.keys())

    @property
    def scratch_root(self) -> Path:
        return self._scratch_root


__all__ = ["McpSession", "SessionManager"]
