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

    def require_device(self) -> CompGenDevice:
        if self.device is None:
            raise RuntimeError(
                "No target open in this session. Call open_target first."
            )
        return self.device

    def require_driver(self) -> LLMDrivenCompiler:
        if self.driver is None:
            raise RuntimeError(
                "No model loaded in this session. Call load_model first."
            )
        return self.driver

    def require_compiled(self) -> CompiledModel:
        if self.compiled is None:
            raise RuntimeError(
                "No compiled model in this session. Call load_model first."
            )
        return self.compiled


class SessionManager:
    """Process-wide map of ``session_id -> :class:`McpSession```.

    Thread safety is delegated to the MCP server framework — handlers
    run sequentially on stdio transport. If the transport is ever
    swapped for something concurrent, add a lock here.
    """

    def __init__(self, scratch_root: Path | None = None) -> None:
        self._sessions: dict[str, McpSession] = {}
        self._scratch_root = scratch_root or Path(
            tempfile.gettempdir()
        ) / "compgen-mcp"
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
