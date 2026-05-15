"""Compilation trace package.

The trace bus captures every LLM prompt, MCP tool call, pass, analysis,
and agent decision in a single correlated JSONL stream. See
:mod:`compgen.trace.bus` for the entry points.
"""

from __future__ import annotations

from compgen.trace.adapters import (
    TracingLLMRecorder,
    TracingMcpTranscriptRecorder,
    TracingToolCallRecorder,
)
from compgen.trace.bus import (
    TraceBus,
    get_active_bus,
    get_current_llm_turn_id,
    install_bus,
    set_active_bus,
    set_current_llm_turn_id,
)
from compgen.trace.events import (
    DEFAULT_LEVEL_BY_KIND,
    EventKind,
    Level,
    Phase,
    TraceEvent,
    category_for,
    default_level_for,
)
from compgen.trace.ir_dump import (
    IRDumpEntry,
    IRDumpWriter,
    dump_enabled_from_env,
    get_ir_dump_writer,
    install_ir_dump_writer,
)
from compgen.trace.publishers import (
    AnalysisPublisher,
    DecisionPublisher,
    DecisionSitePublisher,
    IRDumpPublisher,
    LLMPublisher,
    MCPPublisher,
    OraclePublisher,
    PassPublisher,
    StagePublisher,
    ToolPublisher,
)
from compgen.trace.render import render_trace
from compgen.trace.session_id import build_session_id

__all__ = [
    "AnalysisPublisher",
    "DEFAULT_LEVEL_BY_KIND",
    "DecisionPublisher",
    "DecisionSitePublisher",
    "EventKind",
    "Level",
    "IRDumpEntry",
    "IRDumpPublisher",
    "IRDumpWriter",
    "LLMPublisher",
    "MCPPublisher",
    "OraclePublisher",
    "PassPublisher",
    "Phase",
    "StagePublisher",
    "ToolPublisher",
    "TraceBus",
    "TraceEvent",
    "TracingLLMRecorder",
    "TracingMcpTranscriptRecorder",
    "TracingToolCallRecorder",
    "build_session_id",
    "category_for",
    "default_level_for",
    "dump_enabled_from_env",
    "get_active_bus",
    "get_current_llm_turn_id",
    "get_ir_dump_writer",
    "install_bus",
    "install_ir_dump_writer",
    "render_trace",
    "set_active_bus",
    "set_current_llm_turn_id",
]
