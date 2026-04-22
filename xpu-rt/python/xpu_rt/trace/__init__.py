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
from compgen.trace.events import EventKind, Phase, TraceEvent
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

__all__ = [
    "AnalysisPublisher",
    "DecisionPublisher",
    "DecisionSitePublisher",
    "EventKind",
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
    "dump_enabled_from_env",
    "get_active_bus",
    "get_current_llm_turn_id",
    "get_ir_dump_writer",
    "install_bus",
    "install_ir_dump_writer",
    "set_active_bus",
    "set_current_llm_turn_id",
]
