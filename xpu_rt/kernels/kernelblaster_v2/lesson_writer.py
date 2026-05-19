"""Append accepted strategies to the Target Card's lessons.jsonl.

A lesson is one row per accepted candidate the agent loop produces: it
captures (state, action, measured_gain). Future runs read it from
:func:`xpu_rt.memory.target_knowledge.iter_lessons` and the prompt
builder folds the matching rows into the next propose request.

This module is intentionally tiny — the structured shape lives in
:class:`xpu_rt.memory.target_knowledge.Lesson` so KB v2 doesn't own a
parallel schema.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from xpu_rt.kernels.kernelblaster_v2.contract_state import StateVector
from xpu_rt.memory.target_knowledge import (
    Lesson,
    TargetKnowledgeCard,
    append_lesson,
)


@dataclass(frozen=True)
class LessonWriter:
    """Bind a target card to a lesson sink."""

    card: TargetKnowledgeCard

    @property
    def path(self) -> Path:
        return self.card.lessons_path

    def write(
        self,
        *,
        state: StateVector,
        action: str,
        measured_gain: float,
        notes: str = "",
    ) -> Lesson:
        lesson = Lesson(
            timestamp=datetime.now(timezone.utc).isoformat(),
            archetype=state.archetype,
            dtype_class=state.dtype_class,
            layout_kind=state.layout_kind,
            op_family=state.op_family,
            action=action,
            measured_gain=measured_gain,
            sample_count=1,
            notes=notes,
        )
        append_lesson(self.card, lesson)
        return lesson
