"""Terminal UI primitives for the XPU-RT CLI.

Two pieces live here today:

* ``question`` — a Claude-Code-style interactive question widget
  (numbered options, ``[✔] / [ ]`` markers, "Type something" /
  "Skip interview" fallbacks). Built on prompt_toolkit.
* ``dashboard`` — a Rich Live multi-panel display (stage progress,
  ASCII Gantt, predicted-vs-measured deltas, decision-log stream)
  that tails the QNN events JSONL.

Both are imported lazily so users that don't run the QNN flow never
pay for the prompt_toolkit / Rich import cost.
"""

from __future__ import annotations

__all__ = ["ask_user_question", "QuestionOption", "QuestionAnswer"]


def __getattr__(name: str):  # pragma: no cover (re-export shim)
    if name in {"ask_user_question", "QuestionOption", "QuestionAnswer"}:
        from xpu_rt.ui.question import (
            QuestionAnswer,
            QuestionOption,
            ask_user_question,
        )

        return {
            "ask_user_question": ask_user_question,
            "QuestionOption": QuestionOption,
            "QuestionAnswer": QuestionAnswer,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
