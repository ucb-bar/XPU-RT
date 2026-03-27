"""LLM prompt library for agentic compilation.

Each module exports ``format_prompt(context) -> str`` and
``parse_response(response) -> Action`` for a specific optimization mode.
"""

from __future__ import annotations
