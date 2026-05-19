"""Agent-first compiler environment.

The primary interface for LLM/agent-driven compilation. Designed so that
the system tells the agent what it CAN do, validates actions BEFORE
execution, and returns structured rewards — not raw IR text.

The core abstraction is ``CompilerEnv``:
    - ``reset()`` starts a new optimization episode
    - ``observe()`` returns structured state (not IR text)
    - ``legal_actions()`` returns what the agent can do right now with costs
    - ``step(action)`` applies an action and returns reward + next state
"""

from __future__ import annotations

__all__: list[str] = []
