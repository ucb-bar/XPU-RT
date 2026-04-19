"""Process-wide map of registered :class:`AuthoredTool` candidates.

The graduation loop refuses to materialise a tool whose source it
hasn't seen — we never synthesise impl bytes from a trial-log digest
alone. Callers that author tools register them here so the next call
to :func:`promote_authored_tools` can find them.

This module is the simplest possible shared registry: a dict guarded
by a lock, with a snapshot helper for the read path.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:   # pragma: no cover
    from compgen.agent.self_extension.authored_tool import AuthoredTool

from compgen.agent.self_extension.authored_tool import authored_tool_key


_lock = threading.Lock()
_index: dict[str, "AuthoredTool"] = {}


def register_authored_tool(tool: "AuthoredTool") -> str:
    """Add ``tool`` to the process-wide index keyed by name@digest."""
    key = authored_tool_key(tool)
    with _lock:
        _index[key] = tool
    return key


def snapshot_authored_index() -> dict[str, "AuthoredTool"]:
    """Return a shallow copy so callers don't see mid-write state."""
    with _lock:
        return dict(_index)


def clear_authored_index() -> None:
    """Drop all entries; intended for tests."""
    with _lock:
        _index.clear()


__all__ = [
    "clear_authored_index",
    "register_authored_tool",
    "snapshot_authored_index",
]
