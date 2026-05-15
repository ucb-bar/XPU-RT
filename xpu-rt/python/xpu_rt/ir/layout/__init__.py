"""Layout IR -- virtual layout encoding dialect for data tiling bridge.

A dialect that represents layout decisions as IR operations, enabling
virtual layout encodings to propagate through the IR before materializing
into physical pack/transpose operations at boundaries.

The ``Layout`` dialect contains 4 operations and 2 custom attributes.
"""

from __future__ import annotations

__all__: list[str] = []
