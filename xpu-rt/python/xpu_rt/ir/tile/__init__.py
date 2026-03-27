"""Tile IR -- backend-plural tile virtual ISA for leaf kernels.

A small internal dialect that abstracts over tiled computation patterns
shared by multiple backends (Triton, Exo, custom accelerator, ukernel).

The ``Tile`` dialect object is the xDSL dialect registration containing
7 operations and 3 custom attributes.
"""

from __future__ import annotations

__all__: list[str] = []
