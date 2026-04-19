"""Wave 1+ reconstructed xDSL PatternRewriter passes.

Every pass here is CompGen-owned: zero external references to IREE
or XLA as runtime implementations. Each file ports the corresponding
upstream semantic (named in the module docstring) onto xDSL's
``PatternRewriter`` infrastructure directly.
"""

from __future__ import annotations

__all__: list[str] = []
