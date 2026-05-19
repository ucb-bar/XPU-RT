"""Provider adapter package.

Adapter modules wrap existing kernel providers under the
card-driven interface. The first two adapters formalize the
working CPU and GPU paths:

* ``cffi_c`` — wraps :class:`xpu_rt.kernels.providers.c_reference.CReferenceProvider`.
* ``triton`` — wraps :class:`xpu_rt.kernels.providers.triton_templates.TritonTemplatesProvider`.

The adapter pattern is intentionally thin: existing kernel
providers stay where they live (``xpu_rt.kernels.providers.*``),
and the card's ``entrypoint`` string resolves to them. does
not duplicate codegen logic — it only adds the resolution helper
plus tests that pin the ``ProviderResult ≠ Certificate``
discipline.
"""

from __future__ import annotations

from xpu_rt.providers.adapters.base import (
    AdapterResolutionError,
    resolve_provider_class,
)

__all__ = [
    "AdapterResolutionError",
    "resolve_provider_class",
]
