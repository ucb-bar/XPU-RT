"""Proxy models for the admission probe.

Each proxy is a real (tiny) :class:`torch.nn.Module` that exercises a
representative subset of ops for a model family. Proxies must:

- eager-run successfully on CPU,
- pass through TorchDynamo without crashing,
- compile under ``torch.compile(backend='inductor')``.

These three guarantees are checked in
``tests/model_admission/test_proxy_models.py``. A proxy that cannot
satisfy them is a real defect, not a placeholder -- per the
``feedback_no_stubs_real_examples.md`` user policy.

Each module exposes::

    def build_proxy(slice_id: str = "") -> tuple[nn.Module, tuple | dict | (tuple, dict)]:
        ...
"""

from __future__ import annotations
