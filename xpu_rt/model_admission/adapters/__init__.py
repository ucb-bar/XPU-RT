"""Model-family-specific adapters.

Some models don't fit the standard ``forward(input_ids=..., pixel_values=...)``
pattern. An adapter wraps such a model into a torch.nn.Module with a
standard forward signature so the admission probe can run eager / fx /
export / dynamo / torch.compile uniformly across model families.

Adapter contract
----------------

Each adapter module exposes::

    def build(model: nn.Module, processor: Any | None) -> tuple[
        nn.Module,                # wrapped module with a torch-friendly forward
        tuple[Any, ...],          # positional sample_inputs for the wrapper
        dict[str, Any],           # keyword sample_inputs for the wrapper
    ]: ...

Adapters live next to the loader; they are NOT monkey patches. The
underlying model's source code is untouched. We just expose a
torch-standard call surface on top.
"""

from __future__ import annotations
