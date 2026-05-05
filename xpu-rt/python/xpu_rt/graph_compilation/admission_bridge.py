"""Bridge from ``compgen.model_admission`` configs into graph_compilation.

graph_compilation natively understands the simple ``graphcomp_model_config_v1``
schema (``model_path`` + ``factory``). The model-admission package owns a
richer ``model_config_v1`` schema with proxy / HF / compgen_model_spec
loaders that already handle real models — including the proxy modules
under ``compgen.model_admission.proxies`` for Qwen-VL, LLaVA, OpenVLA,
etc., that look like real models without requiring multi-GB HF weights.

This bridge lets graph_compilation consume a ``model_config_v1`` YAML by
synthesising a ``(model, sample_inputs)`` factory from the admission
loader. The compiler core (``compgen.ir``, ``compgen.capture``,
``compgen.pipeline``) is unaffected — this is a pure user-space adapter.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch.nn as nn
import yaml

from compgen.model_admission.loaders import LoadedModel, LoaderUnavailable, load
from compgen.model_admission.schemas import ModelConfig as AdmissionModelConfig

# Public constant — used by ``ModelConfig.load`` to recognise admission configs.
ADMISSION_SCHEMA_VERSION = "model_config_v1"


def is_admission_config(yaml_path: Path) -> bool:
    """Cheap probe — peek at ``schema_version`` without full parsing."""
    try:
        raw = yaml.safe_load(Path(yaml_path).read_text(encoding="utf-8")) or {}
    except Exception:
        return False
    return raw.get("schema_version") == ADMISSION_SCHEMA_VERSION


class _KwargAdapter(nn.Module):
    """Wrap a real model so positional args and frozen kwargs work together.

    HF VLMs typically expect ``input_ids=…, pixel_values=…`` keyword
    arguments. Dynamo capture in graph_compilation calls
    ``model(*sample_inputs)`` — purely positional. We bake the kwargs in
    here so the wrapped module can be called with the positional args
    only, and reproduces ``model(*args, **frozen_kwargs)``.
    """

    def __init__(self, model: nn.Module, frozen_kwargs: dict[str, Any]) -> None:
        super().__init__()
        self.inner = model
        # ``object.__setattr__`` so frozen_kwargs isn't registered as
        # a buffer / parameter / submodule.
        object.__setattr__(self, "_frozen_kwargs", dict(frozen_kwargs))

    def forward(self, *args: Any) -> Any:  # noqa: D401 — generic forward
        return self.inner(*args, **self._frozen_kwargs)


def make_factory_from_admission_config(
    yaml_path: Path,
    *,
    slice_id: str | None = None,
) -> Callable[[], tuple[nn.Module, tuple[Any, ...]]]:
    """Return a zero-arg ``factory()`` that yields ``(model, sample_inputs)``.

    The closure defers the actual admission load until call time — this
    matches graph_compilation's contract that ``factory()`` is invoked
    inside ``_run_capture`` after the seed is set.
    """
    yaml_path = Path(yaml_path).resolve()

    def factory() -> tuple[nn.Module, tuple[Any, ...]]:
        admission_cfg = AdmissionModelConfig.from_yaml(yaml_path)
        slice_cfg = None
        if slice_id:
            slice_cfg = next(
                (s for s in admission_cfg.slices if s.slice_id == slice_id),
                None,
            )
            if slice_cfg is None:
                raise ValueError(
                    f"slice_id={slice_id!r} not in admission config {yaml_path.name!r}"
                )

        try:
            loaded: LoadedModel = load(admission_cfg, slice_cfg)
        except LoaderUnavailable as exc:
            # Bubble up as a normal exception — graph_compilation's
            # capture stage already records `error_kind` / `detail`.
            raise RuntimeError(
                f"admission loader unavailable for {admission_cfg.model_id}: "
                f"{exc.status.value} — {exc.reason}"
            ) from exc

        model = loaded.model
        if loaded.sample_kwargs:
            model = _KwargAdapter(model, loaded.sample_kwargs)
        return model.eval(), tuple(loaded.sample_inputs)

    return factory
