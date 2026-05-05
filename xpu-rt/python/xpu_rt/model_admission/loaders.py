"""Loader dispatch for model admission probes.

Loaders are pure: given a :class:`ModelConfig` (and optional
:class:`SliceConfig` for slice loaders), they return either a
:class:`LoadedModel` -- a real :class:`torch.nn.Module` plus the
``sample_inputs`` tuple it expects -- or raise :class:`LoaderUnavailable`
with a typed status enum value.

Critical contract: loaders **never** download weights. HuggingFace loaders
check ``HF_HOME`` (or the default cache) for already-present snapshots and
raise ``LoaderUnavailable(unavailable_missing_weights, ...)`` if the
snapshot is absent.
"""

from __future__ import annotations

import importlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import structlog
import torch.nn as nn

from compgen.model_admission.schemas import (
    AdmissionStatus,
    ModelConfig,
    ModelLoaderConfig,
    SliceConfig,
)

log = structlog.get_logger(__name__)


@dataclass(frozen=True)
class LoadedModel:
    """Result of a successful loader call."""

    model: nn.Module
    sample_inputs: tuple[Any, ...]
    sample_kwargs: dict[str, Any]


class LoaderUnavailable(Exception):
    """Raised by a loader when it cannot produce a model.

    The probe maps ``status`` directly into the
    :class:`AdmissionReport.status` field. ``reason`` is human-readable,
    ``error`` is the underlying exception text if any. ``hardware_requirements``
    is set when status is ``UNAVAILABLE_HARDWARE_CONSTRAINT`` -- documents
    what the model would need to run.
    """

    def __init__(
        self,
        status: AdmissionStatus,
        reason: str,
        error: str | None = None,
        hardware_requirements: Any = None,
    ) -> None:
        super().__init__(f"{status.value}: {reason}")
        self.status = status
        self.reason = reason
        self.error = error
        self.hardware_requirements = hardware_requirements


def _effective_loader(model_cfg: ModelConfig, slice_cfg: SliceConfig | None) -> ModelLoaderConfig:
    if slice_cfg is not None and slice_cfg.loader_override is not None:
        return slice_cfg.loader_override
    return model_cfg.loader


def load(model_cfg: ModelConfig, slice_cfg: SliceConfig | None = None) -> LoadedModel:
    """Dispatch to the right loader by ``loader.kind``.

    Raises:
        LoaderUnavailable: with a typed :class:`AdmissionStatus`.
    """

    loader = _effective_loader(model_cfg, slice_cfg)
    kind = loader.kind

    if kind == "unavailable":
        # ``support.reason`` carries the human-readable explanation; map to
        # the most accurate AdmissionStatus when we can.
        reason = model_cfg.support.reason or "loader.kind=unavailable"
        status = AdmissionStatus.UNAVAILABLE_MISSING_DEPENDENCY
        lower = reason.lower()
        if "too large" in lower or "huge" in lower or "unavailable_for_full_local" in lower:
            status = AdmissionStatus.UNAVAILABLE_TOO_LARGE
        elif (
            "compute capability" in lower
            or "flash_attn" in lower
            or "cc>=" in lower
            or "requires gpu" in lower
            or "ampere" in lower
            or "hopper" in lower
        ):
            status = AdmissionStatus.UNAVAILABLE_HARDWARE_CONSTRAINT
        # Parse a structured ``hardware_requirements`` block from the YAML's
        # support section if present.
        from compgen.model_admission.schemas import HardwareRequirements  # noqa: PLC0415

        hwr = None
        hw_yaml = getattr(model_cfg.support, "hardware_requirements", None)
        if isinstance(hw_yaml, dict):
            hwr = HardwareRequirements(
                min_compute_capability=str(hw_yaml.get("min_compute_capability", "")),
                min_vram_gb=float(hw_yaml.get("min_vram_gb", 0.0) or 0.0),
                required_dtypes=tuple(str(d) for d in hw_yaml.get("required_dtypes", []) or []),
                required_runtime_packages=tuple(str(p) for p in hw_yaml.get("required_runtime_packages", []) or []),
                notes=str(hw_yaml.get("notes", "")),
            )
            status = AdmissionStatus.UNAVAILABLE_HARDWARE_CONSTRAINT
        raise LoaderUnavailable(status, reason=reason, hardware_requirements=hwr)

    if loader.device_policy == "unavailable_for_full_local" and slice_cfg is None:
        raise LoaderUnavailable(
            AdmissionStatus.UNAVAILABLE_TOO_LARGE,
            reason=model_cfg.support.reason or "device_policy=unavailable_for_full_local; slice required",
        )

    if kind == "proxy":
        return _load_proxy(model_cfg, slice_cfg, loader)
    if kind == "compgen_model_spec":
        return _load_compgen_model_spec(model_cfg, slice_cfg, loader)
    if kind.startswith("hf_transformers_"):
        return _load_hf_transformers(model_cfg, slice_cfg, loader)
    raise LoaderUnavailable(
        AdmissionStatus.UNAVAILABLE_MISSING_DEPENDENCY,
        reason=f"unknown loader.kind={kind!r}",
    )


# --------------------------------------------------------------------------- #
# Loader: proxy.
# --------------------------------------------------------------------------- #


def _load_proxy(
    model_cfg: ModelConfig,
    slice_cfg: SliceConfig | None,
    loader: ModelLoaderConfig,
) -> LoadedModel:
    module_path = loader.proxy_module
    if not module_path:
        raise LoaderUnavailable(
            AdmissionStatus.UNAVAILABLE_MISSING_DEPENDENCY,
            reason="loader.proxy_module empty for proxy loader",
        )
    try:
        mod = importlib.import_module(module_path)
    except Exception as exc:
        raise LoaderUnavailable(
            AdmissionStatus.UNAVAILABLE_MISSING_DEPENDENCY,
            reason=f"failed to import proxy module {module_path!r}",
            error=repr(exc),
        ) from exc
    factory = getattr(mod, "build_proxy", None)
    if factory is None:
        raise LoaderUnavailable(
            AdmissionStatus.UNAVAILABLE_MISSING_DEPENDENCY,
            reason=f"proxy module {module_path!r} has no 'build_proxy' callable",
        )
    slice_id = slice_cfg.slice_id if slice_cfg is not None else ""
    try:
        result = factory(slice_id=slice_id)
    except Exception as exc:
        raise LoaderUnavailable(
            AdmissionStatus.FAILED_EAGER,
            reason="proxy build_proxy raised",
            error=repr(exc),
        ) from exc
    if not isinstance(result, tuple) or len(result) != 2:
        raise LoaderUnavailable(
            AdmissionStatus.FAILED_EAGER,
            reason="proxy build_proxy must return (nn.Module, sample_inputs)",
        )
    model, sample = result
    if not isinstance(model, nn.Module):
        raise LoaderUnavailable(
            AdmissionStatus.FAILED_EAGER,
            reason="proxy build_proxy did not return an nn.Module",
        )
    args, kwargs = _split_sample(sample)
    return LoadedModel(model=model.eval(), sample_inputs=args, sample_kwargs=kwargs)


# --------------------------------------------------------------------------- #
# Loader: compgen_model_spec (bridge to existing python/compgen/models catalog).
# --------------------------------------------------------------------------- #


def _load_compgen_model_spec(
    model_cfg: ModelConfig,
    slice_cfg: SliceConfig | None,
    loader: ModelLoaderConfig,
) -> LoadedModel:
    spec_id = loader.model_spec_id
    if not spec_id:
        raise LoaderUnavailable(
            AdmissionStatus.UNAVAILABLE_MISSING_DEPENDENCY,
            reason="loader.model_spec_id empty for compgen_model_spec loader",
        )
    try:
        from compgen.models import build_default_model_catalog  # type: ignore[import-not-found]
    except Exception as exc:
        raise LoaderUnavailable(
            AdmissionStatus.UNAVAILABLE_MISSING_DEPENDENCY,
            reason="cannot import compgen.models.build_default_model_catalog",
            error=repr(exc),
        ) from exc
    try:
        catalog = build_default_model_catalog()
    except Exception as exc:
        raise LoaderUnavailable(
            AdmissionStatus.UNAVAILABLE_MISSING_DEPENDENCY,
            reason="build_default_model_catalog() raised",
            error=repr(exc),
        ) from exc
    if spec_id not in catalog.models:
        raise LoaderUnavailable(
            AdmissionStatus.UNAVAILABLE_MISSING_DEPENDENCY,
            reason=f"model_spec_id={spec_id!r} not in compgen.models catalog",
        )
    spec = catalog.get(spec_id)
    try:
        model, sample = spec.load(None)
    except FileNotFoundError as exc:
        raise LoaderUnavailable(
            AdmissionStatus.UNAVAILABLE_MISSING_WEIGHTS,
            reason=f"model_spec={spec_id!r}: weights or external repo missing",
            error=repr(exc),
        ) from exc
    except ImportError as exc:
        raise LoaderUnavailable(
            AdmissionStatus.UNAVAILABLE_MISSING_DEPENDENCY,
            reason=f"model_spec={spec_id!r}: missing python dependency",
            error=repr(exc),
        ) from exc
    except Exception as exc:
        raise LoaderUnavailable(
            AdmissionStatus.FAILED_EAGER,
            reason=f"model_spec={spec_id!r}: spec.load() raised",
            error=repr(exc),
        ) from exc
    args, kwargs = _split_sample(sample)
    return LoadedModel(model=model.eval(), sample_inputs=args, sample_kwargs=kwargs)


# --------------------------------------------------------------------------- #
# Loader: huggingface transformers (cache-only; never downloads).
# --------------------------------------------------------------------------- #


def _hf_cache_root() -> Path:
    """Resolve the HuggingFace hub cache directory.

    Defers to ``huggingface_hub.constants.HF_HUB_CACHE`` -- the canonical
    resolver -- so we honour the same precedence (HF_HUB_CACHE > HF_HOME/hub
    > ~/.cache/huggingface/hub) and don't double-append ``/hub``.
    """

    try:
        from huggingface_hub.constants import HF_HUB_CACHE  # noqa: PLC0415
        return Path(HF_HUB_CACHE)
    except Exception:
        env_hub = os.environ.get("HUGGINGFACE_HUB_CACHE")
        if env_hub:
            return Path(env_hub)
        hf_home = os.environ.get("HF_HOME")
        if hf_home:
            return Path(hf_home) / "hub"
        return Path.home() / ".cache" / "huggingface" / "hub"


def _hf_snapshot_present(model_ref: str) -> bool:
    """Check if a HuggingFace snapshot is cached locally without touching the network."""

    if model_ref in {"", "TO_BE_VERIFIED_ONLINE"}:
        return False
    sanitized = "models--" + model_ref.replace("/", "--")
    return (_hf_cache_root() / sanitized).exists()


def _load_hf_transformers(
    model_cfg: ModelConfig,
    slice_cfg: SliceConfig | None,
    loader: ModelLoaderConfig,
) -> LoadedModel:
    """Load an HF transformers model from the local cache and build inputs.

    The loader is offline-only: ``TRANSFORMERS_OFFLINE=1`` and ``HF_HUB_OFFLINE=1``
    are forced, so any hidden network access fails fast with
    ``unavailable_missing_weights``.

    Family dispatch:

    - ``vlm`` / ``embodied_vlm_vla`` : AutoProcessor + AutoModelForVision2Seq
      (or AutoModel as fallback) with a synthetic 224x224 RGB image and prompt.
    - ``ocr`` : AutoProcessor + AutoModel with a synthetic 224x224 RGB image.
    - ``llm`` / ``llm_moe`` : AutoTokenizer + AutoModelForCausalLM with a short
      synthetic prompt.
    - ``vla`` / ``vla_diffusion`` / ``robot_policy`` : AutoProcessor (if present)
      + AutoModel; falls back to AutoTokenizer-only path when no processor.
    """

    import torch  # noqa: PLC0415  -- local import keeps loaders.py importable in CPU-only contexts

    model_ref = model_cfg.source.model_ref
    if not _hf_snapshot_present(model_ref):
        raise LoaderUnavailable(
            AdmissionStatus.UNAVAILABLE_MISSING_WEIGHTS,
            reason=f"HF snapshot for {model_ref!r} not present in {_hf_cache_root()}",
        )
    try:
        transformers = importlib.import_module("transformers")
    except Exception as exc:
        raise LoaderUnavailable(
            AdmissionStatus.UNAVAILABLE_MISSING_DEPENDENCY,
            reason="transformers package not installed",
            error=repr(exc),
        ) from exc

    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    revision = model_cfg.source.revision or None
    common_kwargs: dict[str, Any] = {
        "trust_remote_code": loader.trust_remote_code,
        "local_files_only": True,
    }
    if revision:
        common_kwargs["revision"] = revision

    model_load_kwargs = dict(common_kwargs)
    dtype = getattr(torch, loader.dtype, None) if loader.dtype else None
    if isinstance(dtype, torch.dtype):
        model_load_kwargs["torch_dtype"] = dtype
    # Load directly onto GPU when available — avoids leaving rotary embedding
    # buffers / position_ids on CPU after a post-hoc .to('cuda').
    if torch.cuda.is_available():
        model_load_kwargs["device_map"] = "cuda:0"

    family = model_cfg.family
    adapter_path = (loader.adapter or "").strip()
    try:
        if adapter_path:
            # Adapter mode: only load the model; the adapter builds inputs
            # and wraps forward. Skip the family-specific input cascade.
            model = _load_hf_model_only(transformers, model_ref, model_load_kwargs, common_kwargs)
            adapter_mod = importlib.import_module(adapter_path)
            build_fn = getattr(adapter_mod, "build", None)
            if build_fn is None:
                raise LoaderUnavailable(
                    AdmissionStatus.UNAVAILABLE_MISSING_DEPENDENCY,
                    reason=f"adapter {adapter_path!r} has no 'build' function",
                )
            wrapped, args, kwargs = build_fn(model, None)
            return LoadedModel(model=wrapped.eval(), sample_inputs=args, sample_kwargs=kwargs)

        if family in ("llm", "llm_moe"):
            model, sample = _load_hf_llm(transformers, model_ref, model_load_kwargs, common_kwargs)
        elif family == "ocr":
            model, sample = _load_hf_ocr(transformers, model_ref, model_load_kwargs, common_kwargs)
        elif family in ("vlm", "embodied_vlm_vla"):
            model, sample = _load_hf_vlm(transformers, model_ref, model_load_kwargs, common_kwargs)
        elif family in ("vla", "vla_diffusion", "robot_policy"):
            model, sample = _load_hf_vla(transformers, model_ref, model_load_kwargs, common_kwargs)
        else:
            raise LoaderUnavailable(
                AdmissionStatus.UNAVAILABLE_MISSING_DEPENDENCY,
                reason=f"hf_transformers loader has no family handler for family={family!r}",
            )
    except LoaderUnavailable:
        raise
    except OSError as exc:
        raise LoaderUnavailable(
            AdmissionStatus.UNAVAILABLE_MISSING_WEIGHTS,
            reason=f"transformers refused to load {model_ref!r} from cache",
            error=repr(exc),
        ) from exc
    except ImportError as exc:
        raise LoaderUnavailable(
            AdmissionStatus.UNAVAILABLE_MISSING_DEPENDENCY,
            reason=f"missing python dependency for {model_ref!r}",
            error=repr(exc),
        ) from exc
    except Exception as exc:
        raise LoaderUnavailable(
            AdmissionStatus.FAILED_EAGER,
            reason=f"hf_transformers loader failed for {model_ref!r}",
            error=f"{type(exc).__name__}: {exc}",
        ) from exc

    args, kwargs = _split_sample(sample)
    return LoadedModel(model=model.eval(), sample_inputs=args, sample_kwargs=kwargs)


def _load_hf_model_only(
    transformers: Any,
    model_ref: str,
    model_kwargs: dict[str, Any],
    common_kwargs: dict[str, Any],
) -> Any:
    """Load just the HF model -- used when an adapter handles inputs.

    Walks the standard AutoModel head list and falls back to the dynamic
    class loader via ``config.auto_map`` (covers OpenVLA / Moondream2 /
    DeepSeek-OCR custom configs).
    """

    auto_classes = [
        getattr(transformers, "AutoModelForImageTextToText", None),
        getattr(transformers, "AutoModelForVision2Seq", None),
        getattr(transformers, "AutoModelForCausalLM", None),
        transformers.AutoModel,
    ]
    last_err: Exception | None = None
    for auto_cls in auto_classes:
        if auto_cls is None:
            continue
        try:
            return auto_cls.from_pretrained(model_ref, **model_kwargs)
        except (ValueError, KeyError) as exc:
            last_err = exc
            continue
    try:
        cfg = transformers.AutoConfig.from_pretrained(model_ref, **dict(common_kwargs, trust_remote_code=True))
        auto_map = getattr(cfg, "auto_map", {}) or {}
        class_ref = next(
            (auto_map[k] for k in (
                "AutoModelForImageTextToText",
                "AutoModelForVision2Seq",
                "AutoModelForCausalLM",
                "AutoModel",
            ) if k in auto_map),
            None,
        )
        if class_ref:
            from transformers.dynamic_module_utils import get_class_from_dynamic_module  # noqa: PLC0415

            cls = get_class_from_dynamic_module(class_ref, model_ref)
            return cls.from_pretrained(model_ref, **model_kwargs)
    except Exception as exc:
        last_err = exc
    raise LoaderUnavailable(
        AdmissionStatus.FAILED_EAGER,
        reason=f"no AutoModel head accepted {model_ref!r}",
        error=repr(last_err) if last_err else "all AutoModel classes rejected",
    )


def _synthetic_pil_image(size: int = 224) -> Any:
    """Return a tiny RGB PIL image -- imported lazily so PIL stays optional
    until a VLM/OCR loader actually fires.
    """

    from PIL import Image  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    arr = np.random.randint(0, 255, size=(size, size, 3), dtype="uint8")
    return Image.fromarray(arr, mode="RGB")


def _load_hf_llm(
    transformers: Any,
    model_ref: str,
    model_kwargs: dict[str, Any],
    common_kwargs: dict[str, Any],
) -> tuple[Any, dict[str, Any]]:
    auto_cls = getattr(transformers, "AutoModelForCausalLM", None) or transformers.AutoModel
    tok_kwargs = dict(common_kwargs)
    tok_kwargs.pop("torch_dtype", None)

    # Prefer built-in transformers code over remote_code: well-known LLMs
    # (Phi-3, Llama, Qwen, etc.) have native implementations that track the
    # current rope_scaling schema, while bundled remote_code can be stale and
    # raise ``KeyError: 'type'`` against newer configs.
    builtin_kwargs = {**model_kwargs, "trust_remote_code": False}
    builtin_tok_kwargs = {**tok_kwargs, "trust_remote_code": False}
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_ref, **builtin_tok_kwargs)
        model = auto_cls.from_pretrained(model_ref, **builtin_kwargs)
    except (KeyError, ValueError):
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_ref, **tok_kwargs)
        model = auto_cls.from_pretrained(model_ref, **model_kwargs)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer("Hello, world.", return_tensors="pt")
    return model, dict(inputs)


def _load_hf_ocr(
    transformers: Any,
    model_ref: str,
    model_kwargs: dict[str, Any],
    common_kwargs: dict[str, Any],
) -> tuple[Any, dict[str, Any]]:
    proc_kwargs = dict(common_kwargs)
    proc_kwargs.pop("torch_dtype", None)

    # Try AutoProcessor; fall back to AutoTokenizer for OCRs without an image
    # processor (DeepSeek-OCR exposes only a custom tokenizer).
    processor: Any | None = None
    try:
        processor = transformers.AutoProcessor.from_pretrained(model_ref, **proc_kwargs)
    except (ValueError, KeyError):
        pass
    if processor is None:
        try:
            processor = transformers.AutoTokenizer.from_pretrained(model_ref, **proc_kwargs)
        except Exception:
            processor = None

    auto_classes: list[Any] = [
        getattr(transformers, "AutoModelForImageTextToText", None),
        getattr(transformers, "AutoModelForVision2Seq", None),
        getattr(transformers, "AutoModelForCausalLM", None),
        transformers.AutoModel,
    ]
    last_err: Exception | None = None
    model: Any | None = None
    for auto_cls in auto_classes:
        if auto_cls is None:
            continue
        try:
            model = auto_cls.from_pretrained(model_ref, **model_kwargs)
            break
        except (ValueError, KeyError) as exc:
            last_err = exc
            continue
    if model is None:
        # Dynamic-class fallback via config.auto_map.
        try:
            cfg = transformers.AutoConfig.from_pretrained(model_ref, **dict(common_kwargs, trust_remote_code=True))
            auto_map = getattr(cfg, "auto_map", {}) or {}
            class_ref = next(
                (auto_map[k] for k in (
                    "AutoModelForImageTextToText",
                    "AutoModelForVision2Seq",
                    "AutoModelForCausalLM",
                    "AutoModel",
                ) if k in auto_map),
                None,
            )
            if class_ref:
                from transformers.dynamic_module_utils import get_class_from_dynamic_module  # noqa: PLC0415

                cls = get_class_from_dynamic_module(class_ref, model_ref)
                model = cls.from_pretrained(model_ref, **model_kwargs)
        except Exception as exc:
            last_err = exc
    if model is None:
        raise LoaderUnavailable(
            AdmissionStatus.FAILED_EAGER,
            reason=f"no AutoModel head accepted {model_ref!r}",
            error=repr(last_err) if last_err else "all AutoModel classes rejected the config",
        )

    image = _synthetic_pil_image(size=224)
    if processor is not None:
        # Try AutoProcessor first.
        for attempt in (
            lambda: processor(images=image, return_tensors="pt"),
            lambda: processor(image, return_tensors="pt"),
            lambda: processor("describe", return_tensors="pt"),
        ):
            try:
                return model, dict(attempt())
            except (TypeError, ValueError, AttributeError):
                continue
    raise LoaderUnavailable(
        AdmissionStatus.FAILED_EAGER,
        reason=f"could not build OCR inputs for {model_ref!r}",
    )


def _load_hf_vlm(
    transformers: Any,
    model_ref: str,
    model_kwargs: dict[str, Any],
    common_kwargs: dict[str, Any],
) -> tuple[Any, dict[str, Any]]:
    proc_kwargs = dict(common_kwargs)
    proc_kwargs.pop("torch_dtype", None)
    processor: Any | None = None
    try:
        processor = transformers.AutoProcessor.from_pretrained(model_ref, **proc_kwargs)
    except (ValueError, KeyError):
        pass
    auto_classes: list[Any] = [
        getattr(transformers, "AutoModelForImageTextToText", None),
        getattr(transformers, "AutoModelForVision2Seq", None),
        getattr(transformers, "AutoModelForCausalLM", None),
        transformers.AutoModel,
    ]
    model: Any | None = None
    last_err: Exception | None = None
    for auto_cls in auto_classes:
        if auto_cls is None:
            continue
        try:
            model = auto_cls.from_pretrained(model_ref, **model_kwargs)
            break
        except (ValueError, KeyError) as exc:
            last_err = exc
            continue
    if model is None:
        # Dynamic-class fallback via config.auto_map (moondream2, etc.).
        try:
            cfg = transformers.AutoConfig.from_pretrained(model_ref, **dict(common_kwargs, trust_remote_code=True))
            auto_map = getattr(cfg, "auto_map", {}) or {}
            class_ref = next(
                (
                    auto_map[k]
                    for k in (
                        "AutoModelForImageTextToText",
                        "AutoModelForVision2Seq",
                        "AutoModelForCausalLM",
                        "AutoModel",
                    )
                    if k in auto_map
                ),
                None,
            )
            if class_ref:
                from transformers.dynamic_module_utils import get_class_from_dynamic_module  # noqa: PLC0415

                cls = get_class_from_dynamic_module(class_ref, model_ref)
                model = cls.from_pretrained(model_ref, **model_kwargs)
        except Exception as exc:
            last_err = exc
    if model is None:
        raise LoaderUnavailable(
            AdmissionStatus.FAILED_EAGER,
            reason=f"no AutoModel head accepted {model_ref!r}",
            error=repr(last_err) if last_err else "all AutoModel classes rejected the config",
        )
    if processor is None:
        # Some VLMs have only a tokenizer (no processor). Fall back to that
        # path so we can still build a single-image-qa-style probe input.
        try:
            processor = transformers.AutoTokenizer.from_pretrained(model_ref, **proc_kwargs)
        except Exception as exc:
            raise LoaderUnavailable(
                AdmissionStatus.FAILED_EAGER,
                reason=f"no processor or tokenizer found for {model_ref!r}",
                error=repr(exc),
            ) from exc
    image = _synthetic_pil_image(size=224)
    inputs = _build_vlm_inputs(processor, image)
    if inputs is None:
        raise LoaderUnavailable(
            AdmissionStatus.FAILED_EAGER,
            reason=f"could not build VLM inputs from processor for {model_ref!r}",
        )
    return model, dict(inputs)


def _build_vlm_inputs(processor: Any, image: Any) -> Any | None:
    """Try several common VLM input patterns and return the first that works.

    Order of attempts:

    1. ``processor.apply_chat_template(...)`` (most modern VLMs).
    2. Inline image token (``<image>describe``) -- LLaVA, SmolVLM, etc.
    3. ``<|image|>describe`` -- Qwen, IDEFICS variants.
    4. ``processor.image_token`` attribute, if present.
    5. Image-only.
    """

    msg = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "describe"}]}]
    try:
        out = processor.apply_chat_template(
            msg,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            images=[image],
        )
        if _has_input_ids(out):
            return out
    except (TypeError, ValueError, AttributeError):
        pass

    image_token = getattr(processor, "image_token", None) or "<image>"
    for prompt in (
        f"{image_token}describe",
        "<|image|>describe",
        "USER: <image>\ndescribe\nASSISTANT:",
        "describe",
    ):
        for kwargs in (
            dict(text=prompt, images=image, return_tensors="pt"),
            dict(images=image, text=prompt, return_tensors="pt"),
        ):
            try:
                out = processor(**kwargs)
                if _has_input_ids(out):
                    return out
            except (TypeError, ValueError, AttributeError):
                continue
    try:
        out = processor(images=image, return_tensors="pt")
        if _has_input_ids(out):
            return out
    except (TypeError, ValueError, AttributeError):
        pass
    return None


def _has_input_ids(out: Any) -> bool:
    """Most VLMs/LLMs need input_ids; reject inputs that lack them."""

    if out is None:
        return False
    keys = set(out.keys()) if hasattr(out, "keys") else set()
    return "input_ids" in keys or "pixel_values" in keys


def _load_hf_vla(
    transformers: Any,
    model_ref: str,
    model_kwargs: dict[str, Any],
    common_kwargs: dict[str, Any],
) -> tuple[Any, dict[str, Any]]:
    """VLAs are heterogeneous. Try processor-based, fall back to tokenizer-only."""

    # Try every reasonable AutoModel head; some VLAs (OpenVLA / Prismatic) have
    # custom configs that aren't registered for AutoModelForImageTextToText.
    auto_classes: list[Any] = [
        getattr(transformers, "AutoModelForImageTextToText", None),
        getattr(transformers, "AutoModelForVision2Seq", None),
        getattr(transformers, "AutoModelForCausalLM", None),
        transformers.AutoModel,
    ]
    last_err: Exception | None = None
    model = None
    for auto_cls in auto_classes:
        if auto_cls is None:
            continue
        try:
            model = auto_cls.from_pretrained(model_ref, **model_kwargs)
            break
        except (ValueError, KeyError) as exc:
            last_err = exc
            continue
    if model is None:
        # Last-resort: walk the config's ``auto_map`` and load via
        # ``get_class_from_dynamic_module``. Required for OpenVLA / Prismatic,
        # whose auto_map references ``AutoModelForVision2Seq`` -- a class that
        # was renamed to ``AutoModelForImageTextToText`` in transformers 5.x
        # and is no longer registered.
        try:
            cfg = transformers.AutoConfig.from_pretrained(model_ref, **dict(common_kwargs, trust_remote_code=True))
            auto_map = getattr(cfg, "auto_map", {}) or {}
            class_ref = next(
                (
                    auto_map[k]
                    for k in (
                        "AutoModelForImageTextToText",
                        "AutoModelForVision2Seq",
                        "AutoModelForCausalLM",
                        "AutoModel",
                    )
                    if k in auto_map
                ),
                None,
            )
            if class_ref:
                from transformers.dynamic_module_utils import get_class_from_dynamic_module  # noqa: PLC0415

                cls = get_class_from_dynamic_module(class_ref, model_ref)
                model = cls.from_pretrained(model_ref, **model_kwargs)
        except Exception as exc:
            last_err = exc
    if model is None:
        raise LoaderUnavailable(
            AdmissionStatus.FAILED_EAGER,
            reason=f"no AutoModel head accepted {model_ref!r}",
            error=repr(last_err) if last_err else "all AutoModel classes rejected the config",
        )
    proc_kwargs = dict(common_kwargs)
    proc_kwargs.pop("torch_dtype", None)
    image = _synthetic_pil_image(size=224)
    try:
        processor = transformers.AutoProcessor.from_pretrained(model_ref, **proc_kwargs)
    except Exception:
        processor = None
    if processor is not None:
        for attempt in (
            lambda: processor(images=image, text="pick up the cube", return_tensors="pt"),
            lambda: processor(image, "pick up the cube", return_tensors="pt"),
            lambda: processor(images=image, return_tensors="pt"),
        ):
            try:
                inputs = attempt()
                return model, dict(inputs)
            except (TypeError, ValueError):
                continue
    # tokenizer-only fallback (rare, but covers some action-only VLAs)
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_ref, **proc_kwargs)
    except Exception as exc:
        raise LoaderUnavailable(
            AdmissionStatus.FAILED_EAGER,
            reason=f"VLA model loaded but neither processor nor tokenizer worked for {model_ref!r}",
            error=repr(exc),
        ) from exc
    inputs = tokenizer("pick up the cube", return_tensors="pt")
    return model, dict(inputs)


# --------------------------------------------------------------------------- #
# Helpers.
# --------------------------------------------------------------------------- #


def _split_sample(sample: Any) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Normalise sample input convention.

    Accepts:
    - ``tuple`` -> positional args, no kwargs.
    - ``dict`` -> kwargs only (no positional).
    - ``(tuple, dict)`` -> both.
    """

    if isinstance(sample, tuple) and len(sample) == 2 and isinstance(sample[1], dict):
        args, kwargs = sample
        if not isinstance(args, tuple):
            args = (args,)
        return args, dict(kwargs)
    if isinstance(sample, dict):
        return (), dict(sample)
    if isinstance(sample, tuple):
        return sample, {}
    return (sample,), {}


__all__ = [
    "LoadedModel",
    "LoaderUnavailable",
    "load",
]
