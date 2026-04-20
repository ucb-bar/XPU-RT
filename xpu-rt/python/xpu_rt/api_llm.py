"""LLM-driven Python entry point.

:func:`compile_with_llm` is the counterpart to :func:`compgen.api.compile_model`
that lets a user point any registered LLM provider at an arbitrary model
and have the agentic loop drive compilation end-to-end.

Example::

    from compgen import compile_with_llm
    compiled = compile_with_llm(
        model=my_torch_model,
        target="examples/target_profiles/cuda_a100.yaml",
        llm="gemini",
        sample_inputs=(torch.randn(1, 64),),
        budget=10,
    )
    result = compiled(torch.randn(1, 64))   # benchmark

The function is a thin wrapper on top of the existing
:func:`compgen.api.device` + :func:`compgen.api.compile_model` + the
:class:`~compgen.agent.loop.AgenticCompilationLoop` — it never
reimplements the loop; it just plumbs the LLM backend through.

Advanced callers may drive one step at a time via
:class:`~compgen.agent.llm_driver.LLMDrivenCompiler` — the same backbone
the MCP server uses. :func:`open_llm_session` exposes that handle.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import structlog
import torch
import torch.nn as nn

from compgen.agent.llm_driver import LLMDrivenCompiler
from compgen.agent.loop import CompilationResult
from compgen.api import CompGenDevice, CompiledModel, compile_model
from compgen.api import device as _device
from compgen.llm.base import CompGenLLMProtocol
from compgen.llm.recorder import LLMRecorder

log = structlog.get_logger()


# ---------------------------------------------------------------------------
# Result envelope
# ---------------------------------------------------------------------------


@dataclass
class LLMCompileResult:
    """Everything ``compile_with_llm`` returns.

    ``compiled`` is the standard :class:`CompiledModel`; ``llm_result``
    is populated when an agentic loop actually ran, and ``driver`` is
    the :class:`LLMDrivenCompiler` instance when the caller asked us
    to keep the session open.
    """

    compiled: CompiledModel
    llm_result: CompilationResult | None = None
    driver: LLMDrivenCompiler | None = None
    provider: str = ""
    transcript_dir: Path | None = None

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Forward to ``CompiledModel.__call__`` so the result is benchmark-ready."""
        return self.compiled(*args, **kwargs)


# ---------------------------------------------------------------------------
# Resolution helpers
# ---------------------------------------------------------------------------


def _resolve_target(target: str | Path | CompGenDevice) -> CompGenDevice:
    if isinstance(target, CompGenDevice):
        return target
    target_path = Path(target)
    if not target_path.exists():
        raise FileNotFoundError(
            f"Target profile not found: {target_path}. "
            f"Pass either a path to a hardware spec YAML or an already-built "
            f"CompGenDevice."
        )
    return _device(target_path)


def _resolve_llm(
    llm: str | CompGenLLMProtocol,
) -> tuple[CompGenLLMProtocol, str]:
    """Return (client, provider_name). Accepts a client instance or a provider name.

    The special value ``"mock"`` returns a :class:`MockLLMClient`, which is
    useful for offline smoke tests and MCP demos that must not require real
    API credentials.
    """
    if isinstance(llm, str):
        if llm.strip().lower() == "mock":
            from compgen.llm.mock_client import MockLLMClient

            return MockLLMClient(strict=False), "mock"
        from compgen.llm.factory import create_llm_client, resolve_provider_name

        provider = resolve_provider_name(llm)
        return create_llm_client(provider), provider
    # Duck-typed client already — accept if it looks like a CompGenLLMProtocol.
    name = type(llm).__name__
    return llm, name


def _resolve_model(
    model: str | Path | nn.Module,
    sample_inputs: tuple[Any, ...] | None,
) -> tuple[nn.Module, tuple[Any, ...]]:
    """Turn the ``model`` arg into an ``(nn.Module, sample_inputs)`` pair.

    Supported forms:

    * ``nn.Module`` — returned as-is.
    * ``str | Path`` pointing to a Python file — imports it and calls
      ``build_model()`` which must return ``(module, sample_inputs)``.
    * ``str`` that looks like a HF repo id (``"org/name"``) — optional,
      only available when ``transformers`` is importable.
    """
    if isinstance(model, nn.Module):
        if sample_inputs is None:
            raise ValueError("sample_inputs must be provided when passing a raw nn.Module.")
        return model, sample_inputs

    if isinstance(model, (str, Path)):
        as_str = str(model)
        as_path = Path(as_str)
        if as_path.exists() and as_path.suffix == ".py":
            return _load_model_from_python_file(as_path, sample_inputs)
        if "/" in as_str and not as_path.exists():
            # Looks like a HF repo id; try transformers.
            return _load_hf_model(as_str, sample_inputs)

    raise TypeError(
        f"Unsupported model argument: {model!r}. Pass an nn.Module, a "
        f"path to a .py file exposing build_model(), or a HuggingFace "
        f"repo id."
    )


def _load_model_from_python_file(
    path: Path,
    sample_inputs: tuple[Any, ...] | None,
) -> tuple[nn.Module, tuple[Any, ...]]:
    import importlib.util

    spec = importlib.util.spec_from_file_location("compgen_user_model", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot build spec for {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "build_model"):
        raise AttributeError(f"{path} must define build_model() -> (nn.Module, sample_inputs)")
    result = mod.build_model()
    if not (isinstance(result, tuple) and len(result) == 2):
        raise ValueError(f"{path}.build_model() must return a (module, sample_inputs) tuple")
    module, inferred_inputs = result
    if sample_inputs is not None:
        inferred_inputs = sample_inputs
    if not isinstance(inferred_inputs, tuple):
        inferred_inputs = (inferred_inputs,)
    return module, inferred_inputs


def _load_hf_model(
    hf_id: str,
    sample_inputs: tuple[Any, ...] | None,
) -> tuple[nn.Module, tuple[Any, ...]]:
    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "transformers is required to load HF models by repo id; "
            "install with `pip install compgen[demo]` or pass an nn.Module "
            "directly."
        ) from exc
    model = AutoModel.from_pretrained(hf_id)
    model.eval()
    if sample_inputs is None:
        try:
            tok = AutoTokenizer.from_pretrained(hf_id)
            enc = tok("CompGen sample input", return_tensors="pt")
            sample_inputs = (enc["input_ids"],)
        except Exception:
            # Fallback: generic int tensor. User can always override.
            sample_inputs = (torch.randint(0, 100, (1, 8), dtype=torch.long),)
    return model, sample_inputs


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def compile_with_llm(
    model: str | Path | nn.Module,
    target: str | Path | CompGenDevice,
    *,
    llm: str | CompGenLLMProtocol = "gemini",
    sample_inputs: tuple[Any, ...] | None = None,
    objective: str = "latency",
    budget: int = 10,
    recover_unsupported: bool = False,  # P2 feature; accepted but no-op in P1
    with_recipe: bool = True,
    transcript_dir: str | Path | None = None,
    return_driver: bool = False,
) -> LLMCompileResult:
    """Compile ``model`` for ``target`` driven by ``llm``.

    Args:
        model: A PyTorch ``nn.Module``, a path to a ``.py`` file defining
            ``build_model()``, or a HuggingFace repo id (when
            ``transformers`` is installed).
        target: A path to a hardware-spec YAML, or an already-built
            :class:`CompGenDevice`.
        llm: Provider name (``"gemini" | "openai" | "anthropic" |
            "claude-cli" | "codex-cli"``) or an instance satisfying
            :class:`CompGenLLMProtocol`. Resolved through
            :func:`compgen.llm.factory.create_llm_client`.
        sample_inputs: Tuple of sample tensors — required when
            ``model`` is a raw ``nn.Module``.
        objective: Optimisation objective, passed straight to
            :func:`compgen.api.compile_model`.
        budget: Max LLM-driven iterations during the agentic loop.
        recover_unsupported: When True, unsupported operators detected
            during capture are routed through
            :func:`compgen.agent.llm_driver_recovery.plan_recovery`; the
            LLM is consulted on low-confidence classifications and the
            decisions land on ``compiled.recovery_plan``.
        with_recipe: Use :meth:`AgenticCompilationLoop.run_with_recipe`
            (default, recommended) vs plain :meth:`run`.
        transcript_dir: Where to write recorder logs. Defaults to
            ``~/.compgen/transcripts/<session>``.
        return_driver: When True, keep the :class:`LLMDrivenCompiler`
            session open and return it — letting the caller inspect /
            drive further steps after the autonomous loop converges.

    Returns:
        :class:`LLMCompileResult` — callable like :class:`CompiledModel`.
    """
    log.info(
        "api_llm.start",
        model=type(model).__name__ if isinstance(model, nn.Module) else str(model),
        target=str(target),
        llm=llm if isinstance(llm, str) else type(llm).__name__,
    )

    mod, inputs = _resolve_model(model, sample_inputs)
    dev = _resolve_target(target)
    client, provider = _resolve_llm(llm)

    # Stage 1: the usual deterministic pipeline via compile_model.
    # When recover_unsupported=True, compile_model consults the LLM on
    # ambiguous unsupported-op classifications before the FX→xDSL import.
    compiled = compile_model(
        mod,
        dev,
        objective=objective,
        sample_inputs=inputs,
        recover_unsupported=recover_unsupported,
        recovery_llm_client=client if recover_unsupported else None,
    )
    log.info(
        "api_llm.compile_model.done",
        pipeline_passed=compiled.pipeline_result.passed,
        recovery_ok=(compiled.recovery_plan.ok() if compiled.recovery_plan is not None else None),
    )

    # Stage 2: wrap the LLM client with a recorder scoped to this run.
    # Honour COMPGEN_SESSION_DIR so tests + containerised users can
    # redirect transcripts away from ~/.compgen.
    import os

    if transcript_dir is not None:
        transcript_root = Path(transcript_dir).expanduser()
    elif os.environ.get("COMPGEN_SESSION_DIR"):
        transcript_root = Path(os.environ["COMPGEN_SESSION_DIR"]).expanduser()
    else:
        transcript_root = Path("~/.compgen/transcripts").expanduser()
    transcript_root.mkdir(parents=True, exist_ok=True)
    recorder = LLMRecorder(
        wrapped=client,
        log_dir=transcript_root / "llm",
        enabled=True,
    )

    # Stage 3: run the agentic loop. We use the existing
    # CompiledModel.run_agentic helper so we don't reimplement the loop.
    llm_result: CompilationResult | None = None
    driver: LLMDrivenCompiler | None = None
    try:
        llm_result = compiled.run_agentic(
            recorder,
            budget=budget,
            with_recipe=with_recipe,
        )
        log.info(
            "api_llm.loop.done", iterations=llm_result.iterations_run, improvement_pct=llm_result.total_improvement_pct
        )
    except Exception as e:  # noqa: BLE001
        log.warning("api_llm.loop.failed", error=str(e))

    # Stage 4: optionally leave a driver session open for advanced callers.
    if return_driver:
        env = compiled.create_agent_env(budget=budget)
        driver = LLMDrivenCompiler(
            env=env,
            target=dev.profile,
            llm_client=recorder,
            transcript_dir=transcript_root,
            budget=budget,
        )

    return LLMCompileResult(
        compiled=compiled,
        llm_result=llm_result,
        driver=driver,
        provider=provider,
        transcript_dir=transcript_root,
    )


def open_llm_session(
    model: nn.Module,
    target: str | Path | CompGenDevice,
    *,
    llm: str | CompGenLLMProtocol = "gemini",
    sample_inputs: tuple[Any, ...] | None = None,
    objective: str = "latency",
    budget: int = 10,
    transcript_dir: str | Path | None = None,
) -> LLMDrivenCompiler:
    """Open a live LLM-driven session and return the driver handle.

    Unlike :func:`compile_with_llm`, this does *not* run the autonomous
    loop — it just prepares the environment and returns the driver so
    the caller (typically the MCP server) can step through one action
    at a time.
    """
    mod, inputs = _resolve_model(model, sample_inputs)
    dev = _resolve_target(target)
    client, _ = _resolve_llm(llm)

    compiled = compile_model(mod, dev, objective=objective, sample_inputs=inputs)
    env = compiled.create_agent_env(budget=budget)
    return LLMDrivenCompiler(
        env=env,
        target=dev.profile,
        llm_client=client,
        transcript_dir=(Path(transcript_dir).expanduser() if transcript_dir is not None else None),
        budget=budget,
    )


__all__ = [
    "LLMCompileResult",
    "compile_with_llm",
    "open_llm_session",
]
