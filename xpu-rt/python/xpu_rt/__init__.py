"""CompGen -- an LLM-driven compiler generator for heterogeneous hardware targets.

CompGen is a *compiler generator*, not a compiler. Given a PyTorch program and a
hardware profile (one or more devices), it generates a deployment recipe containing:

- Graph/lowering transforms (MLIR Transform Dialect scripts)
- Missing custom kernels (via Autocomp/Triton search loops)
- Placement and scheduling decisions
- Packaging and runtime artifacts
- Verification reports

Only verified artifacts get promoted into a deterministic recipe library.

Architecture invariant: the LLM is a proposal engine. It generates bounded,
declarative artifacts (transform scripts, kernel recipes, policies). Deterministic
compiler infrastructure executes them, and verification decides what ships.
"""

from __future__ import annotations

__version__ = "0.2.0"


def _install_gemini_instrumentation_at_import() -> None:
    """Install the google.genai monkey-patch ASAP so any caller in this
    process — agent_decision, MCP tools, efficiency_report, etc. —
    has its Gemini API calls captured by ``compgen-gemini-usage``.

    Opt-out via ``COMPGEN_DISABLE_GEMINI_INSTRUMENTATION=1`` for users
    who run their own observability stack. Best-effort: never raises.
    Subprocess paths (KernelBlaster) still bypass this — for those use
    ``scripts/dev/recover_kernelblaster_usage.py`` (post-hoc recovery).
    """
    import os

    if os.environ.get("COMPGEN_DISABLE_GEMINI_INSTRUMENTATION", "").lower() in (
        "1",
        "true",
        "yes",
    ):
        return
    try:
        from compgen.observability.gemini_usage import (
            install_genai_instrumentation,
            install_openai_instrumentation,
        )

        install_genai_instrumentation()
        # OpenAI-compat SDK pointed at Google's endpoint (KB path).
        install_openai_instrumentation()
    except Exception:  # noqa: BLE001
        # Observability is best-effort; never crash a compile.
        pass


_install_gemini_instrumentation_at_import()

__all__ = [
    "__version__",
    "CompGenDevice",
    "CompiledModel",
    "LLMCompileResult",
    "MegakernelBundle",
    "compile_model",
    "compile_to_megakernel",
    "compile_with_llm",
    "device",
    "has_cuda_runtime",
    "open_llm_session",
]


def has_cuda_runtime() -> bool:
    """Return True if a CUDA-enabled libcompgen_rt is available.

    Used by the remote-Blackwell sanity check
    (``python -c 'import compgen; assert compgen.has_cuda_runtime()'``).
    Returns ``False`` on dev installs without the prebuilt CUDA library
    or hosts where ``torch.cuda.is_available()`` returns False.
    """
    try:
        import torch
    except Exception:
        return False
    if not torch.cuda.is_available():
        return False
    # Existence of the prebuilt CUDA library is the second precondition.
    # The CPU-only library exists on every install; only the CUDA variant
    # is gated by the wheel's ``[cuda]`` extra.
    from pathlib import Path

    prebuilt_dir = Path(__file__).parent / "runtime" / "native" / "prebuilt"
    cuda_lib = list(prebuilt_dir.glob("libcompgen_rt-cuda*.so"))
    return len(cuda_lib) > 0


def __getattr__(name: str):
    """Lazily expose the top-level API without importing heavy deps at package import time."""

    if name in {
        "CompGenDevice",
        "CompiledModel",
        "MegakernelBundle",
        "compile_model",
        "compile_to_megakernel",
        "device",
    }:
        from compgen.api import (
            CompGenDevice,
            CompiledModel,
            MegakernelBundle,
            compile_model,
            compile_to_megakernel,
            device,
        )

        exports = {
            "CompGenDevice": CompGenDevice,
            "CompiledModel": CompiledModel,
            "MegakernelBundle": MegakernelBundle,
            "compile_model": compile_model,
            "compile_to_megakernel": compile_to_megakernel,
            "device": device,
        }
        return exports[name]
    if name in {"compile_with_llm", "open_llm_session", "LLMCompileResult"}:
        from compgen.api_llm import LLMCompileResult, compile_with_llm, open_llm_session

        exports = {
            "compile_with_llm": compile_with_llm,
            "open_llm_session": open_llm_session,
            "LLMCompileResult": LLMCompileResult,
        }
        return exports[name]
    raise AttributeError(name)
