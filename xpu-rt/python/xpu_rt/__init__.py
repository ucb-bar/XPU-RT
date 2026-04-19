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

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "CompGenDevice",
    "CompiledModel",
    "LLMCompileResult",
    "compile_model",
    "compile_with_llm",
    "device",
    "open_llm_session",
]


def __getattr__(name: str):
    """Lazily expose the top-level API without importing heavy deps at package import time."""

    if name in {"CompGenDevice", "CompiledModel", "compile_model", "device"}:
        from compgen.api import CompGenDevice, CompiledModel, compile_model, device

        exports = {
            "CompGenDevice": CompGenDevice,
            "CompiledModel": CompiledModel,
            "compile_model": compile_model,
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
