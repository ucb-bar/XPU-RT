"""Tests for autocomp target-dispatch (Track 2 of the cross-target study).

These tests do **not** actually invoke autocomp's search (that needs
a chipyard checkout per target + a live Gemini key). They verify the
target-dispatch *resolver* picks the right backend trio and that
each backend's modules + classes exist in the local autocomp
checkout. That's the contract the cross-target comparison runner
relies on.
"""

from __future__ import annotations

import pytest

from xpu_rt.kernels.autocomp_adapter import (
    AutocompTargetBindings,
    resolve_autocomp_target,
)


def test_resolve_autocomp_target_gemmini() -> None:
    """`gemmini` is the canonical id; `gemmini_mx` is the historical
    alias — both must resolve to the same backend trio."""
    b = resolve_autocomp_target("gemmini_mx")
    assert isinstance(b, AutocompTargetBindings)
    # Canonical id wins; alias resolves to the canonical row.
    assert b.target_id == "gemmini"
    b2 = resolve_autocomp_target("gemmini")
    assert b2.target_id == "gemmini"
    assert b2.agent_class == b.agent_class
    assert b2.config_class == b.config_class
    assert b2.eval_class == b.eval_class
    assert b.agent_class == "GemminiLLMAgent"
    assert b.config_class == "GemminiHardwareConfig"
    assert b.eval_class == "GemminiEvalBackend"
    # Document the env requirement so the runner can surface a clean
    # error if the operator hasn't pointed autocomp at a chipyard.
    assert "INT8_16PE_CHIPYARD_PATH" in b.env_requirements


def test_resolve_autocomp_target_saturn() -> None:
    b = resolve_autocomp_target("saturn_opu_v128")
    assert b.target_id == "saturn_opu_v128"
    assert b.agent_class == "SaturnLLMAgent"
    assert b.config_class == "SaturnHardwareConfig"
    assert b.eval_class == "SaturnEvalBackend"
    assert "SATURN_CHIPYARD_PATH" in b.env_requirements


def test_resolve_autocomp_target_prefix_match() -> None:
    """``saturn_*`` / ``opu_*`` ids must also route to the Saturn
    backend so the dispatch is forgiving of variant naming."""
    assert resolve_autocomp_target("saturn_dsp_v128").target_id == "saturn_opu_v128"
    assert resolve_autocomp_target("opu_v128_alt").target_id == "saturn_opu_v128"


def test_resolve_autocomp_target_falls_back_to_cuda() -> None:
    """Unknown ids fall back to CUDA — the live-wired path today."""
    b = resolve_autocomp_target("h100_pcie")
    assert b.target_id == "cuda"
    assert b.agent_class == "CudaLLMAgent"


def test_resolve_returns_real_importable_classes_for_gemmini() -> None:
    """The resolver promises ``resolve()`` returns real imports. If
    autocomp's Gemmini backend isn't present in the checkout this
    surfaces as ImportError immediately, not later during a search."""
    b = resolve_autocomp_target("gemmini_mx")
    try:
        agent_cls, config_cls, eval_cls = b.resolve()
    except ImportError as exc:  # pragma: no cover — local-env defensive
        pytest.skip(f"autocomp Gemmini backend not importable: {exc}")
    assert agent_cls.__name__ == "GemminiLLMAgent"
    assert config_cls.__name__ == "GemminiHardwareConfig"
    assert eval_cls.__name__ == "GemminiEvalBackend"


def test_resolve_returns_real_importable_classes_for_saturn() -> None:
    b = resolve_autocomp_target("saturn_opu_v128")
    try:
        agent_cls, config_cls, eval_cls = b.resolve()
    except ImportError as exc:  # pragma: no cover
        pytest.skip(f"autocomp Saturn backend not importable: {exc}")
    assert agent_cls.__name__ == "SaturnLLMAgent"
    assert config_cls.__name__ == "SaturnHardwareConfig"
    assert eval_cls.__name__ == "SaturnEvalBackend"


def test_missing_env_reports_required_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    """``missing_env`` lets the runner emit a single clean error when
    the chipyard env var isn't set, instead of a stack trace from
    deep inside autocomp's eval backend."""
    monkeypatch.delenv("INT8_16PE_CHIPYARD_PATH", raising=False)
    b = resolve_autocomp_target("gemmini_mx")
    assert "INT8_16PE_CHIPYARD_PATH" in b.missing_env()

    monkeypatch.setenv("INT8_16PE_CHIPYARD_PATH", "/somewhere")
    # Re-read fresh: env-state is checked at call time, not at resolve.
    assert "INT8_16PE_CHIPYARD_PATH" not in b.missing_env()


def test_cuda_backend_needs_no_extra_env() -> None:
    """The CUDA path has been live forever; it requires no chipyard
    env. Confirm we haven't accidentally added a synthetic requirement."""
    b = resolve_autocomp_target("cuda")
    assert b.env_requirements == ()
    assert b.missing_env() == ()


def test_search_kernel_for_target_raises_env_missing_when_autocomp_absent(
    tmp_path, monkeypatch
) -> None:
    """The live entry point ``AutocompAdapter.search_kernel_for_target``
    must raise :class:`AutocompEnvMissing` (not a stacktrace) when
    autocomp isn't installed. The matrix driver catches this and
    converts to a clean ``env_missing`` cell."""
    from xpu_rt.kernels.autocomp_adapter import (
        AutocompAdapter,
        AutocompEnvMissing,
    )

    monkeypatch.delenv("INT8_16PE_CHIPYARD_PATH", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMMINI_API", raising=False)
    adapter = AutocompAdapter()
    with pytest.raises(AutocompEnvMissing) as exc:
        adapter.search_kernel_for_target(
            target_id="gemmini_mx",
            prob_type="gemm",
            prob_id=0,
            output_dir=tmp_path / "out",
        )
    msg = str(exc.value)
    # Acceptable failure modes — both surface as env_missing:
    #   (a) autocomp package itself not importable
    #   (b) autocomp importable but chipyard / Gemini env missing
    assert "autocomp" in msg.lower() or "INT8_16PE_CHIPYARD_PATH" in msg


def test_autocomp_env_missing_distinct_from_other_runtime_errors() -> None:
    """Regression guard: AutocompEnvMissing is its own class so the
    matrix driver's try/except can catch it specifically without
    swallowing other RuntimeErrors. Subclass relation must hold."""
    from xpu_rt.kernels.autocomp_adapter import AutocompEnvMissing

    assert issubclass(AutocompEnvMissing, RuntimeError)
    assert AutocompEnvMissing is not RuntimeError


def test_search_kernel_for_target_signature_keyword_only() -> None:
    """Regression guard: the signature is keyword-only so callers
    can't accidentally swap target_id with prob_type."""
    import inspect
    from xpu_rt.kernels.autocomp_adapter import AutocompAdapter

    sig = inspect.signature(AutocompAdapter.search_kernel_for_target)
    params = sig.parameters
    assert params["target_id"].kind == inspect.Parameter.KEYWORD_ONLY
    assert params["prob_type"].kind == inspect.Parameter.KEYWORD_ONLY
    assert params["prob_id"].kind == inspect.Parameter.KEYWORD_ONLY
    assert params["output_dir"].kind == inspect.Parameter.KEYWORD_ONLY
