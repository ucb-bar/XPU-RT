"""Wave 2.3 — user-dialect / custom-lowering entrypoint tests.

The agentic-compilation contract: a PyPI user (or their agent) can
plug in a custom MLIR dialect (cuda-tile, etc.) by registering a
lowering function. The matcher cascade tries user lowerings BEFORE
the built-in diamond/FFN matchers, so domain-specific patterns
take priority.

Two registration paths:

1. ``compgen.plugins.register(GROUP_LOWERINGS, name, fn)`` —
   in-process; the agent calls this once at session start.
2. Entry-point declared in the user's ``pyproject.toml`` under
   ``compgen.runtime.lowerings`` — auto-discovered.

These tests cover path 1 (path 2 is plumbed through the same
registry, so the validator coverage applies to both).
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn


def _setup_method(self):  # noqa: D401 — pytest setup
    """Reset the plugin registry before each test so user lowerings
    don't leak across cases."""
    from compgen.plugins import reset_registry

    reset_registry()


class _Diamond(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Linear(64, 32, bias=False)
        self.b = nn.Linear(64, 32, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (self.a(x) + self.b(x)).relu()


class TestPluginGroupRegistration:
    setup_method = _setup_method

    def test_lowerings_group_registered(self) -> None:
        from compgen.plugins import GROUP_LOWERINGS, KNOWN_GROUPS

        assert GROUP_LOWERINGS == "compgen.runtime.lowerings"
        assert GROUP_LOWERINGS in KNOWN_GROUPS

    def test_register_validates_callable(self) -> None:
        """The validator must reject non-callable entries — the
        registry should never load a user lowering that can't be
        invoked."""
        from compgen.plugins import GROUP_LOWERINGS, register

        with pytest.raises(ValueError, match="callable"):
            register(GROUP_LOWERINGS, "bad", "not-a-callable")  # type: ignore[arg-type]

    def test_register_validates_signature(self) -> None:
        """The validator must reject lowerings that don't accept
        (model, sample_inputs)."""
        from compgen.plugins import GROUP_LOWERINGS, register

        def too_few_args(x):
            return x

        with pytest.raises(ValueError, match="positional"):
            register(GROUP_LOWERINGS, "bad-sig", too_few_args)


class TestMatcherCascade:
    """End-to-end: register a user lowering and verify it runs
    before the built-in matchers."""

    setup_method = _setup_method

    def test_user_lowering_runs_before_builtins(self) -> None:
        """When a user lowering matches the model, it wins over
        the built-in diamond matcher even though both would accept
        the same shape."""
        from compgen.plugins import GROUP_LOWERINGS, register
        from compgen.runtime.lowering import (
            LoweringDecision,
            LoweringResult,
            lower_torch_to_megakernel,
        )

        sentinel = {"matched": False}

        def user_lowering(model, sample_inputs, *, backend_choice=None):
            sentinel["matched"] = True
            # Must return a real LoweringResult — delegate to the
            # built-in diamond matcher and mutate the decision so
            # we can verify in the test that this path ran.
            from compgen.runtime.lowering.fx_to_megakernel import (
                _match_diamond,
            )

            inner = _match_diamond(model, sample_inputs)
            patched_decision = LoweringDecision(
                pattern_name=f"user:{inner.decision.pattern_name}",
                pattern_rationale="user-lowering test sentinel",
                body_decisions=inner.decision.body_decisions,
                schedule_hints=inner.decision.schedule_hints,
                total_tile_tasks=inner.decision.total_tile_tasks,
                backends=inner.decision.backends,
                nvrtc_include_paths=inner.decision.nvrtc_include_paths,
                nvrtc_extra_options=inner.decision.nvrtc_extra_options,
            )
            return LoweringResult(
                megakernel_graph=inner.megakernel_graph,
                device_function_sources=inner.device_function_sources,
                user_buffer_layout=inner.user_buffer_layout,
                decision=patched_decision,
            )

        register(GROUP_LOWERINGS, "test-cuda-tile", user_lowering)

        result = lower_torch_to_megakernel(
            _Diamond(),
            (torch.randn(64, 64),),
        )
        assert sentinel["matched"], "user lowering was not invoked"
        # Pattern name carries the user prefix so the agent's
        # decision-log query distinguishes built-in vs user paths.
        assert result.decision.pattern_name.startswith("user:")

    def test_user_lowering_unsupported_falls_through_to_builtins(
        self,
    ) -> None:
        """When a user lowering raises ``UnsupportedShape``, the
        cascade continues to the built-in matchers. This is the
        expected behavior — user lowerings are tried first but
        don't block built-ins."""
        from compgen.plugins import GROUP_LOWERINGS, register
        from compgen.runtime.lowering import (
            UnsupportedShape,
            lower_torch_to_megakernel,
        )

        def user_doesnt_match(model, sample_inputs, *, backend_choice=None):
            raise UnsupportedShape("user pattern doesn't match this model")

        register(GROUP_LOWERINGS, "user-noop", user_doesnt_match)

        # Built-in diamond matcher handles _Diamond, so the cascade
        # should land there cleanly.
        result = lower_torch_to_megakernel(
            _Diamond(),
            (torch.randn(64, 64),),
        )
        assert result.decision.pattern_name == "diamond"

    def test_user_lowering_error_continues_cascade(self) -> None:
        """If a user lowering raises an unexpected exception (not
        UnsupportedShape), the error is recorded but the cascade
        continues. Stops one bad plugin from blocking everything."""
        from compgen.plugins import GROUP_LOWERINGS, register
        from compgen.runtime.lowering import lower_torch_to_megakernel

        def user_buggy(model, sample_inputs, *, backend_choice=None):
            raise ValueError("intentional bug for test")

        register(GROUP_LOWERINGS, "user-buggy", user_buggy)

        result = lower_torch_to_megakernel(
            _Diamond(),
            (torch.randn(64, 64),),
        )
        # Built-in diamond still wins despite user lowering's bug.
        assert result.decision.pattern_name == "diamond"

    def test_no_user_lowerings_uses_builtins(self) -> None:
        """The default path: no user lowerings registered. Built-in
        matchers run as before."""
        from compgen.runtime.lowering import lower_torch_to_megakernel

        result = lower_torch_to_megakernel(
            _Diamond(),
            (torch.randn(64, 64),),
        )
        assert result.decision.pattern_name == "diamond"

    def test_user_lowering_signature_without_backend_choice(self) -> None:
        """For backwards compat with simpler user lowerings that
        don't accept ``backend_choice``, the matcher retries
        without that kwarg before giving up."""
        from compgen.plugins import GROUP_LOWERINGS, register
        from compgen.runtime.lowering import (
            UnsupportedShape,
            lower_torch_to_megakernel,
        )

        def simple_user(model, sample_inputs):
            # Doesn't accept backend_choice. Matcher should retry
            # without it.
            raise UnsupportedShape("simple matcher doesn't match")

        register(GROUP_LOWERINGS, "simple-user", simple_user)

        # Should fall through cleanly to built-in matchers — no
        # TypeError surfaced from the kwarg mismatch.
        result = lower_torch_to_megakernel(
            _Diamond(),
            (torch.randn(64, 64),),
        )
        assert result.decision.pattern_name == "diamond"
