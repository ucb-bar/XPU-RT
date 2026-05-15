"""Wave 1.8 — composition-aware matcher tests (bridge #102 unblocker).

The matcher previously rejected any model whose top-level
``named_children()`` didn't directly contain matching ``nn.Linear``
instances. Per bridge #102, paper-shape decoder layers wrap an FFN
inside a larger module — so the matcher saw "got 0 linears" and
gave up.

This test pins the fallback that walks ``named_modules()`` looking
for a submodule that reproduces the wrapper's forward output
+ matches a known pattern. Multi-block composition (attention +
FFN both contributing) is out of scope — that's Wave 2.1's MHA
matcher + stitcher.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn


class _FFN(nn.Module):
    """The matchable inner block."""

    def __init__(self, in_dim: int = 64, hidden: int = 128, out_dim: int = 64) -> None:
        super().__init__()
        self.up = nn.Linear(in_dim, hidden, bias=False)
        self.down = nn.Linear(hidden, out_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(torch.relu(self.up(x)))


class _ThinWrapper(nn.Module):
    """The bridge #102 case: outer module wraps a single FFN
    sub-block. ``named_children()`` returns ``ffn`` (an _FFN
    instance), not the underlying nn.Linears."""

    def __init__(self) -> None:
        super().__init__()
        self.ffn = _FFN()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn(x)


class _DiamondInner(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Linear(64, 32, bias=False)
        self.b = nn.Linear(64, 32, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (self.a(x) + self.b(x)).relu()


class _ThinDiamondWrapper(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.diamond = _DiamondInner()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.diamond(x)


class _NestedWrapper(nn.Module):
    """Two levels of wrapping. ``self.outer`` wraps ``self.outer.ffn``."""

    def __init__(self) -> None:
        super().__init__()
        self.outer = _ThinWrapper()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.outer(x)


class _StackedFFN(nn.Module):
    """The "two-block" case the matcher should NOT accept — two
    separate FFNs whose forward outputs differ from either's
    individual forward. Composition is Wave 2.1+."""

    def __init__(self) -> None:
        super().__init__()
        self.ffn1 = _FFN()
        self.ffn2 = _FFN()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn2(self.ffn1(x))


class TestThinWrapperMatch:
    """The fallback should descend one level for thin wrappers."""

    def test_thin_ffn_wrapper_matches(self) -> None:
        from compgen.runtime.lowering import lower_torch_to_megakernel

        model = _ThinWrapper()
        x = torch.randn(64, 64)
        result = lower_torch_to_megakernel(model, (x,))
        # Decision tag carries both the matched pattern and the
        # submodule path so the agent's audit query sees what was
        # lowered.
        assert "ffn@ffn" == result.decision.pattern_name or (
            "ffn" in result.decision.pattern_name and "@ffn" in result.decision.pattern_name
        )
        # Rationale explains the thin-wrapper case explicitly.
        assert "thin-wrapper" in result.decision.pattern_rationale

    def test_thin_diamond_wrapper_matches(self) -> None:
        from compgen.runtime.lowering import lower_torch_to_megakernel

        result = lower_torch_to_megakernel(
            _ThinDiamondWrapper(),
            (torch.randn(64, 64),),
        )
        assert "diamond" in result.decision.pattern_name
        assert "@diamond" in result.decision.pattern_name

    def test_nested_wrapper_walks_through(self) -> None:
        """Two-level wrapper: outer.ffn must still resolve. Submodule
        walk visits at every depth."""
        from compgen.runtime.lowering import lower_torch_to_megakernel

        result = lower_torch_to_megakernel(
            _NestedWrapper(),
            (torch.randn(64, 64),),
        )
        assert "ffn" in result.decision.pattern_name


class TestUnsupportedComposition:
    """The fallback should NOT accept multi-block composition where
    the wrapper's forward DOESN'T equal a single sub-block's output.
    That requires real composition (Wave 2.1+)."""

    def test_stacked_ffn_rejects(self) -> None:
        from compgen.runtime.lowering import (
            UnsupportedShape,
            lower_torch_to_megakernel,
        )

        # Stacked FFN: forward = ffn2(ffn1(x)). Neither ffn1 alone
        # nor ffn2 alone reproduces the top-level forward. With the
        # generic FX fallback disabled, the matcher cascade rejects
        # — pin that path here so future drift in pattern coverage
        # doesn't silently change the cascade's terminal behavior.
        # (When the generic fallback is enabled per Wave 2.2, stacked
        # FFN lowers via :func:`lower_generic_fx`; that path is
        # exercised in the Wave 2.2 tests.)
        with pytest.raises(UnsupportedShape) as exc_info:
            lower_torch_to_megakernel(
                _StackedFFN(),
                (torch.randn(64, 64),),
                allow_generic_fallback=False,
            )
        # Error message should include the submodule walk's failure
        # so the agent knows composition is needed.
        assert "submodule" in str(exc_info.value).lower() or "no FX-graph pattern matched" in str(exc_info.value)

    def test_top_level_match_still_takes_priority(self) -> None:
        """If the top-level model itself matches diamond/FFN, the
        submodule fallback is never reached. Pin the matcher
        precedence."""
        from compgen.runtime.lowering import lower_torch_to_megakernel

        # _DiamondInner matches diamond directly; pattern_name has
        # no "@" suffix because the top-level matcher accepted.
        result = lower_torch_to_megakernel(
            _DiamondInner(),
            (torch.randn(64, 64),),
        )
        assert result.decision.pattern_name == "diamond"
        assert "@" not in result.decision.pattern_name


class TestSubmodulePathPlumbing:
    """Wave 1.8 dispatch fix per bridge #108.

    The matcher's submodule fallback used to compile fine but
    dispatch crashed with `AttributeError: 'ThinWrapper' object
    has no attribute 'up'` because the runtime's
    ``_workload_buffers`` looked for ``model.up.weight`` on the
    wrapper. The fix records ``decision.submodule_path`` so the
    compile path pickles the SUBMODULE (not the wrapper) — at
    dispatch the bundle finds ``up`` directly.
    """

    def test_submodule_path_recorded_on_decision(self) -> None:
        from compgen.runtime.lowering import lower_torch_to_megakernel

        result = lower_torch_to_megakernel(
            _ThinWrapper(),
            (torch.randn(64, 64),),
        )
        assert result.decision.submodule_path == "ffn"
        d = result.decision.to_dict()
        assert d["submodule_path"] == "ffn"

    def test_top_level_match_has_empty_submodule_path(self) -> None:
        """When the top-level model itself matches, submodule_path
        is empty so dispatch uses the model as-is."""
        from compgen.runtime.lowering import lower_torch_to_megakernel

        result = lower_torch_to_megakernel(
            _DiamondInner(),
            (torch.randn(64, 64),),
        )
        assert result.decision.submodule_path == ""

    def test_compile_to_megakernel_pickles_submodule_for_thin_wrapper(self, tmp_path) -> None:
        """The bundle's compile_context.json must contain the
        SUBMODULE pickle so dispatch's weight-extraction sees a
        model with the linears at top level. Per bridge #108: this
        is the AttributeError fix."""
        import base64
        import json
        import pickle

        import compgen

        bundle = compgen.compile_to_megakernel(
            _ThinWrapper(),
            (torch.randn(64, 64),),
            output_dir=str(tmp_path),
        )
        ctx = json.loads((bundle.bundle_dir / "compile_context.json").read_text())
        # The pickled model is the FFN submodule, not the wrapper.
        loaded_model = pickle.loads(base64.b64decode(ctx["model_pickle_b64"]))
        assert hasattr(loaded_model, "up"), (
            "submodule fix: pickled model must have .up directly "
            "(not nested under a wrapper) so dispatch's "
            "_workload_buffers can read .up.weight."
        )
        assert hasattr(loaded_model, "down")
        # Audit trail: original wrapper class + submodule path
        # surface in the context for the agent's audit query.
        assert ctx["submodule_path"] == "ffn"
        assert ctx["wrapper_class"] == "_ThinWrapper"


class TestForwardEquivalenceCheck:
    """The fallback verifies that the submodule reproduces the
    wrapper's output. A submodule with matching shape but different
    output (e.g., the wrapper applies a post-op) must be rejected."""

    def test_wrapper_with_post_op_rejects(self) -> None:
        from compgen.runtime.lowering import (
            UnsupportedShape,
            lower_torch_to_megakernel,
        )

        class _WrapperWithPostScale(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.ffn = _FFN()

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # Post-scales the FFN output. The submodule's forward
                # does NOT equal the wrapper's forward.
                return self.ffn(x) * 2.0

        with pytest.raises(UnsupportedShape):
            lower_torch_to_megakernel(
                _WrapperWithPostScale(),
                (torch.randn(64, 64),),
            )
