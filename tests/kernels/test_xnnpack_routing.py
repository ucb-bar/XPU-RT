"""Tests for XNNPACK's place in the provider auction.

The card at ``xpu-rt/python/xpu_rt/providers/cards/xnnpack.yaml`` plus
the KIND_PREFERENCE edit in ``provider_routing.py`` should put xnnpack
ahead of cffi_c (and the GPU-only options) for the CPU operator
families it covers — matmul, conv, depthwise_conv, pool, softmax,
unary/binary, etc.
"""

from __future__ import annotations

import pytest

from xpu_rt.providers.provider_routing import KIND_PREFERENCE, route_for


# ----- Static KIND_PREFERENCE contract ------------------------------


def test_kind_preference_lists_xnnpack_first_for_matmul_host_cpu() -> None:
    assert KIND_PREFERENCE["matmul"][0] == "xnnpack"


def test_kind_preference_lists_xnnpack_first_for_conv() -> None:
    assert KIND_PREFERENCE["conv"][0] == "xnnpack"
    assert KIND_PREFERENCE["conv2d"][0] == "xnnpack"


def test_kind_preference_lists_xnnpack_first_for_depthwise_conv() -> None:
    assert KIND_PREFERENCE["depthwise_conv2d"][0] == "xnnpack"
    assert KIND_PREFERENCE["depthwise_conv"][0] == "xnnpack"


def test_kind_preference_lists_xnnpack_first_for_softmax_and_pool() -> None:
    assert KIND_PREFERENCE["softmax"][0] == "xnnpack"
    assert KIND_PREFERENCE["avg_pool"][0] == "xnnpack"
    assert KIND_PREFERENCE["max_pool"][0] == "xnnpack"


def test_kind_preference_keeps_triton_first_for_attention_and_fused() -> None:
    # XNNPACK does NOT win attention / fused_region — those still go
    # through triton / thunderkittens / autocomp first.
    assert KIND_PREFERENCE["attention"][0] != "xnnpack"
    assert KIND_PREFERENCE["fused_region"][0] != "xnnpack"


# ----- End-to-end auction ordering ---------------------------------


def test_route_for_matmul_host_cpu_picks_xnnpack_first() -> None:
    ordered = route_for(contract_kind="matmul", target_family="host_cpu")
    assert "xnnpack" in ordered, (
        f"xnnpack missing from route_for(matmul, host_cpu): {ordered}"
    )
    # cffi_c is the deterministic anchor; xnnpack must rank ahead of it.
    if "cffi_c" in ordered:
        assert ordered.index("xnnpack") < ordered.index("cffi_c"), (
            f"xnnpack must outrank cffi_c on host_cpu matmul: {ordered}"
        )


def test_route_for_conv2d_host_cpu_picks_xnnpack_first() -> None:
    ordered = route_for(contract_kind="conv2d", target_family="host_cpu")
    assert "xnnpack" in ordered
    if "cffi_c" in ordered:
        assert ordered.index("xnnpack") < ordered.index("cffi_c")


def test_route_for_softmax_host_cpu_includes_xnnpack() -> None:
    ordered = route_for(contract_kind="softmax", target_family="host_cpu")
    assert "xnnpack" in ordered


def test_route_for_attention_host_cpu_excludes_xnnpack() -> None:
    """XNNPACK doesn't ship an attention kernel; the auction must
    not list it for that kind."""
    ordered = route_for(contract_kind="attention", target_family="host_cpu")
    assert "xnnpack" not in ordered, (
        f"xnnpack should not be routed to attention: {ordered}"
    )


def test_route_for_xnnpack_not_offered_on_cuda() -> None:
    """xnnpack is host_cpu only — must never appear on cuda routing."""
    for kind in ("matmul", "conv2d", "softmax"):
        ordered = route_for(contract_kind=kind, target_family="cuda")
        assert "xnnpack" not in ordered, (
            f"xnnpack must not appear in cuda routing for {kind}: {ordered}"
        )
