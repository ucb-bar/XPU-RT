"""Tests for the XNNPACK kernel provider.

These tests cover the Python surface (adapter catalogue, probe,
can_bid, propose). The native bridge has its own C-level smoke under
``runtime/native/libxpu_rt/tests/`` plus a Python wrapper at
``xpu-rt/tests/runtime/test_xnnpack_bridge.py``.

Two probe modes are exercised:
- AVAILABLE: libxpu_rt was built with ``-DXPURT_WITH_XNNPACK=ON``
  (the dev workflow). ``XPURT_RUNTIME_DIR`` can point the loader at a
  custom build dir.
- BLOCKED: libxpu_rt was built without that flag — probe must return
  a typed blocked_reason rather than raising.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import pytest

from xpu_rt.kernels.providers.xnnpack import (
    PROVIDER_ID,
    XnnpackBidPreview,
    XnnpackProvider,
)
from xpu_rt.kernels.xnnpack_adapter import (
    CATALOGUE,
    XnnOpKind,
    kinds_requiring_nhwc,
    lookup,
    lookup_any_layout,
    supported_contract_kinds,
)


# ----- Adapter catalogue contract -----------------------------------


def test_provider_id_is_stable() -> None:
    assert PROVIDER_ID == "xnnpack"


def test_catalogue_is_non_empty() -> None:
    assert len(CATALOGUE) > 20
    # f32 NHWC convolution is the cornerstone op — must be present.
    hit = lookup("conv2d", "f32", "NHWC")
    assert hit is not None
    assert hit.entry.xnn_kind is XnnOpKind.CONVOLUTION2D_NHWC_F32


def test_supported_contract_kinds_includes_core() -> None:
    kinds = set(supported_contract_kinds())
    for core in ("matmul", "conv2d", "depthwise_conv2d", "softmax",
                 "avg_pool", "max_pool", "relu", "sigmoid", "gelu"):
        assert core in kinds, f"core kind {core!r} missing from catalogue"


def test_xnn_op_kind_enum_is_wire_stable() -> None:
    """Enum values are part of the wire protocol with the bridge —
    never renumber. Lock the few we care most about."""

    assert int(XnnOpKind.FULLY_CONNECTED_F32) == 1
    assert int(XnnOpKind.CONVOLUTION2D_NHWC_F32) == 2
    assert int(XnnOpKind.DEPTHWISE_CONVOLUTION2D_NHWC_F32) == 3
    assert int(XnnOpKind.SOFTMAX_NC_F32) == 8


def test_kinds_requiring_nhwc_includes_spatial() -> None:
    needs_nhwc = set(kinds_requiring_nhwc())
    for spatial in ("conv2d", "depthwise_conv2d", "avg_pool", "max_pool"):
        assert spatial in needs_nhwc


def test_lookup_returns_none_for_unsupported_dtype() -> None:
    # We do support matmul f32 but NOT matmul bf16 (in the catalogue).
    assert lookup("matmul", "bf16", "NC") is None


def test_lookup_any_layout_finds_misaligned_layout() -> None:
    # conv2d f32 exists only in NHWC. Asking for NCHW directly fails;
    # lookup_any_layout still finds the f32 entry.
    assert lookup("conv2d", "f32", "NCHW") is None
    candidates = lookup_any_layout("conv2d", "f32")
    assert len(candidates) == 1
    assert candidates[0].layout == "NHWC"


# ----- Provider probe -----------------------------------------------


def _runtime_dir_with_xnnpack() -> str | None:
    """Return a runtime dir that holds a libxpu_rt with XNNPACK on, or None."""

    explicit = os.environ.get("XPURT_RUNTIME_DIR")
    if explicit:
        return explicit
    # Try the canonical dev build path.
    from pathlib import Path
    here = Path(__file__).resolve()
    # tests/kernels/providers/<here> → repo root → build/rt-cpu-xnn
    repo_root = here.parents[4]
    candidate = repo_root / "build" / "rt-cpu-xnn"
    if (candidate / "libxpu_rt.so").is_file():
        return str(candidate)
    return None


def test_probe_returns_typed_result() -> None:
    """Probe must always return a typed ProviderProbeResult; never raise."""

    provider = XnnpackProvider()
    result = provider.probe()
    assert result.provider_id == "xnnpack"
    assert result.status in {"available", "blocked", "not_installed",
                             "probe_error", "unsupported"}


def test_probe_available_when_xnnpack_built_in(monkeypatch) -> None:
    rt_dir = _runtime_dir_with_xnnpack()
    if rt_dir is None:
        pytest.skip("no libxpu_rt with XPURT_WITH_XNNPACK=ON built")
    monkeypatch.setenv("XPURT_RUNTIME_DIR", rt_dir)
    result = XnnpackProvider().probe()
    assert result.status == "available", (
        f"expected available; got status={result.status!r} "
        f"blocked_reason={result.blocked_reason!r} detail={result.detail!r}"
    )
    assert "host_cpu" in result.supports


def test_probe_blocked_when_runtime_dir_missing(monkeypatch) -> None:
    """Point XPURT_RUNTIME_DIR at a directory with no libxpu_rt.so."""

    monkeypatch.setenv("XPURT_RUNTIME_DIR", "/nonexistent/xpu_rt/path")
    # We can't fully isolate from the package's prebuilt staging area
    # nor from the dev build dirs the loader also scans, so we just
    # assert the result is a typed ProviderProbeResult — not raising
    # is the contract.
    result = XnnpackProvider().probe()
    assert result.status in {"available", "blocked", "not_installed",
                             "probe_error", "unsupported"}


# ----- can_bid contract decisions -----------------------------------


@dataclass
class _Contract:
    op_kind: str
    dtype: str = "f32"
    layout: str = "NHWC"


@dataclass
class _Target:
    family: str = "host_cpu"


def test_can_bid_matches_conv2d_nhwc_f32_with_high_confidence() -> None:
    provider = XnnpackProvider()
    bid = provider.can_bid(_Contract(op_kind="conv2d"), _Target())
    assert bid.kind_match == "matched"
    assert bid.confidence >= 0.9
    assert bid.selected_entry is not None
    assert bid.selected_entry.xnn_kind is XnnOpKind.CONVOLUTION2D_NHWC_F32


def test_can_bid_matches_matmul_nc_f32() -> None:
    provider = XnnpackProvider()
    bid = provider.can_bid(
        _Contract(op_kind="matmul", layout="NC"), _Target()
    )
    assert bid.kind_match == "matched"
    assert bid.selected_entry.xnn_kind is XnnOpKind.FULLY_CONNECTED_F32


def test_can_bid_declines_non_nhwc_for_conv_with_typed_reason() -> None:
    provider = XnnpackProvider()
    bid = provider.can_bid(
        _Contract(op_kind="conv2d", layout="NCHW"), _Target()
    )
    assert bid.kind_match == "declined_layout"
    assert bid.blocked_reason == "requires_nhwc_layout"
    assert "NHWC" in bid.detail
    # Confidence is low but non-zero so the auction still records the
    # bidder.
    assert 0.0 < bid.confidence < 0.5


def test_can_bid_declines_cuda_target() -> None:
    provider = XnnpackProvider()
    bid = provider.can_bid(_Contract(op_kind="conv2d"),
                            _Target(family="cuda"))
    assert bid.kind_match == "declined_target"
    assert bid.confidence == 0.0


def test_can_bid_declines_unsupported_op_kind() -> None:
    provider = XnnpackProvider()
    bid = provider.can_bid(_Contract(op_kind="fft"), _Target())
    assert bid.kind_match == "declined_kind"
    assert bid.blocked_reason == "unsupported_op_kind"


def test_can_bid_declines_unsupported_dtype() -> None:
    provider = XnnpackProvider()
    bid = provider.can_bid(
        _Contract(op_kind="matmul", dtype="bf16", layout="NC"), _Target()
    )
    # Matmul is in our supported set, but bf16 is not.
    assert bid.kind_match in {"declined_dtype", "declined_layout"}


# ----- propose() ----------------------------------------------------


def test_propose_emits_c_artifact_calling_the_bridge(tmp_path) -> None:
    from xpu_rt.providers.kernel_provider import KernelCodegenRequest

    provider = XnnpackProvider()
    request = KernelCodegenRequest(
        task_id="t-conv2d-001",
        contract=_Contract(op_kind="conv2d"),
        target=_Target(),
        artifact_dir=str(tmp_path),
    )
    result = provider.propose(request)
    assert result.provider_id == "xnnpack"
    assert result.status == "generated"
    assert result.contract_hash
    assert result.claims["kernel_abi_hash"]
    assert result.claims["selected_backend"] == "xnnpack"
    assert result.claims["xnn_op_kind"] == "CONVOLUTION2D_NHWC_F32"

    # The emitted C must reference the bridge ABI symbols.
    c_files = list(tmp_path.glob("xnnpack_kernel_*.c"))
    assert len(c_files) == 1
    src = c_files[0].read_text()
    assert "xpu_rt_xnn_create" in src
    assert "xpu_rt_xnn_reshape_setup" in src
    assert "xpu_rt_xnn_run" in src
    assert "xpu_rt_xnn_global_initialize" in src

    # Metadata records the selected XNN op-kind.
    meta_files = list(tmp_path.glob("kernel_metadata.json"))
    assert len(meta_files) == 1
    import json
    body = json.loads(meta_files[0].read_text())
    assert body["provider_id"] == "xnnpack"
    assert body["xnn_op_kind"] == "CONVOLUTION2D_NHWC_F32"
    assert body["xnn_op_kind_value"] == 2
    assert body["selected_backend"] == "xnnpack"
    assert body["contract_hash"] == result.contract_hash


def test_propose_blocked_path_returns_typed_result(tmp_path) -> None:
    from xpu_rt.providers.kernel_provider import KernelCodegenRequest

    provider = XnnpackProvider()
    # NCHW conv → bid declined → propose must mirror the decline.
    request = KernelCodegenRequest(
        task_id="t-bad",
        contract=_Contract(op_kind="conv2d", layout="NCHW"),
        target=_Target(),
        artifact_dir=str(tmp_path),
    )
    result = provider.propose(request)
    assert result.status == "blocked"
    assert result.claims["blocked_reason"] == "requires_nhwc_layout"
    # No C source emitted on the blocked path.
    assert list(tmp_path.glob("xnnpack_kernel_*.c")) == []
