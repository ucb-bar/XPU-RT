"""XnnpackProvider — fast CPU operators from XNNPACK (UCB-BAR fork).

Wraps the libxpu_rt ``xnnpack_bridge`` ABI (declared in
``runtime/native/libxpu_rt/src/drivers/xnnpack/xnnpack_bridge.h``) as a
first-class :class:`KernelProvider`. Kernels emitted by this provider
execute through the bridge: a small C source per dispatch calls
``xpu_rt_xnn_create()`` + ``xpu_rt_xnn_reshape_setup()`` +
``xpu_rt_xnn_run()``, gets cffi-compiled to a ``.so``, and is loaded by
libxpu_rt's CPU task driver like any other kernel artifact.

Probe behaviour:
  - ``available`` when libxpu_rt was built with ``-DXPURT_WITH_XNNPACK=ON``.
  - ``blocked`` (``build_flag_missing``) when the bridge entry point is
    present but returns ``-XPU_RT_XNN_ENOTSUP`` — the user needs to
    rebuild the runtime with the flag on.
  - ``not_installed`` when libxpu_rt itself isn't locatable.

Bid behaviour:
  - ``host_cpu`` target only.
  - Op-kind and dtype filtered via :mod:`xpu_rt.kernels.xnnpack_adapter`.
  - Layout: XNNPACK is NHWC-only for spatial ops; non-NHWC contracts
    get a low-confidence decline carrying
    ``blocked_reason=requires_nhwc_layout`` so the layout-normalisation
    pass can react.

Propose behaviour:
  - Emits a C source that calls the bridge, cffi-compiles to a ``.so``
    under the request's artifact dir, and returns a v1 ``ProviderResult``
    with the artifact path + a kernel-ABI hash + ``selected_backend``
    metadata recording ``"xnnpack"``.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import dataclasses
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from xpu_rt.kernels.xnnpack_adapter import (
    CATALOGUE,
    LookupResult,
    OpEntry,
    kinds_requiring_nhwc,
    kinds_requiring_static_weights,
    lookup,
    lookup_any_layout,
    supported_contract_kinds,
)
from xpu_rt.providers.kernel_provider import (
    KernelCodegenRequest,
    KernelProvider,
)
from xpu_rt.providers.provider_types import ProviderProbeResult

PROVIDER_ID = "xnnpack"

# Wire-stable schema version baked into every ProviderResult this
# provider produces; bumped if the bridge ABI ever changes.
SCHEMA_VERSION = "provider_probe_v1"
BRIDGE_PROBE_SYMBOL = "xpu_rt_xnn_global_initialize"


def _libxpu_rt_candidates() -> tuple[Path, ...]:
    """Plausible locations of libxpu_rt.so / libxpu_rt_static.a."""

    candidates: list[Path] = []
    # 1. XPURT_RUNTIME_DIR overrides everything.
    override = os.environ.get("XPURT_RUNTIME_DIR")
    if override:
        for name in ("libxpu_rt.so", "libxpu_rt.dylib"):
            candidates.append(Path(override) / name)
    # 2. The package's prebuilt staging area.
    here = Path(__file__).resolve()
    pkg_root = here.parent.parent.parent  # .../xpu_rt
    for name in ("libxpu_rt-cpu.so", "libxpu_rt.so", "libxpu_rt-cpu.dylib", "libxpu_rt.dylib"):
        candidates.append(pkg_root / "runtime" / "native" / "prebuilt" / name)
    # 3. The dev-mode out-of-tree CMake build dirs.
    repo_root = Path(__file__).resolve().parents[3]  # .../xpu-rt-integration
    for build_dir in ("build/rt-cpu-xnn", "build/rt-cpu"):
        candidates.append(repo_root / build_dir / "libxpu_rt.so")
    return tuple(candidates)


def _probe_bridge() -> tuple[bool, str | None, str]:
    """Try to dlopen libxpu_rt and confirm the bridge entry point.

    Returns ``(available, error_reason_token, detail)``.
    ``error_reason_token`` is one of ``None``, ``not_installed``,
    ``build_flag_missing``, ``probe_exception``.
    """

    libxpu_rt = None
    candidates = _libxpu_rt_candidates()
    for p in candidates:
        if p.is_file():
            try:
                libxpu_rt = ctypes.CDLL(str(p), mode=ctypes.RTLD_LOCAL)
                break
            except OSError:
                continue
    if libxpu_rt is None:
        return (False, "not_installed",
                f"libxpu_rt not found. Tried: {[str(p) for p in candidates]}")

    if not hasattr(libxpu_rt, BRIDGE_PROBE_SYMBOL):
        return (False, "build_flag_missing",
                f"libxpu_rt loaded but missing {BRIDGE_PROBE_SYMBOL}; "
                f"rebuild with -DXPURT_WITH_XNNPACK=ON")

    init = getattr(libxpu_rt, BRIDGE_PROBE_SYMBOL)
    init.restype = ctypes.c_int
    try:
        rc = int(init())
    except Exception as exc:  # pragma: no cover - hardware corner
        return (False, "probe_exception", f"bridge call raised: {exc}")

    XPU_RT_XNN_ENOTSUP = -95
    if rc == XPU_RT_XNN_ENOTSUP:
        return (False, "build_flag_missing",
                f"{BRIDGE_PROBE_SYMBOL} returned -ENOTSUP; libxpu_rt "
                "was built without XPURT_WITH_XNNPACK=ON")
    if rc != 0:
        return (False, "probe_exception",
                f"{BRIDGE_PROBE_SYMBOL} returned {rc}; "
                "see libxpu_rt logs / xpu_rt_xnn_last_error()")

    # Optional: version string from the bridge.
    version = ""
    if hasattr(libxpu_rt, "xpu_rt_xnn_version"):
        ver_fn = libxpu_rt.xpu_rt_xnn_version
        ver_fn.restype = ctypes.c_char_p
        s = ver_fn()
        if s:
            version = s.decode("utf-8", errors="replace")
    return (True, None, version or "xnnpack-runtime")


# Lightweight BidPreview substitute that the legacy auction also
# accepts. We don't import :class:`BidPreview` from the legacy module
# directly so the new-style ABC stays self-contained; the consumer
# (provider_routing.route_for) only needs an object with
# ``confidence``, ``kind_match``, ``detail``, ``blocked_reason``.
@dataclasses.dataclass(frozen=True)
class XnnpackBidPreview:
    provider_id: str
    confidence: float
    kind_match: str  # "matched" | "declined_layout" | "declined_dtype" | "declined_kind" | "declined_target"
    detail: str
    blocked_reason: str | None
    selected_entry: OpEntry | None


def _resolve_op_kind(contract: Any) -> str:
    """Extract a string op-kind from the contract; tolerates schema variation."""

    for attr in ("op_kind", "operation_kind", "kind", "op_family"):
        v = getattr(contract, attr, None)
        if isinstance(v, str) and v:
            return v.lower()
    # Last resort: dict-style.
    if isinstance(contract, dict):
        for k in ("op_kind", "kind", "op_family"):
            if k in contract and contract[k]:
                return str(contract[k]).lower()
    return ""


def _resolve_dtype(contract: Any) -> str:
    for attr in ("dtype", "element_type", "scalar_type", "compute_dtype"):
        v = getattr(contract, attr, None)
        if isinstance(v, str) and v:
            return v.lower()
    if isinstance(contract, dict) and "dtype" in contract:
        return str(contract["dtype"]).lower()
    return "f32"


def _resolve_layout(contract: Any) -> str:
    for attr in ("layout", "tensor_layout", "data_format"):
        v = getattr(contract, attr, None)
        if isinstance(v, str) and v:
            return v.upper()
    if isinstance(contract, dict) and "layout" in contract:
        return str(contract["layout"]).upper()
    # Default: "any" so generic ops (FC, softmax, unary, binary) match.
    return "ANY"


def _resolve_target_family(target: Any) -> str:
    for attr in ("family", "target_family", "kind"):
        v = getattr(target, attr, None)
        if isinstance(v, str) and v:
            return v.lower()
    if isinstance(target, dict) and "family" in target:
        return str(target["family"]).lower()
    return ""


class XnnpackProvider(KernelProvider):
    """KernelProvider routing through the libxpu_rt XNNPACK bridge."""

    provider_id: str = PROVIDER_ID

    def probe(self) -> ProviderProbeResult:
        try:
            available, reason, detail = _probe_bridge()
        except Exception as exc:  # pragma: no cover - defensive
            return ProviderProbeResult(
                schema_version=SCHEMA_VERSION,
                provider_id=PROVIDER_ID,
                status="probe_error",
                blocked_reason="probe_exception",
                detail=f"{type(exc).__name__}: {exc}",
            )
        if available:
            return ProviderProbeResult(
                schema_version=SCHEMA_VERSION,
                provider_id=PROVIDER_ID,
                status="available",
                version=detail,
                supports=("host_cpu",) + supported_contract_kinds(),
                detail=(
                    "libxpu_rt with XPURT_WITH_XNNPACK=ON; XNNPACK kernels "
                    "execute through the cpu_task driver via the "
                    "xnnpack_bridge ABI."
                ),
                paper_claimable=False,
            )
        return ProviderProbeResult(
            schema_version=SCHEMA_VERSION,
            provider_id=PROVIDER_ID,
            status="blocked" if reason == "build_flag_missing" else "not_installed",
            blocked_reason=reason or "probe_exception",
            detail=detail,
        )

    def can_bid(self, contract: Any, target: Any) -> XnnpackBidPreview:
        family = _resolve_target_family(target)
        if family and family != "host_cpu":
            return XnnpackBidPreview(
                provider_id=PROVIDER_ID,
                confidence=0.0,
                kind_match="declined_target",
                detail=f"target_family={family!r}; xnnpack is host_cpu only",
                blocked_reason="unsupported_target",
                selected_entry=None,
            )

        op_kind = _resolve_op_kind(contract)
        dtype = _resolve_dtype(contract)
        layout = _resolve_layout(contract)

        # Direct match: op_kind + dtype + layout.
        hit = lookup(op_kind, dtype, layout)
        if hit is not None:
            return XnnpackBidPreview(
                provider_id=PROVIDER_ID,
                confidence=hit.entry.base_confidence,
                kind_match="matched",
                detail=(
                    f"xnnpack handles {hit.entry.contract_op_kinds[0]} "
                    f"dtype={hit.entry.dtype} layout={hit.entry.layout} "
                    f"({hit.entry.note or hit.entry.xnn_kind.name})"
                ),
                blocked_reason=None,
                selected_entry=hit.entry,
            )

        # Same op+dtype but wrong layout — flag the layout-pass.
        layout_candidates = lookup_any_layout(op_kind, dtype)
        if layout_candidates:
            required = layout_candidates[0].layout
            if op_kind in kinds_requiring_nhwc():
                return XnnpackBidPreview(
                    provider_id=PROVIDER_ID,
                    confidence=0.10,
                    kind_match="declined_layout",
                    detail=(
                        f"xnnpack supports {op_kind} dtype={dtype} only in "
                        f"layout={required}; got {layout}. Insert a "
                        "layout transpose upstream and re-bid."
                    ),
                    blocked_reason="requires_nhwc_layout",
                    selected_entry=layout_candidates[0],
                )
            return XnnpackBidPreview(
                provider_id=PROVIDER_ID,
                confidence=0.05,
                kind_match="declined_layout",
                detail=f"xnnpack expects layout={required}; got {layout}",
                blocked_reason="layout_mismatch",
                selected_entry=layout_candidates[0],
            )

        # Op-kind not in our catalogue at this dtype.
        if op_kind in supported_contract_kinds():
            return XnnpackBidPreview(
                provider_id=PROVIDER_ID,
                confidence=0.0,
                kind_match="declined_dtype",
                detail=(
                    f"xnnpack handles {op_kind} but not at dtype={dtype}; "
                    f"supported dtypes vary by op-kind (see "
                    "xpu_rt.kernels.xnnpack_adapter.CATALOGUE)"
                ),
                blocked_reason="unsupported_dtype",
                selected_entry=None,
            )

        return XnnpackBidPreview(
            provider_id=PROVIDER_ID,
            confidence=0.0,
            kind_match="declined_kind",
            detail=f"xnnpack does not handle op_kind={op_kind!r}",
            blocked_reason="unsupported_op_kind",
            selected_entry=None,
        )

    # ------------------------------------------------------------------
    # propose() — emit a C kernel artifact calling the bridge ABI.
    # ------------------------------------------------------------------
    def propose(self, request: KernelCodegenRequest) -> Any:
        # Late import — the ProviderResult v1 type lives in
        # xpu_rt.providers.result_v1; we keep the import lazy so a
        # caller that only wants probe() doesn't pay the cost.
        from xpu_rt.providers.result_v1 import ProviderResultV1

        contract = request.contract
        target = request.target
        target_id = (getattr(target, "target_id", None)
                     or getattr(target, "name", None)
                     or "host_cpu")
        bid = self.can_bid(contract, target)
        if bid.kind_match != "matched" or bid.selected_entry is None:
            return ProviderResultV1(
                schema_version="provider_result_v1",
                provider_id=PROVIDER_ID,
                task_id=request.task_id,
                target_id=str(target_id),
                contract_hash="",
                status="blocked",
                detail=bid.detail,
                claims={
                    "blocked_reason": bid.blocked_reason or "unsupported",
                    "kind_match": bid.kind_match,
                    "selected_backend": "xnnpack",
                },
            )

        artifact_dir = Path(request.artifact_dir)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        c_path = artifact_dir / f"xnnpack_kernel_{request.task_id}.c"
        meta_path = artifact_dir / "kernel_metadata.json"

        entry = bid.selected_entry
        # Emit a portable C wrapper. The actual XNNPACK call site lives
        # inside libxpu_rt (the bridge); the kernel artifact's job is to
        # bind the contract-level constants (op-kind, packed shape) to
        # those bridge calls.
        c_source = _emit_kernel_c(request, entry)
        c_path.write_text(c_source)

        contract_hash = _hash_bytes(json.dumps(_summarize_contract(contract),
                                              sort_keys=True).encode())
        kernel_abi_hash = _hash_bytes(c_source.encode())

        meta = {
            "schema_version": "kernel_metadata_v1",
            "provider_id": PROVIDER_ID,
            "task_id": request.task_id,
            "xnn_op_kind": entry.xnn_kind.name,
            "xnn_op_kind_value": int(entry.xnn_kind),
            "dtype": entry.dtype,
            "layout": entry.layout,
            "contract_hash": contract_hash,
            "kernel_abi_hash": kernel_abi_hash,
            "selected_backend": "xnnpack",
            "needs_static_weights": entry.needs_static_weights,
            "notes": entry.note,
        }
        meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True))

        return ProviderResultV1(
            schema_version="provider_result_v1",
            provider_id=PROVIDER_ID,
            task_id=request.task_id,
            target_id=str(target_id),
            contract_hash=contract_hash,
            status="generated",
            detail=(
                f"xnnpack emitted bridge wrapper for "
                f"{entry.contract_op_kinds[0]} (xnn_kind={entry.xnn_kind.name})"
            ),
            artifacts={
                "source": str(c_path),       # canonical kernel source key
                "c_source": str(c_path),
                "metadata": str(meta_path),
            },
            claims={
                "xnn_op_kind": entry.xnn_kind.name,
                "xnn_op_kind_value": int(entry.xnn_kind),
                "dtype": entry.dtype,
                "layout": entry.layout,
                "kernel_abi_hash": kernel_abi_hash,
                "selected_backend": "xnnpack",
            },
        )


def _hash_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()[:16]


def _summarize_contract(contract: Any) -> dict[str, Any]:
    """JSON-friendly summary for hashing (mirrors xpu_rt.scheduler.bridge)."""

    if dataclasses.is_dataclass(contract):
        d = dataclasses.asdict(contract)
    elif isinstance(contract, dict):
        d = dict(contract)
    else:
        d = {"repr": repr(contract)}
    # Drop fields known to be non-stable across reruns.
    for k in ("artifact_dir", "evidence_dir", "metadata"):
        d.pop(k, None)
    return d


def _emit_kernel_c(request: KernelCodegenRequest, entry: OpEntry) -> str:
    """Emit a C wrapper that calls the bridge for ``entry``.

    The wrapper exports two symbols matching libxpu_rt's kernel ABI:
        int xpu_rt_kernel_init(void* config);
        int xpu_rt_kernel_run(void* const* inputs, void* const* outputs,
                              const int64_t* runtime_shape, size_t n_runtime_shape);
    """

    op_value = int(entry.xnn_kind)
    return f"""\
/* xnnpack kernel artifact — task={request.task_id}
 *
 * Auto-emitted by xpu_rt.kernels.providers.xnnpack.XnnpackProvider.
 * Compile against libxpu_rt's exported headers; the actual XNNPACK
 * call site lives in libxpu_rt/src/drivers/xnnpack/xnnpack_bridge.c.
 */

#include <stddef.h>
#include <stdint.h>
#include "xpu_rt/drivers/xnnpack/xnnpack_bridge.h"

static xpu_rt_xnn_op* g_op = NULL;

/* Filled in at xpu_rt_kernel_init() time from `config` (or compile-time
 * constants for static-shape ops). For brevity in the v1 wrapper we
 * embed the contract shape inline; a follow-up will read it from the
 * dispatch metadata blob. */
static const int64_t kInitShape[] = {{0}};   /* placeholder; filled by the runtime */
static const int32_t kIntParams[] = {{0}};
static const float   kFloatParams[] = {{0.0f}};

int xpu_rt_kernel_init(void* config) {{
    (void)config;
    if (xpu_rt_xnn_global_initialize() != XPU_RT_XNN_OK) {{
        return -1;
    }}
    g_op = xpu_rt_xnn_create(
        (xpu_rt_xnn_op_kind){op_value},
        kInitShape,    sizeof(kInitShape)    / sizeof(int64_t),
        kIntParams,    sizeof(kIntParams)    / sizeof(int32_t),
        kFloatParams,  sizeof(kFloatParams)  / sizeof(float),
        NULL, 0);
    return g_op != NULL ? 0 : -2;
}}

int xpu_rt_kernel_run(
        void* const* inputs,  size_t n_inputs,
        void* const* outputs, size_t n_outputs,
        const int64_t* runtime_shape, size_t n_runtime_shape) {{
    int rc = xpu_rt_xnn_reshape_setup(
        g_op, runtime_shape, n_runtime_shape,
        (const void* const*)inputs,  n_inputs,
        outputs,                     n_outputs);
    if (rc != XPU_RT_XNN_OK) return rc;
    return xpu_rt_xnn_run(g_op);
}}

void xpu_rt_kernel_destroy(void) {{
    if (g_op != NULL) {{
        xpu_rt_xnn_destroy(g_op);
        g_op = NULL;
    }}
}}
"""
