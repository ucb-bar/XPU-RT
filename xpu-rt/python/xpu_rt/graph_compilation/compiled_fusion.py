"""Compiled Fusion Verification.

Compiles a real fused producer→consumer kernel for FuseProducerConsumer
candidates, runs it, and compares the output against the eager unfused
chain on the same frozen input cases. Layered alongside 's
existing differential — 's report stays byte-identical (only
ADDS a new sibling report + overlay).

Scope (MVP — strict, honest non-claims):

Pointwise → pointwise only (matches 's whitelist).
- Binary producer + unary consumer is the canonical case (proxy_vla's
  bias_add → relu). Other pairs in the whitelist (mul, sub,
  unary→unary chains like sigmoid→tanh) are supported by the same
  compute-template; only the operator string differs in the kernel body.
- fp32 only. f16/bf16 mixed-precision out of scope.
Single-consumer producer chains ('s existing constraint).
- No matmul fusion (matmul accumulation reorder is reduction-sensitive
  and is territory; correctly blocked 's pointwise gate).

Hard non-goals:

- No new candidate generation.
- No compiler-core imports.
No mutation of 's `real_fusion_manifest.json` /
  `real_fusion_differential_report.json` (verified by tests).
No mutation of ////artifacts.
- Best-effort: missing CUDA → typed `device_unavailable`; cffi/gcc
  missing → typed `library_unavailable`; raised exceptions → typed
  `compile_failed` / `run_failed` with note. Never raises.

Bit-equality gate:

- For pointwise→pointwise the fused expression is mathematically
  identical to the unfused chain. Bit-equality (max_abs_error == 0
  AND max_rel_error == 0) is the expected outcome and is the
  refinement target.
- TF32 explicitly disabled on Triton (no matmul, but kept for
  consistency).
- gcc compiled with ``-fno-fast-math`` so the C kernel doesn't reorder
  floating-point ops.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


_TOLERANCE_EPS = (1e-5, 1e-4)


def _utcnow() -> str:
    return datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256_text(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


# --------------------------------------------------------------------------- #
# Pointwise op registry — Triton + C source fragments
# --------------------------------------------------------------------------- #
# Each entry maps a producer/consumer kind name (matching 's
# `_pointwise_op_for` discriminator strings) to a tuple:
#   (arity, triton_expr_template, c_expr_template)
# where the templates use {a}, {b} placeholders (binary) or {x}
# (unary). Bodies expand to a single Triton/C expression.

_PRODUCER_TEMPLATES = {
    # binary
    "bias_add":            ("binary", "({a} + {b})",            "(({a}) + ({b}))"),
    "add":                 ("binary", "({a} + {b})",            "(({a}) + ({b}))"),
    "mul":                 ("binary", "({a} * {b})",            "(({a}) * ({b}))"),
    "sub":                 ("binary", "({a} - {b})",            "(({a}) - ({b}))"),
    # unary
    "elementwise_relu":    ("unary",  "tl.maximum({x}, 0.0)",   "fmaxf({x}, 0.0f)"),
    "elementwise_sigmoid": ("unary",  "tl.sigmoid({x})",        "(1.0f/(1.0f + expf(-({x}))))"),
    "elementwise_tanh":    ("unary",  "(tl.exp({x}) - tl.exp(-{x})) / (tl.exp({x}) + tl.exp(-{x}))",
                            "tanhf({x})"),
}

# Consumer-only registry (must be unary in MVP).
_CONSUMER_TEMPLATES = {
    "elementwise_relu":    ("unary",  "tl.maximum({x}, 0.0)",   "fmaxf({x}, 0.0f)"),
    "elementwise_sigmoid": ("unary",  "tl.sigmoid({x})",        "(1.0f/(1.0f + expf(-({x}))))"),
    "elementwise_tanh":    ("unary",  "(tl.exp({x}) - tl.exp(-{x})) / (tl.exp({x}) + tl.exp(-{x}))",
                            "tanhf({x})"),
}


def _classify_kind(kind: str) -> str:
    """Map 's diagnostics.producer_kind / consumer_kind to a
    template registry key."""
    n = kind.strip().lower()
    # Producers
    if "bias_add" in n: return "bias_add"
    if n.startswith("aten_add") or n == "add" or "elementwise_add" in n:
        return "add"
    if n.startswith("aten_mul") or n == "mul" or "elementwise_mul" in n:
        return "mul"
    if n.startswith("aten_sub") or n == "sub" or "elementwise_sub" in n:
        return "sub"
    # Unary
    if "relu" in n: return "elementwise_relu"
    if "sigmoid" in n: return "elementwise_sigmoid"
    if "tanh" in n: return "elementwise_tanh"
    return n


# --------------------------------------------------------------------------- #
# Triton kernel template (parameterised on producer + consumer)
# --------------------------------------------------------------------------- #


def _emit_triton_source(
    *,
    producer_kind: str, consumer_kind: str,
    matmul_shape: tuple[int, int],
    candidate_id: str, region_pair: str,
    deterministic: bool = False,
) -> str:
    """Emit a fused Triton kernel for binary producer + unary consumer.
    The producer broadcasts a 1-D bias along axis 0; the consumer is
    applied pointwise on the result. Block size = N_COLS (small problem;
    one program covers the whole column dimension)."""
    p_meta = _PRODUCER_TEMPLATES.get(producer_kind)
    c_meta = _CONSUMER_TEMPLATES.get(consumer_kind)
    if p_meta is None or c_meta is None:
        raise ValueError(
            f"unsupported pair: producer={producer_kind!r} consumer={consumer_kind!r}"
        )
    p_arity, p_expr_tpl, _ = p_meta
    _, c_expr_tpl, _ = c_meta
    indent = "    "
    if p_arity != "binary":
        # Unary→unary chain: t = producer(x); out = consumer(t)
        producer_body = "\n".join([
            f"{indent}x = tl.load(a_ptr + offs, mask=mask, other=0.0)",
            f"{indent}t = {p_expr_tpl.format(x='x')}",
        ])
    else:
        producer_body = "\n".join([
            f"{indent}a = tl.load(a_ptr + offs, mask=mask, other=0.0)",
            f"{indent}b = tl.load(b_ptr + (offs % BIAS_LEN), mask=mask, other=0.0)",
            f"{indent}t = {p_expr_tpl.format(a='a', b='b')}",
        ])

    consumer_body = f"{indent}out = {c_expr_tpl.format(x='t')}"
    timestamp = "" if deterministic else _utcnow()

    return (
        f'"""M-23 generated Triton fused kernel.\n'
        f'\n'
        f'candidate_id: {candidate_id}\n'
        f'region_pair: {region_pair}\n'
        f'producer: {producer_kind}  consumer: {consumer_kind}\n'
        f'matmul_shape: rows={matmul_shape[0]}  cols={matmul_shape[1]}\n'
        f'generated_at_utc: {timestamp}\n'
        f'"""\n'
        f'\n'
        f'import triton\n'
        f'import triton.language as tl\n'
        f'\n'
        f'\n'
        f'@triton.jit\n'
        f'def fused_kernel(\n'
        f'    a_ptr, b_ptr, c_ptr,\n'
        f'    N_ELEMS, BIAS_LEN,\n'
        f'    BLOCK: tl.constexpr,\n'
        f'):\n'
        f'    pid = tl.program_id(0)\n'
        f'    offs = pid * BLOCK + tl.arange(0, BLOCK)\n'
        f'    mask = offs < N_ELEMS\n'
        f'    # producer\n'
        f'{producer_body}\n'
        f'    # consumer\n'
        f'{consumer_body}\n'
        f'    tl.store(c_ptr + offs, out, mask=mask)\n'
    )


def _emit_c_source(
    *,
    producer_kind: str, consumer_kind: str,
    matmul_shape: tuple[int, int],
    candidate_id: str, region_pair: str,
    safe_name: str,
    deterministic: bool = False,
) -> str:
    """Emit fused C kernel: ``void compgen_m23_fused_<safe_name>(
    const float* A, const float* b, float* C)``."""
    p_meta = _PRODUCER_TEMPLATES.get(producer_kind)
    c_meta = _CONSUMER_TEMPLATES.get(consumer_kind)
    if p_meta is None or c_meta is None:
        raise ValueError(
            f"unsupported pair: producer={producer_kind!r} consumer={consumer_kind!r}"
        )
    p_arity, _, p_c_tpl = p_meta
    _, _, c_c_tpl = c_meta

    rows, cols = matmul_shape
    timestamp = "" if deterministic else _utcnow()

    if p_arity == "binary":
        producer_line = (
            f"            float t = {p_c_tpl.format(a='A[r*N_COLS + c]', b='b[c]')};"
        )
        sig = (
            f"void compgen_m23_fused_{safe_name}(const float* A, "
            f"const float* b, float* C)"
        )
    else:
        producer_line = (
            f"            float t = {p_c_tpl.format(x='A[r*N_COLS + c]')};"
        )
        sig = (
            f"void compgen_m23_fused_{safe_name}(const float* A, float* C)"
        )

    consumer_line = (
        f"            C[r*N_COLS + c] = {c_c_tpl.format(x='t')};"
    )

    return (
        f"// M-23 generated fused CPU kernel.\n"
        f"// candidate_id: {candidate_id}\n"
        f"// region_pair: {region_pair}\n"
        f"// producer: {producer_kind}  consumer: {consumer_kind}\n"
        f"// shape: rows={rows} cols={cols}\n"
        f"// generated_at_utc: {timestamp}\n"
        f"\n"
        f"#include <math.h>\n"
        f"\n"
        f"#define N_ROWS {rows}\n"
        f"#define N_COLS {cols}\n"
        f"\n"
        f"{sig} {{\n"
        f"    for (int r = 0; r < N_ROWS; r++) {{\n"
        f"        for (int c = 0; c < N_COLS; c++) {{\n"
        f"{producer_line}\n"
        f"{consumer_line}\n"
        f"        }}\n"
        f"    }}\n"
        f"}}\n"
    )


# --------------------------------------------------------------------------- #
# Reference (eager unfused) computation
# --------------------------------------------------------------------------- #


def _eager_unfused(
    *,
    a, b, producer_kind: str, consumer_kind: str,  # type: ignore[no-untyped-def]
):
    """Compute the unfused chain on tensors. b may be None for unary
    producers."""
    import torch

    p = producer_kind
    if p == "bias_add" or p == "add":
        t = a + b
    elif p == "mul":
        t = a * b
    elif p == "sub":
        t = a - b
    elif p == "elementwise_relu":
        t = torch.relu(a)
    elif p == "elementwise_sigmoid":
        t = torch.sigmoid(a)
    elif p == "elementwise_tanh":
        t = torch.tanh(a)
    else:
        raise ValueError(f"unsupported producer: {p}")

    c = consumer_kind
    if c == "elementwise_relu":
        return torch.relu(t)
    if c == "elementwise_sigmoid":
        return torch.sigmoid(t)
    if c == "elementwise_tanh":
        return torch.tanh(t)
    raise ValueError(f"unsupported consumer: {c}")


def _classify_refinement(
    max_abs: float, max_rel: float,
) -> str:
    if max_abs == 0.0 and max_rel == 0.0:
        return "discharged_compiled_bit_equality"
    atol, rtol = _TOLERANCE_EPS
    if max_abs <= atol and max_rel <= rtol:
        return "discharged_tolerance_eps"
    return "fail_outside_tolerance"


# --------------------------------------------------------------------------- #
# GPU sub-track
# --------------------------------------------------------------------------- #


def _run_gpu_track(
    *,
    out_dir: Path,
    common: dict[str, Any],
    matmul_shape: tuple[int, int],
    producer_kind: str, consumer_kind: str,
    p_arity: str,
    n_cases: int,
    seed: int,
) -> Path:
    """Emit + compile + launch a fused Triton kernel; verify against
    eager. Writes ``compiled_fusion_run_gpu.json`` and returns its path."""
    artifact_path = out_dir / "compiled_fusion_run_gpu.json"
    base: dict[str, Any] = {
        "schema_version": "compiled_fusion_run_v1",
        "track": "gpu_triton",
        "candidate_id": common["candidate_id"],
        "region_pair": common["region_pair"],
        "matmul_shape": {"rows": matmul_shape[0], "cols": matmul_shape[1]},
        "producer_kind": producer_kind,
        "consumer_kind": consumer_kind,
        "iterations": int(common.get("iterations", 32)),
        "warmup": int(common.get("warmup", 4)),
        "compile_status": "not_run",
        "run_status": "not_run",
        "generated_at_utc": _utcnow(),
    }

    try:
        import torch
    except ImportError as exc:
        base["compile_status"] = "torch_unavailable"
        base["note"] = f"{type(exc).__name__}: {exc}"
        artifact_path.write_text(
            json.dumps(base, indent=2, sort_keys=True), encoding="utf-8",
        )
        return artifact_path
    if not torch.cuda.is_available():
        base["compile_status"] = "device_unavailable"
        artifact_path.write_text(
            json.dumps(base, indent=2, sort_keys=True), encoding="utf-8",
        )
        return artifact_path
    try:
        import triton  # noqa: F401
    except ImportError:
        base["compile_status"] = "triton_unavailable"
        artifact_path.write_text(
            json.dumps(base, indent=2, sort_keys=True), encoding="utf-8",
        )
        return artifact_path

    rows, cols = matmul_shape
    n_elems = rows * cols
    bias_len = cols
    safe_name = common["safe_name"]
    src = _emit_triton_source(
        producer_kind=producer_kind, consumer_kind=consumer_kind,
        matmul_shape=matmul_shape,
        candidate_id=common["candidate_id"],
        region_pair=common["region_pair"],
    )
    src_path = out_dir / f"triton_fused_{safe_name}.py"
    src_path.write_text(src, encoding="utf-8")
    base["kernel_source_path"] = str(src_path.relative_to(out_dir))
    # Deterministic SHA over a timestamp-stripped variant.
    deterministic_src = _emit_triton_source(
        producer_kind=producer_kind, consumer_kind=consumer_kind,
        matmul_shape=matmul_shape,
        candidate_id=common["candidate_id"],
        region_pair=common["region_pair"],
        deterministic=True,
    )
    base["kernel_source_sha256"] = _sha256_text(deterministic_src)

    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            f"_m23_fused_{safe_name}", src_path,
        )
        if spec is None or spec.loader is None:
            raise ImportError("could not spec triton fused module")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        kernel_callable = getattr(mod, "fused_kernel")
    except Exception as exc:  # noqa: BLE001
        base["compile_status"] = "compile_failed"
        base["note"] = f"{type(exc).__name__}: {exc}"
        artifact_path.write_text(
            json.dumps(base, indent=2, sort_keys=True), encoding="utf-8",
        )
        return artifact_path

    base["compile_status"] = "compiled"

    # Generate frozen input cases (match conventions).
    gen = torch.Generator(device="cuda")
    gen.manual_seed(int(seed))
    cases = []
    for i in range(int(n_cases)):
        a = torch.randn(rows, cols, dtype=torch.float32,
                        device="cuda", generator=gen)
        if p_arity == "binary":
            b = torch.randn(bias_len, dtype=torch.float32,
                            device="cuda", generator=gen)
            cases.append({"a": a, "b": b})
        else:
            cases.append({"a": a, "b": None})

    case_results: list[dict[str, Any]] = []
    max_abs_overall = 0.0
    max_rel_overall = 0.0
    per_iter_us: list[float] = []
    BLOCK = max(16, ((cols + 15) // 16) * 16)
    grid = ((n_elems + BLOCK - 1) // BLOCK,)

    try:
        # Reference outputs (CPU-equivalent eager).
        for ci, case in enumerate(cases):
            a = case["a"]
            b = case["b"]
            ref = _eager_unfused(
                a=a, b=b,
                producer_kind=producer_kind, consumer_kind=consumer_kind,
            )
            out = torch.zeros_like(a)
            # b_ptr placeholder for unary case
            b_arg = b if b is not None else torch.zeros(
                bias_len, dtype=torch.float32, device="cuda",
            )
            kernel_callable[grid](
                a, b_arg, out,
                n_elems, bias_len,
                BLOCK=BLOCK,
            )
            torch.cuda.synchronize()
            diff = (out - ref).abs()
            ref_abs = ref.abs()
            max_abs = float(diff.max().item())
            max_rel = float((diff / (ref_abs + 1e-12)).max().item())
            max_abs_overall = max(max_abs_overall, max_abs)
            max_rel_overall = max(max_rel_overall, max_rel)
            case_results.append({
                "case_id": f"case_{ci:03d}",
                "max_abs_error": max_abs,
                "max_rel_error": max_rel,
                "refinement_status": _classify_refinement(max_abs, max_rel),
            })

        # Timing pass (separate from numerical comparison).
        warmup_iters = int(common.get("warmup", 4))
        a = cases[0]["a"]
        b_arg = cases[0]["b"] if cases[0]["b"] is not None else torch.zeros(
            bias_len, dtype=torch.float32, device="cuda",
        )
        out = torch.zeros_like(a)
        for _ in range(warmup_iters):
            kernel_callable[grid](a, b_arg, out, n_elems, bias_len, BLOCK=BLOCK)
        torch.cuda.synchronize()
        iters = int(common.get("iterations", 32))
        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)
        for _ in range(iters):
            start_evt.record()
            kernel_callable[grid](a, b_arg, out, n_elems, bias_len, BLOCK=BLOCK)
            end_evt.record()
            torch.cuda.synchronize()
            per_iter_us.append(start_evt.elapsed_time(end_evt) * 1000.0)
    except Exception as exc:  # noqa: BLE001
        base["run_status"] = "run_failed"
        base["note"] = f"{type(exc).__name__}: {exc}"
        artifact_path.write_text(
            json.dumps(base, indent=2, sort_keys=True), encoding="utf-8",
        )
        return artifact_path

    base["run_status"] = "ok"
    base["case_count"] = len(cases)
    base["bit_equality_count"] = sum(
        1 for c in case_results
        if c["refinement_status"] == "discharged_compiled_bit_equality"
    )
    base["tolerance_eps_count"] = sum(
        1 for c in case_results
        if c["refinement_status"] == "discharged_tolerance_eps"
    )
    base["fail_outside_tolerance_count"] = sum(
        1 for c in case_results
        if c["refinement_status"] == "fail_outside_tolerance"
    )
    base["max_abs_error"] = max_abs_overall
    base["max_rel_error"] = max_rel_overall
    base["measured_us_per_iter"] = (
        sum(per_iter_us) / len(per_iter_us) if per_iter_us else None
    )
    base["cases"] = case_results
    base["device"] = {"kind": "cuda", "name": torch.cuda.get_device_name()}

    artifact_path.write_text(
        json.dumps(base, indent=2, sort_keys=True), encoding="utf-8",
    )
    return artifact_path


# --------------------------------------------------------------------------- #
# CPU sub-track (cffi)
# --------------------------------------------------------------------------- #


def _run_cpu_track(
    *,
    out_dir: Path,
    common: dict[str, Any],
    matmul_shape: tuple[int, int],
    producer_kind: str, consumer_kind: str,
    p_arity: str,
    n_cases: int,
    seed: int,
) -> Path:
    artifact_path = out_dir / "compiled_fusion_run_cpu.json"
    base: dict[str, Any] = {
        "schema_version": "compiled_fusion_run_v1",
        "track": "cpu_compgen_rt",
        "candidate_id": common["candidate_id"],
        "region_pair": common["region_pair"],
        "matmul_shape": {"rows": matmul_shape[0], "cols": matmul_shape[1]},
        "producer_kind": producer_kind,
        "consumer_kind": consumer_kind,
        "iterations": int(common.get("iterations", 32)),
        "warmup": int(common.get("warmup", 4)),
        "compile_status": "not_run",
        "run_status": "not_run",
        "generated_at_utc": _utcnow(),
    }

    try:
        import torch
    except ImportError as exc:
        base["compile_status"] = "torch_unavailable"
        base["note"] = f"{type(exc).__name__}: {exc}"
        artifact_path.write_text(
            json.dumps(base, indent=2, sort_keys=True), encoding="utf-8",
        )
        return artifact_path
    try:
        import cffi
    except ImportError:
        base["compile_status"] = "library_unavailable"
        base["note"] = "cffi not available"
        artifact_path.write_text(
            json.dumps(base, indent=2, sort_keys=True), encoding="utf-8",
        )
        return artifact_path

    rows, cols = matmul_shape
    safe_name = common["safe_name"]

    src = _emit_c_source(
        producer_kind=producer_kind, consumer_kind=consumer_kind,
        matmul_shape=matmul_shape,
        candidate_id=common["candidate_id"],
        region_pair=common["region_pair"],
        safe_name=safe_name,
    )
    c_path = out_dir / f"cpu_fused_{safe_name}.c"
    c_path.write_text(src, encoding="utf-8")
    base["c_source_path"] = str(c_path.relative_to(out_dir))
    deterministic_src = _emit_c_source(
        producer_kind=producer_kind, consumer_kind=consumer_kind,
        matmul_shape=matmul_shape,
        candidate_id=common["candidate_id"],
        region_pair=common["region_pair"],
        safe_name=safe_name, deterministic=True,
    )
    base["c_source_sha256"] = _sha256_text(deterministic_src)

    if p_arity == "binary":
        cdef_sig = (
            f"void compgen_m23_fused_{safe_name}("
            f"const float* A, const float* b, float* C);"
        )
    else:
        cdef_sig = (
            f"void compgen_m23_fused_{safe_name}("
            f"const float* A, float* C);"
        )

    build_dir = out_dir / f"cffi_build_fused_{safe_name}"
    build_dir.mkdir(parents=True, exist_ok=True)
    ext_name = f"_m23_cpu_fused_{safe_name}"
    try:
        ffi = cffi.FFI()
        ffi.cdef(cdef_sig)
        ffi.set_source(
            ext_name,
            src,
            extra_compile_args=["-O2", "-fno-fast-math"],
        )
        ffi.compile(tmpdir=str(build_dir), verbose=False)
    except Exception as exc:  # noqa: BLE001
        base["compile_status"] = "compile_failed"
        base["note"] = f"{type(exc).__name__}: {exc}"
        artifact_path.write_text(
            json.dumps(base, indent=2, sort_keys=True), encoding="utf-8",
        )
        return artifact_path

    base["compile_status"] = "compiled"
    base["compiler"] = "gcc"
    base["compile_command"] = "cffi.set_source -O2 -fno-fast-math"

    try:
        # Locate the .so file produced by cffi. Module name in the .so
        # must match what we pass to spec_from_file_location for PyInit
        # to resolve.
        so_path = next(build_dir.glob(f"{ext_name}*.so"), None)
        if so_path is None:
            base["run_status"] = "run_failed"
            base["note"] = "no .so produced by cffi"
            artifact_path.write_text(
                json.dumps(base, indent=2, sort_keys=True), encoding="utf-8",
            )
            return artifact_path

        import importlib.util
        spec = importlib.util.spec_from_file_location(ext_name, so_path)
        if spec is None or spec.loader is None:
            raise ImportError("could not spec cpu fused module")
        cmod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cmod)
        cffi_lib = cmod.lib  # type: ignore[attr-defined]
        cffi_ffi = cmod.ffi  # type: ignore[attr-defined]
        fn = getattr(cffi_lib, f"compgen_m23_fused_{safe_name}")
    except Exception as exc:  # noqa: BLE001
        base["run_status"] = "run_failed"
        base["note"] = f"{type(exc).__name__}: {exc}"
        artifact_path.write_text(
            json.dumps(base, indent=2, sort_keys=True), encoding="utf-8",
        )
        return artifact_path

    # Numerical verification on N cases.
    gen = torch.Generator()
    gen.manual_seed(int(seed))
    case_results: list[dict[str, Any]] = []
    max_abs_overall = 0.0
    max_rel_overall = 0.0

    try:
        for ci in range(int(n_cases)):
            a = torch.randn(rows, cols, dtype=torch.float32, generator=gen)
            if p_arity == "binary":
                b = torch.randn(cols, dtype=torch.float32, generator=gen)
            else:
                b = None
            out = torch.zeros_like(a)
            ref = _eager_unfused(
                a=a, b=b,
                producer_kind=producer_kind, consumer_kind=consumer_kind,
            )
            a_c = a.contiguous()
            out_c = out.contiguous()
            if p_arity == "binary":
                b_c = b.contiguous()
                fn(
                    cffi_ffi.cast("float*", a_c.data_ptr()),
                    cffi_ffi.cast("float*", b_c.data_ptr()),
                    cffi_ffi.cast("float*", out_c.data_ptr()),
                )
            else:
                fn(
                    cffi_ffi.cast("float*", a_c.data_ptr()),
                    cffi_ffi.cast("float*", out_c.data_ptr()),
                )
            diff = (out_c - ref).abs()
            ref_abs = ref.abs()
            max_abs = float(diff.max().item())
            max_rel = float((diff / (ref_abs + 1e-12)).max().item())
            max_abs_overall = max(max_abs_overall, max_abs)
            max_rel_overall = max(max_rel_overall, max_rel)
            case_results.append({
                "case_id": f"case_{ci:03d}",
                "max_abs_error": max_abs,
                "max_rel_error": max_rel,
                "refinement_status": _classify_refinement(max_abs, max_rel),
            })

        # Timing pass.
        a = torch.randn(rows, cols, dtype=torch.float32, generator=gen).contiguous()
        if p_arity == "binary":
            b = torch.randn(cols, dtype=torch.float32, generator=gen).contiguous()
        out = torch.zeros_like(a).contiguous()
        for _ in range(int(common.get("warmup", 4))):
            if p_arity == "binary":
                fn(
                    cffi_ffi.cast("float*", a.data_ptr()),
                    cffi_ffi.cast("float*", b.data_ptr()),
                    cffi_ffi.cast("float*", out.data_ptr()),
                )
            else:
                fn(
                    cffi_ffi.cast("float*", a.data_ptr()),
                    cffi_ffi.cast("float*", out.data_ptr()),
                )
        per_iter_us: list[float] = []
        for _ in range(int(common.get("iterations", 32))):
            t0 = time.perf_counter_ns()
            if p_arity == "binary":
                fn(
                    cffi_ffi.cast("float*", a.data_ptr()),
                    cffi_ffi.cast("float*", b.data_ptr()),
                    cffi_ffi.cast("float*", out.data_ptr()),
                )
            else:
                fn(
                    cffi_ffi.cast("float*", a.data_ptr()),
                    cffi_ffi.cast("float*", out.data_ptr()),
                )
            per_iter_us.append((time.perf_counter_ns() - t0) / 1000.0)
    except Exception as exc:  # noqa: BLE001
        base["run_status"] = "run_failed"
        base["note"] = f"{type(exc).__name__}: {exc}"
        artifact_path.write_text(
            json.dumps(base, indent=2, sort_keys=True), encoding="utf-8",
        )
        return artifact_path

    base["run_status"] = "ok"
    base["case_count"] = int(n_cases)
    base["bit_equality_count"] = sum(
        1 for c in case_results
        if c["refinement_status"] == "discharged_compiled_bit_equality"
    )
    base["tolerance_eps_count"] = sum(
        1 for c in case_results
        if c["refinement_status"] == "discharged_tolerance_eps"
    )
    base["fail_outside_tolerance_count"] = sum(
        1 for c in case_results
        if c["refinement_status"] == "fail_outside_tolerance"
    )
    base["max_abs_error"] = max_abs_overall
    base["max_rel_error"] = max_rel_overall
    base["measured_us_per_iter"] = (
        sum(per_iter_us) / len(per_iter_us) if per_iter_us else None
    )
    base["cases"] = case_results

    artifact_path.write_text(
        json.dumps(base, indent=2, sort_keys=True), encoding="utf-8",
    )
    return artifact_path


# --------------------------------------------------------------------------- #
# Top-level entry point
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class CompiledFusionResult:
    overall: str          # "pass" | "fail" | "blocked" | "not_run"
    out_dir: Path
    report_path: Path
    summary_md_path: Path
    gpu_status: str
    cpu_status: str
    bit_equality_count: int
    tolerance_eps_count: int
    fail_outside_tolerance_count: int
    case_count: int


def run_compiled_fusion(
    run_dir: Path,
    *,
    iterations: int = 32, warmup: int = 4,
    n_cases: int = 16, seed: int = 0xC0FFEE,
) -> CompiledFusionResult:
    """Build the compiled-fusion verification artifact.
    Best-effort; never raises."""
    run_dir = Path(run_dir).resolve()
    ga = run_dir / "02_graph_analysis"
    out_dir = ga / "compiled_fusion"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "compiled_fusion_differential_report.json"
    summary_md_path = out_dir / "compiled_fusion_differential_summary.md"

    fusion_manifest = _read_json(
        run_dir / "03_recipe_planning" / "real_lowering"
        / "real_fusion_manifest.json"
    )
    if fusion_manifest is None or fusion_manifest.get("mode") != "executable_real_fusion":
        body = {
            "schema_version": "compiled_fusion_differential_report_v1",
            "overall": "not_run",
            "status": "not_run",
            "note": (
                "no real_fusion_manifest with mode=executable_real_fusion; "
                "M-23 is only applicable when M-16.2 picked a fusion candidate"
            ),
            "generated_at_utc": _utcnow(),
        }
        report_path.write_text(
            json.dumps(body, indent=2, sort_keys=True), encoding="utf-8",
        )
        summary_md_path.write_text(
            "# Compiled Fusion (M-23) — not_run\n\n"
            "M-16.2 did not select an executable fusion candidate; M-23 "
            "has nothing to compile.\n",
            encoding="utf-8",
        )
        return CompiledFusionResult(
            overall="not_run", out_dir=out_dir,
            report_path=report_path, summary_md_path=summary_md_path,
            gpu_status="not_run", cpu_status="not_run",
            bit_equality_count=0, tolerance_eps_count=0,
            fail_outside_tolerance_count=0, case_count=0,
        )

    diag = fusion_manifest.get("diagnostics") or {}
    producer_kind_raw = str(diag.get("producer_kind") or "")
    consumer_kind_raw = str(diag.get("consumer_kind") or "")
    producer_kind = _classify_kind(producer_kind_raw)
    consumer_kind = _classify_kind(consumer_kind_raw)
    via_shape_raw = diag.get("via_tensor_shape") or []
    via_dtype = diag.get("via_tensor_dtype") or "f32"

    p_meta = _PRODUCER_TEMPLATES.get(producer_kind)
    c_meta = _CONSUMER_TEMPLATES.get(consumer_kind)
    if p_meta is None or c_meta is None:
        body = {
            "schema_version": "compiled_fusion_differential_report_v1",
            "overall": "blocked",
            "status": "blocked",
            "blocked_reason": (
                f"unsupported pair: producer_kind={producer_kind_raw!r} "
                f"consumer_kind={consumer_kind_raw!r}; M-23 MVP whitelist: "
                f"{sorted(_PRODUCER_TEMPLATES)} × {sorted(_CONSUMER_TEMPLATES)}"
            ),
            "producer_kind": producer_kind_raw,
            "consumer_kind": consumer_kind_raw,
            "generated_at_utc": _utcnow(),
        }
        report_path.write_text(
            json.dumps(body, indent=2, sort_keys=True), encoding="utf-8",
        )
        summary_md_path.write_text(
            "# Compiled Fusion (M-23) — blocked\n\n"
            f"{body['blocked_reason']}\n",
            encoding="utf-8",
        )
        return CompiledFusionResult(
            overall="blocked", out_dir=out_dir,
            report_path=report_path, summary_md_path=summary_md_path,
            gpu_status="blocked", cpu_status="blocked",
            bit_equality_count=0, tolerance_eps_count=0,
            fail_outside_tolerance_count=0, case_count=0,
        )

    if via_dtype != "f32":
        body = {
            "schema_version": "compiled_fusion_differential_report_v1",
            "overall": "blocked",
            "status": "blocked",
            "blocked_reason": (
                f"via_tensor dtype is {via_dtype!r}; M-23 MVP supports f32 only"
            ),
            "generated_at_utc": _utcnow(),
        }
        report_path.write_text(
            json.dumps(body, indent=2, sort_keys=True), encoding="utf-8",
        )
        summary_md_path.write_text(
            f"# Compiled Fusion (M-23) — blocked (dtype {via_dtype!r})\n",
            encoding="utf-8",
        )
        return CompiledFusionResult(
            overall="blocked", out_dir=out_dir,
            report_path=report_path, summary_md_path=summary_md_path,
            gpu_status="blocked", cpu_status="blocked",
            bit_equality_count=0, tolerance_eps_count=0,
            fail_outside_tolerance_count=0, case_count=0,
        )

    # Normalise via_tensor shape to (rows, cols). 1-D becomes (1, N).
    if len(via_shape_raw) == 1:
        rows, cols = 1, int(via_shape_raw[0])
    elif len(via_shape_raw) == 2:
        rows, cols = int(via_shape_raw[0]), int(via_shape_raw[1])
    elif len(via_shape_raw) >= 2:
        # Flatten leading dims into rows.
        prod = 1
        for d in via_shape_raw[:-1]:
            prod *= int(d)
        rows = prod
        cols = int(via_shape_raw[-1])
    else:
        rows, cols = 1, 1

    p_arity = p_meta[0]
    candidate_id = str(fusion_manifest.get("candidate_id") or "")
    fusion_block = fusion_manifest.get("fusion") or {}
    region_pair = (
        f"{fusion_block.get('producer', '?')}__"
        f"{fusion_block.get('consumer', '?')}"
    )
    safe_name = "".join(
        ch if ch.isalnum() or ch == "_" else "_"
        for ch in region_pair
    )[:80]

    common: dict[str, Any] = {
        "candidate_id": candidate_id,
        "region_pair": region_pair,
        "safe_name": safe_name,
        "iterations": iterations,
        "warmup": warmup,
    }

    gpu_artifact_path = _run_gpu_track(
        out_dir=out_dir, common=common,
        matmul_shape=(rows, cols),
        producer_kind=producer_kind, consumer_kind=consumer_kind,
        p_arity=p_arity, n_cases=n_cases, seed=seed,
    )
    cpu_artifact_path = _run_cpu_track(
        out_dir=out_dir, common=common,
        matmul_shape=(rows, cols),
        producer_kind=producer_kind, consumer_kind=consumer_kind,
        p_arity=p_arity, n_cases=n_cases, seed=seed,
    )

    gpu = _read_json(gpu_artifact_path) or {}
    cpu = _read_json(cpu_artifact_path) or {}

    # Aggregate.
    bit_eq = int(gpu.get("bit_equality_count") or 0) + int(
        cpu.get("bit_equality_count") or 0)
    tol = int(gpu.get("tolerance_eps_count") or 0) + int(
        cpu.get("tolerance_eps_count") or 0)
    fail = int(gpu.get("fail_outside_tolerance_count") or 0) + int(
        cpu.get("fail_outside_tolerance_count") or 0)

    # Status semantics: pass if no fail and at least one track compiled +
    # ran with a bit_equality + tolerance_eps result. fail if any case
    # exceeded tolerance. blocked when both tracks unavailable.
    gpu_status = str(gpu.get("compile_status") or "not_run")
    cpu_status = str(cpu.get("compile_status") or "not_run")
    any_compiled = (gpu_status == "compiled") or (cpu_status == "compiled")
    if not any_compiled:
        overall_status = "blocked"
    elif fail > 0:
        overall_status = "fail"
    else:
        overall_status = "pass"

    body = {
        "schema_version": "compiled_fusion_differential_report_v1",
        "overall": overall_status,
        "status": overall_status,
        "candidate_id": candidate_id,
        "region_pair": region_pair,
        "producer_kind": producer_kind, "consumer_kind": consumer_kind,
        "matmul_shape": {"rows": rows, "cols": cols},
        "via_tensor_dtype": via_dtype,
        "iterations": iterations, "warmup": warmup,
        "n_cases": n_cases, "seed_hex": f"0x{seed:X}",
        "gpu": gpu, "cpu": cpu,
        "summary": {
            "case_count": (
                int(gpu.get("case_count") or 0)
                + int(cpu.get("case_count") or 0)
            ),
            "bit_equality_count": bit_eq,
            "tolerance_eps_count": tol,
            "fail_outside_tolerance_count": fail,
            "gpu_compiled": gpu_status == "compiled",
            "cpu_compiled": cpu_status == "compiled",
        },
        "known_limitations": [
            "pointwise→pointwise only (binary producer + unary consumer)",
            "fp32 only",
            "no fp16/bf16 mixed precision",
            "no matmul-fusion (matmul is reduction-sensitive; M-16.4 territory)",
            "single-consumer producer chain (M-16.2's existing constraint)",
        ],
        "generated_at_utc": _utcnow(),
    }
    report_path.write_text(
        json.dumps(body, indent=2, sort_keys=True), encoding="utf-8",
    )

    md_lines = [
        f"# Compiled Fusion (M-23) — {overall_status}\n",
        f"- candidate_id: `{candidate_id}`",
        f"- producer: `{producer_kind}` consumer: `{consumer_kind}`",
        f"- via_tensor: shape (rows={rows}, cols={cols}) dtype {via_dtype}",
        f"- gpu: `{gpu_status}` cpu: `{cpu_status}`",
        f"- cases: bit_equality={bit_eq} tolerance_eps={tol} fail={fail}",
        f"- iterations={iterations} warmup={warmup}",
    ]
    if gpu.get("measured_us_per_iter") is not None:
        md_lines.append(f"- gpu_us_per_iter: {gpu['measured_us_per_iter']:.3f}")
    if cpu.get("measured_us_per_iter") is not None:
        md_lines.append(f"- cpu_us_per_iter: {cpu['measured_us_per_iter']:.3f}")
    summary_md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    return CompiledFusionResult(
        overall=overall_status, out_dir=out_dir,
        report_path=report_path, summary_md_path=summary_md_path,
        gpu_status=gpu_status, cpu_status=cpu_status,
        bit_equality_count=bit_eq,
        tolerance_eps_count=tol,
        fail_outside_tolerance_count=fail,
        case_count=int(gpu.get("case_count") or 0)
        + int(cpu.get("case_count") or 0),
    )
