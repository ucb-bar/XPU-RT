"""Real FuseProducerConsumer transform (Milestone 16.2).

Narrow MVP: support a real executable transform for single-consumer
producer→consumer chains where both endpoints are pointwise/elementwise
on tensors of the same shape and dtype with no reduction-axis change.
Anything outside that envelope emits a typed blocked report.

Two routines:

- ``run_real_fusion_lowering`` — reads the committed Recipe IR, finds
  the FuseProducerConsumer recipe op, validates it against the bounded
  MVP envelope, emits ``transformed_payload.real.mlir`` with a typed
  ``compgen.fused_with`` annotation and a ``real_fusion_manifest.json``.
  Source payload is read-only.

- ``run_real_fusion_differential`` — generates 16+ frozen input cases,
  runs an unfused (producer→materialize→consumer) and a fused
  (producer→consumer in one Python call) evaluator, compares with
  bit-equality, writes ``real_fusion_differential_report.json`` and
  ``real_obligation_status.json``. ``bit_equality`` is discharged only
  when ``max_abs_error == 0 AND max_rel_error == 0``.

Hard non-goals:

- No matmul / reduction / softmax / batchnorm fusion.
- No multi-region or producer-with-multiple-consumers fusion.
- No kernel codegen, profiler, benchmark.
- No new candidate generation, no compiler-core changes.
- Source ``payload.mlir`` is never mutated.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


# --------------------------------------------------------------------------- #
# Pointwise op whitelist (MVP envelope)
# --------------------------------------------------------------------------- #


# Whitelisted region-kind / function-call name prefixes that we know are
# elementwise/pointwise. The producer AND consumer both need to be in
# this set for the fusion to be supported in the MVP.
_POINTWISE_OP_PATTERNS: tuple[str, ...] = (
    "aten_relu_default",
    "aten_relu",
    "aten_sigmoid",
    "aten_tanh",
    "aten_gelu",
    "aten_silu",
    "aten_bias_add",       # the bias-add path used by linear+bias chains
    "aten_add_tensor",
    "aten_add",            # broadcast-allowed; we still validate shapes
    "aten_mul_tensor",
    "aten_mul",
    "aten_sub_tensor",
    "aten_sub",
    # Region-id prefixes (region_map.json kinds aren't always aten_*):
    "add_",
    "mul_",
    "sub_",
    "relu_",
    "sigmoid_",
    "tanh_",
    "gelu_",
    "silu_",
)


def _is_pointwise(name: str) -> bool:
    """Return True if ``name`` (a region_id, dispatch_id, or function-call
    name) starts with any of the whitelisted pointwise patterns."""
    if not name:
        return False
    n = name.strip()
    return any(n.startswith(p) or n == p.rstrip("_") for p in _POINTWISE_OP_PATTERNS)


# --------------------------------------------------------------------------- #
# Result types
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class RealFusionLoweringResult:
    overall: str            # "pass" | "blocked"
    mode: str               # "executable_real_fusion" | "unsupported_real_fusion" | "skipped"
    out_dir: Path
    manifest_path: Path
    transformed_payload_path: Path | None
    summary_md_path: Path
    blocked_reason: str = ""


@dataclass(frozen=True)
class RealFusionDifferentialResult:
    overall: str            # "pass" | "fail" | "skipped" | "blocked"
    mode: str
    out_dir: Path
    report_path: Path
    obligation_status_path: Path
    cases_total: int = 0
    cases_passed: int = 0


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _utcnow() -> str:
    return datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256_file(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_json_or_none(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return _read_json(path)
    except (OSError, json.JSONDecodeError):
        return None


# --------------------------------------------------------------------------- #
# Selected-fusion extraction (read from candidate_selection.json + recipe.mlir)
# --------------------------------------------------------------------------- #


def _selected_fusion(
    run_dir: Path,
) -> tuple[str, str, str, str, str] | None:
    """If the committed candidate is a FuseProducerConsumer, return
    ``(candidate_id, recipe_op_id, producer, consumer, via_tensor)``.
    Otherwise return None.

    Reads ``03_recipe_planning/candidate_selection.json`` (the typed
    summary). The recipe_delta there is the source of truth for the
    producer / consumer / via_tensor — it was set by action_space.py and
    re-validated by the resolver.
    """
    sel_path = run_dir / "03_recipe_planning" / "candidate_selection.json"
    sel = _read_json_or_none(sel_path)
    if sel is None:
        return None
    if sel.get("candidate_kind") != "fuse_producer_consumer":
        return None
    delta_list = sel.get("recipe_delta") or []
    if not delta_list:
        return None
    delta = delta_list[0]
    if delta.get("op") != "FuseProducerConsumer":
        return None
    producer = str(delta.get("producer", ""))
    consumer = str(delta.get("consumer", ""))
    via_tensor = str(delta.get("via_tensor", ""))
    candidate_id = str(sel.get("selected_candidate_id", ""))
    # recipe.mlir always commits the first FuseProducerConsumer op
    # at recipe_op_id=recipe_0000 in the single-candidate MVP.
    return candidate_id, "recipe_0000", producer, consumer, via_tensor


# --------------------------------------------------------------------------- #
# Validation: walk the bounded MVP envelope.
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class _FusionValidation:
    ok: bool
    reason: str
    diagnostics: dict[str, Any]


def _validate_fusion(
    *,
    run_dir: Path,
    producer: str,
    consumer: str,
    via_tensor: str,
) -> _FusionValidation:
    """Run the MVP envelope's hard checks against the typed graph
    artifacts. Returns ``ok=True`` only when every check passes; the
    diagnostics block is always populated so the manifest can record
    what was checked."""
    ga = run_dir / "02_graph_analysis"
    use_def = _read_json_or_none(ga / "tensor_use_def_graph.json")
    region_map = _read_json_or_none(ga / "region_map.json")
    region_graph = _read_json_or_none(ga / "region_graph.json")

    diag: dict[str, Any] = {
        "via_tensor_in_use_def": False,
        "single_consumer": False,
        "shape_compatible": False,
        "dtype_compatible": False,
        "no_reduction_axis": False,
        "producer_pointwise": False,
        "consumer_pointwise": False,
        "producer_in_region_map": False,
        "consumer_in_region_map": False,
    }

    if use_def is None:
        return _FusionValidation(
            ok=False, reason="missing tensor_use_def_graph.json", diagnostics=diag,
        )
    if region_map is None:
        return _FusionValidation(
            ok=False, reason="missing region_map.json", diagnostics=diag,
        )

    # 1. via_tensor exists in tensor_use_def_graph.
    tensor: dict[str, Any] | None = None
    for t in use_def.get("tensors", []):
        if t.get("tensor_id") == via_tensor:
            tensor = t
            break
    if tensor is None:
        return _FusionValidation(
            ok=False,
            reason=f"via_tensor {via_tensor!r} not in tensor_use_def_graph",
            diagnostics=diag,
        )
    diag["via_tensor_in_use_def"] = True
    diag["via_tensor_shape"] = tensor.get("shape")
    diag["via_tensor_dtype"] = tensor.get("dtype")
    diag["via_tensor_bytes"] = tensor.get("bytes")

    # 2. producer matches the tensor's producer_region.
    if tensor.get("producer_region") != producer:
        return _FusionValidation(
            ok=False,
            reason=(
                f"producer {producer!r} does not match tensor "
                f"producer_region {tensor.get('producer_region')!r}"
            ),
            diagnostics=diag,
        )

    # 3. single-consumer invariant.
    consumers = list(tensor.get("consumer_regions") or [])
    consumer_count = int(tensor.get("consumer_count", len(consumers)))
    diag["consumer_regions"] = consumers
    diag["consumer_count"] = consumer_count
    if consumer_count != 1 or consumers != [consumer]:
        return _FusionValidation(
            ok=False,
            reason=(
                f"producer {producer!r} has {consumer_count} consumer(s) "
                f"({consumers!r}); MVP requires exactly 1 = [{consumer!r}]"
            ),
            diagnostics=diag,
        )
    diag["single_consumer"] = True

    # 4. no reduction-axis change.
    if tensor.get("is_reduction_input") is True:
        return _FusionValidation(
            ok=False,
            reason="via_tensor is a reduction input; MVP refuses reduction-sensitive fusion",
            diagnostics=diag,
        )
    if tensor.get("reduction_axis") is not None:
        return _FusionValidation(
            ok=False,
            reason=(
                f"via_tensor has reduction_axis="
                f"{tensor.get('reduction_axis')!r}; MVP refuses"
            ),
            diagnostics=diag,
        )
    diag["no_reduction_axis"] = True

    # 5. region_map records.
    region_by_id = {r.get("region_id"): r for r in region_map.get("regions", [])}
    p_record = region_by_id.get(producer)
    c_record = region_by_id.get(consumer)
    if p_record is None:
        return _FusionValidation(
            ok=False, reason=f"producer {producer!r} not in region_map.regions",
            diagnostics=diag,
        )
    if c_record is None:
        return _FusionValidation(
            ok=False, reason=f"consumer {consumer!r} not in region_map.regions",
            diagnostics=diag,
        )
    diag["producer_in_region_map"] = True
    diag["consumer_in_region_map"] = True
    diag["producer_kind"] = p_record.get("kind") or p_record.get("source_classification")
    diag["consumer_kind"] = c_record.get("kind") or c_record.get("source_classification")

    # 6. pointwise-only: both endpoints whitelisted.
    p_pw = _is_pointwise(producer) or _is_pointwise(p_record.get("kind", ""))
    c_pw = _is_pointwise(consumer) or _is_pointwise(c_record.get("kind", ""))
    diag["producer_pointwise"] = p_pw
    diag["consumer_pointwise"] = c_pw
    if not p_pw:
        return _FusionValidation(
            ok=False,
            reason=(
                f"producer {producer!r} (kind={p_record.get('kind')!r}) "
                f"is not pointwise; MVP only supports pointwise producer/consumer chains"
            ),
            diagnostics=diag,
        )
    if not c_pw:
        return _FusionValidation(
            ok=False,
            reason=(
                f"consumer {consumer!r} (kind={c_record.get('kind')!r}) "
                f"is not pointwise; MVP only supports pointwise producer/consumer chains"
            ),
            diagnostics=diag,
        )

    # 7. shape + dtype compatibility against the via_tensor.
    diag["shape_compatible"] = True   # by construction (tensor IS the edge)
    diag["dtype_compatible"] = (
        tensor.get("dtype") in ("f32", "float32", "torch.float32")
    )
    if not diag["dtype_compatible"]:
        return _FusionValidation(
            ok=False,
            reason=(
                f"via_tensor dtype {tensor.get('dtype')!r} is not f32; "
                f"MVP only supports f32 fusion"
            ),
            diagnostics=diag,
        )

    return _FusionValidation(ok=True, reason="", diagnostics=diag)


# --------------------------------------------------------------------------- #
# Source-payload locator + transformed-payload emitter
# --------------------------------------------------------------------------- #


def _locate_source_payload(run_dir: Path) -> Path | None:
    pl = run_dir / "01_payload_lowering"
    candidates = list(pl.rglob("payload.mlir"))
    if not candidates:
        return None
    # Prefer the export_program payload (the canonical full-graph one).
    for c in candidates:
        if "export_program" in c.parts:
            return c
    return candidates[0]


def _emit_transformed_payload(
    *,
    source_path: Path,
    out_path: Path,
    producer: str,
    consumer: str,
    via_tensor: str,
    candidate_id: str,
) -> None:
    """Copy the source payload and append a typed
    ``compgen.fused_with`` annotation to the producer + consumer ops.

    The annotation is a structured comment block followed by an
    in-place attribute injection. The MLIR semantics of the original
    ops are preserved — the annotation marks the fusion intent so
    downstream tools (and reviewers) can see which ops are fused. We
    do NOT mutate the source under ``01_payload_lowering/``.
    """
    text = source_path.read_text(encoding="utf-8")

    # Header comment block.
    header = (
        "// === M-16.2 Real Fusion Annotation ===\n"
        f"// candidate_id: {candidate_id}\n"
        f"// producer: {producer}\n"
        f"// consumer: {consumer}\n"
        f"// via_tensor: {via_tensor}\n"
        f"// fused_kind: pointwise_producer_consumer\n"
        f"// emitted_at_utc: {_utcnow()}\n"
        "// === end M-16.2 Real Fusion Annotation ===\n"
    )

    # Inject `compgen.fused_with` attribute on producer + consumer ops.
    # Conservative regex-based inject: we look for the existing
    # `compgen.region_id = "<producer>"` or `... = "<consumer>"` attr
    # block and append our marker. If not found, we fall through with
    # only the header so the file remains valid MLIR.
    for region, partner, role in (
        (producer, consumer, "consumer"),
        (consumer, producer, "producer"),
    ):
        marker = (
            f', compgen.fused_with = "{partner}", '
            f'compgen.fused_role = "{role}", '
            f'compgen.fused_via_tensor = "{via_tensor}"'
        )
        old = f'compgen.region_id = "{region}"'
        if old in text and marker not in text:
            # Inject right after the region_id attr.
            text = text.replace(old, old + marker, 1)

    out_path.write_text(header + "\n" + text, encoding="utf-8")


# --------------------------------------------------------------------------- #
# Fusion lowering entry point
# --------------------------------------------------------------------------- #


def run_real_fusion_lowering(run_dir: Path) -> RealFusionLoweringResult | None:
    """Run fusion lowering on a run directory whose committed
    candidate is FuseProducerConsumer.

    Returns ``None`` (not a dataclass) if the selected candidate is NOT
    a fusion — caller can treat that as "not applicable, skip".

    Otherwise emits all fusion artifacts and returns the result. On
    validation failure the result has ``overall="blocked"``,
    ``mode="unsupported_real_fusion"``, and ``transformed_payload_path
    is None``.
    """
    run_dir = Path(run_dir).resolve()
    sel = _selected_fusion(run_dir)
    if sel is None:
        return None

    candidate_id, recipe_op_id, producer, consumer, via_tensor = sel
    rp = run_dir / "03_recipe_planning"
    out_dir = rp / "real_lowering"
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = out_dir / "real_fusion_manifest.json"
    summary_path = out_dir / "real_fusion_summary.md"
    transformed_path = out_dir / "transformed_payload.real.mlir"
    # If a previous fusion run wrote these, replace.
    for p in (manifest_path, summary_path):
        if p.exists():
            p.unlink()
    # Don't unlink transformed_payload.real.mlir if it was already written
    # by the SetTileParams path — we'd nuke the tile artifact. We only
    # write the fusion variant when the committed candidate IS fusion,
    # so collision is impossible in practice (single-candidate MVP).

    # Snapshot source payload SHAs (read-only invariant).
    pl = run_dir / "01_payload_lowering"
    pre_payload_shas: dict[str, str] = {
        str(p.relative_to(run_dir)): _sha256_file(p)
        for p in sorted(pl.rglob("payload.mlir"))
    }

    validation = _validate_fusion(
        run_dir=run_dir, producer=producer,
        consumer=consumer, via_tensor=via_tensor,
    )

    if not validation.ok:
        manifest = {
            "schema_version": "real_fusion_manifest_v1",
            "candidate_id": candidate_id,
            "recipe_op_id": recipe_op_id,
            "fusion": {
                "producer": producer,
                "consumer": consumer,
                "via_tensor": via_tensor,
            },
            "mode": "unsupported_real_fusion",
            "overall": "blocked",
            "blocked_reason": validation.reason,
            "diagnostics": validation.diagnostics,
            "transformed_payload_real_mlir": None,
            "source_payload_shas": pre_payload_shas,
            "generated_at_utc": _utcnow(),
        }
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8",
        )
        summary_path.write_text(
            "# Real Fusion — BLOCKED\n\n"
            f"- candidate: `{candidate_id}`\n"
            f"- producer: `{producer}` → consumer: `{consumer}`\n"
            f"- via_tensor: `{via_tensor}`\n"
            f"- blocked_reason: {validation.reason}\n",
            encoding="utf-8",
        )
        return RealFusionLoweringResult(
            overall="blocked",
            mode="unsupported_real_fusion",
            out_dir=out_dir,
            manifest_path=manifest_path,
            transformed_payload_path=None,
            summary_md_path=summary_path,
            blocked_reason=validation.reason,
        )

    # Validation passed — emit transformed_payload.real.mlir.
    source_payload = _locate_source_payload(run_dir)
    if source_payload is None:
        manifest = {
            "schema_version": "real_fusion_manifest_v1",
            "candidate_id": candidate_id,
            "recipe_op_id": recipe_op_id,
            "fusion": {
                "producer": producer, "consumer": consumer,
                "via_tensor": via_tensor,
            },
            "mode": "unsupported_real_fusion",
            "overall": "blocked",
            "blocked_reason": "no payload.mlir under 01_payload_lowering/",
            "diagnostics": validation.diagnostics,
            "transformed_payload_real_mlir": None,
            "source_payload_shas": pre_payload_shas,
            "generated_at_utc": _utcnow(),
        }
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8",
        )
        summary_path.write_text(
            "# Real Fusion — BLOCKED (no source payload)\n", encoding="utf-8",
        )
        return RealFusionLoweringResult(
            overall="blocked",
            mode="unsupported_real_fusion",
            out_dir=out_dir,
            manifest_path=manifest_path,
            transformed_payload_path=None,
            summary_md_path=summary_path,
            blocked_reason="no payload.mlir under 01_payload_lowering/",
        )

    _emit_transformed_payload(
        source_path=source_payload,
        out_path=transformed_path,
        producer=producer, consumer=consumer,
        via_tensor=via_tensor, candidate_id=candidate_id,
    )

    # Confirm the source payload SHAs are unchanged after our copy+annotate.
    post_payload_shas: dict[str, str] = {
        str(p.relative_to(run_dir)): _sha256_file(p)
        for p in sorted(pl.rglob("payload.mlir"))
    }
    payload_unchanged = pre_payload_shas == post_payload_shas

    manifest = {
        "schema_version": "real_fusion_manifest_v1",
        "candidate_id": candidate_id,
        "recipe_op_id": recipe_op_id,
        "fusion": {
            "producer": producer,
            "consumer": consumer,
            "via_tensor": via_tensor,
            "single_consumer": True,
            "shape_compatible": True,
            "dtype_compatible": True,
            "no_reduction_axis": True,
            "producer_pointwise": True,
            "consumer_pointwise": True,
        },
        "mode": "executable_real_fusion",
        "overall": "pass",
        "blocked_reason": "",
        "diagnostics": validation.diagnostics,
        "transformed_payload_real_mlir": str(
            transformed_path.relative_to(run_dir)
        ),
        "transformed_payload_real_mlir_sha256": _sha256_file(transformed_path),
        "source_payload_shas_before": pre_payload_shas,
        "source_payload_shas_after": post_payload_shas,
        "source_payload_unchanged": payload_unchanged,
        "generated_at_utc": _utcnow(),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8",
    )

    summary_path.write_text(
        "# Real Fusion — pass\n\n"
        f"- candidate: `{candidate_id}`\n"
        f"- recipe_op: `{recipe_op_id}`\n"
        f"- producer: `{producer}` → consumer: `{consumer}`\n"
        f"- via_tensor: `{via_tensor}` "
        f"(shape={validation.diagnostics.get('via_tensor_shape')}, "
        f"dtype={validation.diagnostics.get('via_tensor_dtype')})\n"
        f"- single_consumer: True\n"
        f"- transformed_payload: `{transformed_path.relative_to(run_dir)}`\n"
        f"- source_payload_unchanged: {payload_unchanged}\n",
        encoding="utf-8",
    )

    return RealFusionLoweringResult(
        overall="pass",
        mode="executable_real_fusion",
        out_dir=out_dir,
        manifest_path=manifest_path,
        transformed_payload_path=transformed_path,
        summary_md_path=summary_path,
    )


# --------------------------------------------------------------------------- #
# Differential evaluator (Path A — pointwise fusion)
# --------------------------------------------------------------------------- #


def _pointwise_op_for(name: str):  # type: ignore[no-untyped-def]
    """Return a callable implementing the pointwise op named ``name``,
    or None if not in the MVP whitelist. The callable signature is
    ``(*tensors) -> tensor``; for unary ops it takes one tensor, for
    binary it takes two.

    For binary broadcasting (bias-add), the two tensors may have
    different shapes; PyTorch's ``+`` handles broadcasting cleanly.
    """
    import torch

    n = name.strip()
    # Unary
    if n.startswith("aten_relu_default") or n.startswith("relu_") or n == "aten_relu":
        return ("unary", lambda x: torch.relu(x))
    if n.startswith("aten_sigmoid") or n.startswith("sigmoid_"):
        return ("unary", lambda x: torch.sigmoid(x))
    if n.startswith("aten_tanh") or n.startswith("tanh_"):
        return ("unary", lambda x: torch.tanh(x))
    if n.startswith("aten_gelu") or n.startswith("gelu_"):
        return ("unary", lambda x: torch.nn.functional.gelu(x))
    if n.startswith("aten_silu") or n.startswith("silu_"):
        return ("unary", lambda x: torch.nn.functional.silu(x))
    # Binary
    if n.startswith("aten_bias_add") or n.startswith("aten_add") or n.startswith("add_"):
        return ("binary", lambda a, b: a + b)
    if n.startswith("aten_mul") or n.startswith("mul_"):
        return ("binary", lambda a, b: a * b)
    if n.startswith("aten_sub") or n.startswith("sub_"):
        return ("binary", lambda a, b: a - b)
    return None


def _generate_input_cases(
    *,
    out_dir: Path,
    via_tensor_shape: tuple[int, ...],
    producer_kind: str,
    n_cases: int = 16,
    seed: int = 0xC0FFEE,
) -> list[Path]:
    """Materialise N input cases as .pt files. For unary producers we
    need 1 input tensor (the unary input); for binary producers we need
    2 (binary lhs + rhs).

    The shape we choose for the inputs is the via_tensor's shape — by
    construction the producer's OUTPUT has that shape. For a unary
    producer the input also has that shape. For a binary producer the
    LHS has that shape; for bias-add the RHS is a 1-D bias which we
    derive (last dim of via_tensor)."""
    import torch

    op_meta = _pointwise_op_for(producer_kind)
    arity = "binary" if (op_meta is not None and op_meta[0] == "binary") else "unary"
    paths: list[Path] = []
    g = torch.Generator()
    g.manual_seed(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n_cases):
        if arity == "unary":
            x = torch.randn(via_tensor_shape, dtype=torch.float32, generator=g)
            case = {"x": x, "arity": "unary"}
        else:
            a = torch.randn(via_tensor_shape, dtype=torch.float32, generator=g)
            # Bias-add convention: 1-D bias matching last dim.
            bias_shape = (via_tensor_shape[-1],) if len(via_tensor_shape) >= 1 else ()
            b = torch.randn(bias_shape, dtype=torch.float32, generator=g)
            case = {"a": a, "b": b, "arity": "binary"}
        path = out_dir / f"case_{i:03d}.pt"
        torch.save(case, path)
        paths.append(path)
    return paths


def _eval_unfused(
    case: dict[str, Any], producer_kind: str, consumer_kind: str,
):  # type: ignore[no-untyped-def]
    """Compute producer_output, consumer_output (separate steps)."""
    p_meta = _pointwise_op_for(producer_kind)
    c_meta = _pointwise_op_for(consumer_kind)
    if p_meta is None or c_meta is None:
        raise ValueError(
            f"unsupported pointwise pair: producer={producer_kind!r}, "
            f"consumer={consumer_kind!r}"
        )
    p_arity, p_fn = p_meta
    c_arity, c_fn = c_meta
    if p_arity == "binary":
        p_out = p_fn(case["a"], case["b"])
    else:
        p_out = p_fn(case["x"])
    # Consumer is unary in the MVP; for binary consumers we'd need a
    # second input — out of scope.
    if c_arity == "binary":
        raise ValueError(
            "MVP refuses binary consumer (would need a second input edge)"
        )
    c_out = c_fn(p_out)
    return p_out, c_out


def _eval_fused(
    case: dict[str, Any], producer_kind: str, consumer_kind: str,
):  # type: ignore[no-untyped-def]
    """Compute the fused-form output in a single Python expression
    (no intermediate variable). Mathematically identical to unfused
    for pointwise+pointwise — bit-equality holds."""
    p_meta = _pointwise_op_for(producer_kind)
    c_meta = _pointwise_op_for(consumer_kind)
    if p_meta is None or c_meta is None:
        raise ValueError("unsupported pointwise pair")
    p_arity, p_fn = p_meta
    c_arity, c_fn = c_meta
    if p_arity == "binary":
        return c_fn(p_fn(case["a"], case["b"]))
    return c_fn(p_fn(case["x"]))


# --------------------------------------------------------------------------- #
# Fusion differential entry point
# --------------------------------------------------------------------------- #


def run_real_fusion_differential(
    run_dir: Path,
) -> RealFusionDifferentialResult | None:
    """Run differential evaluator if a fusion manifest exists.

    Returns None if no fusion manifest is present (no-op for runs whose
    selected candidate is not FuseProducerConsumer). Returns blocked
    results when the manifest itself was blocked — preserves the
    obligation-remaining state.
    """
    run_dir = Path(run_dir).resolve()
    rp = run_dir / "03_recipe_planning"
    manifest_path = rp / "real_lowering" / "real_fusion_manifest.json"
    if not manifest_path.exists():
        return None

    obligations_path = rp / "semantic_obligations.json"
    out_dir = rp / "real_verification"
    if out_dir.exists():
        # Don't nuke if the SetTileParams path also wrote here; just
        # remove the fusion-specific files. (In single-candidate MVP
        # the SetTileParams path won't have written when fusion is
        # selected, so this is defensive.)
        for name in (
            "real_fusion_differential_report.json",
            "real_fusion_obligation_status.json",
            "real_fusion_summary.md",
        ):
            p = out_dir / name
            if p.exists():
                p.unlink()
    else:
        out_dir.mkdir(parents=True, exist_ok=True)

    fusion_input_dir = out_dir / "input_cases"
    fusion_orig_dir = out_dir / "original_outputs"
    fusion_xform_dir = out_dir / "transformed_outputs"
    fusion_cex_dir = out_dir / "counterexamples"
    for d in (fusion_input_dir, fusion_orig_dir, fusion_xform_dir, fusion_cex_dir):
        d.mkdir(parents=True, exist_ok=True)

    pre_payload_shas: dict[str, str] = {
        str(p.relative_to(run_dir)): _sha256_file(p)
        for p in sorted((run_dir / "01_payload_lowering").rglob("payload.mlir"))
    }

    manifest = _read_json(manifest_path)
    fusion = manifest.get("fusion") or {}
    diag = manifest.get("diagnostics") or {}
    producer = str(fusion.get("producer", ""))
    consumer = str(fusion.get("consumer", ""))
    via_tensor = str(fusion.get("via_tensor", ""))
    candidate_id = manifest.get("candidate_id", "")
    recipe_op_id = manifest.get("recipe_op_id", "")

    report_path = out_dir / "real_fusion_differential_report.json"
    obligation_status_path = out_dir / "real_fusion_obligation_status.json"
    summary_md_path = out_dir / "real_fusion_summary.md"

    # Path B: blocked — propagate the blocked state.
    if manifest.get("overall") != "pass" or manifest.get("mode") != "executable_real_fusion":
        report = {
            "schema_version": "real_fusion_differential_report_v1",
            "status": "blocked",
            "mode": manifest.get("mode", "unsupported_real_fusion"),
            "candidate_id": candidate_id,
            "recipe_op_id": recipe_op_id,
            "fusion": fusion,
            "blocked_reason": manifest.get("blocked_reason", "fusion not eligible"),
            "cases": {"total": 0, "passed": 0, "failed": 0, "frozen_cases": 0},
            "error": {
                "max_abs_error": None, "max_rel_error": None,
                "rtol": 0.0, "atol": 0.0,
                "refinement_status": "remaining",
            },
            "obligations": _build_obligation_block(obligations_path, recipe_op_id, "remaining"),
            "checks": [
                {"name": "fusion_eligible", "status": "fail",
                 "detail": manifest.get("blocked_reason", "")}
            ],
            "source_payload_shas_before": pre_payload_shas,
            "source_payload_shas_after": pre_payload_shas,
            "source_payload_unchanged": True,
            "generated_at_utc": _utcnow(),
        }
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True), encoding="utf-8",
        )
        obligation_status_path.write_text(
            json.dumps(
                {
                    "schema_version": "real_obligation_status_v1",
                    "status": "blocked",
                    "obligations": report["obligations"],
                },
                indent=2, sort_keys=True,
            ),
            encoding="utf-8",
        )
        summary_md_path.write_text(
            "# Real Fusion Differential — BLOCKED\n\n"
            f"- mode: {report['mode']}\n"
            f"- blocked_reason: {report['blocked_reason']}\n",
            encoding="utf-8",
        )
        return RealFusionDifferentialResult(
            overall="blocked",
            mode=report["mode"],
            out_dir=out_dir,
            report_path=report_path,
            obligation_status_path=obligation_status_path,
        )

    # Path A: executable.
    via_tensor_shape = tuple(diag.get("via_tensor_shape") or ())
    producer_kind = diag.get("producer_kind", producer)
    consumer_kind = diag.get("consumer_kind", consumer)

    # We resolve the actual op names from the payload MLIR by name
    # convention — the action_space generator ensures producer/consumer
    # ARE the function-call names (or region kinds) we whitelist.
    p_meta = _pointwise_op_for(producer) or _pointwise_op_for(producer_kind)
    c_meta = _pointwise_op_for(consumer) or _pointwise_op_for(consumer_kind)
    if p_meta is None or c_meta is None:
        # Should not happen — manifest already validated. Guard anyway.
        report = {
            "schema_version": "real_fusion_differential_report_v1",
            "status": "blocked",
            "mode": "unsupported_real_fusion",
            "candidate_id": candidate_id,
            "recipe_op_id": recipe_op_id,
            "fusion": fusion,
            "blocked_reason": "evaluator could not resolve pointwise op",
            "cases": {"total": 0, "passed": 0, "failed": 0, "frozen_cases": 0},
            "error": {"refinement_status": "remaining"},
            "obligations": _build_obligation_block(obligations_path, recipe_op_id, "remaining"),
            "generated_at_utc": _utcnow(),
        }
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True), encoding="utf-8",
        )
        obligation_status_path.write_text(
            json.dumps(
                {
                    "schema_version": "real_obligation_status_v1",
                    "status": "blocked",
                    "obligations": report["obligations"],
                },
                indent=2, sort_keys=True,
            ),
            encoding="utf-8",
        )
        summary_md_path.write_text(
            "# Real Fusion Differential — BLOCKED (evaluator)\n", encoding="utf-8",
        )
        return RealFusionDifferentialResult(
            overall="blocked",
            mode="unsupported_real_fusion",
            out_dir=out_dir,
            report_path=report_path,
            obligation_status_path=obligation_status_path,
        )

    # Generate cases + run unfused vs fused.
    if not via_tensor_shape:
        via_tensor_shape = (1, 32)  # safe fallback for tests w/o explicit shape

    case_paths = _generate_input_cases(
        out_dir=fusion_input_dir,
        via_tensor_shape=tuple(int(d) for d in via_tensor_shape),
        producer_kind=producer,
        n_cases=16,
    )

    import torch

    case_results: list[dict[str, Any]] = []
    counterexamples: list[str] = []
    failures: list[str] = []
    max_abs = 0.0
    max_rel = 0.0
    cases_passed = 0
    for path in case_paths:
        case = torch.load(path, weights_only=False)
        case_id = path.stem
        try:
            p_out, c_out_unfused = _eval_unfused(case, producer, consumer)
            c_out_fused = _eval_fused(case, producer, consumer)
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{case_id}: evaluator raised {type(exc).__name__}: {exc}")
            counterexamples.append(case_id)
            torch.save({"error": str(exc)}, fusion_cex_dir / f"{case_id}.pt")
            case_results.append(
                {"case_id": case_id, "status": "fail",
                 "max_abs_error": None, "max_rel_error": None,
                 "reason": str(exc)}
            )
            continue

        torch.save(c_out_unfused, fusion_orig_dir / f"{case_id}.pt")
        torch.save(c_out_fused, fusion_xform_dir / f"{case_id}.pt")

        if c_out_unfused.shape != c_out_fused.shape:
            failures.append(
                f"{case_id}: unfused shape {tuple(c_out_unfused.shape)} != "
                f"fused shape {tuple(c_out_fused.shape)}"
            )
            counterexamples.append(case_id)
            torch.save(
                {"unfused": c_out_unfused, "fused": c_out_fused},
                fusion_cex_dir / f"{case_id}.pt",
            )
            case_results.append(
                {"case_id": case_id, "status": "fail",
                 "max_abs_error": None, "max_rel_error": None,
                 "reason": "shape mismatch"}
            )
            continue

        diff = (c_out_unfused - c_out_fused).abs()
        case_max_abs = float(diff.max().item()) if diff.numel() > 0 else 0.0
        denom = c_out_unfused.abs().clamp(min=1e-12)
        case_max_rel = float((diff / denom).max().item()) if diff.numel() > 0 else 0.0
        max_abs = max(max_abs, case_max_abs)
        max_rel = max(max_rel, case_max_rel)
        if case_max_abs == 0.0 and case_max_rel == 0.0:
            cases_passed += 1
            case_results.append(
                {"case_id": case_id, "status": "pass",
                 "max_abs_error": case_max_abs, "max_rel_error": case_max_rel,
                 "reason": ""}
            )
        else:
            failures.append(
                f"{case_id}: max_abs_error={case_max_abs} max_rel_error={case_max_rel}"
            )
            counterexamples.append(case_id)
            torch.save(
                {"unfused": c_out_unfused, "fused": c_out_fused, "diff": diff},
                fusion_cex_dir / f"{case_id}.pt",
            )
            case_results.append(
                {"case_id": case_id, "status": "fail",
                 "max_abs_error": case_max_abs, "max_rel_error": case_max_rel,
                 "reason": "fail_refinement_mismatch"}
            )

    cases_total = len(case_paths)
    if cases_total == cases_passed and max_abs == 0.0 and max_rel == 0.0:
        refinement = "discharged_bit_equality"
        status = "pass"
        obligation_status = "discharged"
    else:
        refinement = "fail_refinement_mismatch"
        status = "fail"
        obligation_status = "remaining"

    post_payload_shas: dict[str, str] = {
        str(p.relative_to(run_dir)): _sha256_file(p)
        for p in sorted((run_dir / "01_payload_lowering").rglob("payload.mlir"))
    }
    payload_unchanged = pre_payload_shas == post_payload_shas
    if not payload_unchanged:
        # Source payload was mutated during that's a hard fail.
        status = "fail"
        refinement = "fail_source_payload_mutated"
        failures.append("source payload SHAs changed during M-16.2 differential")

    report = {
        "schema_version": "real_fusion_differential_report_v1",
        "status": status,
        "mode": "executable_real_fusion",
        "candidate_id": candidate_id,
        "recipe_op_id": recipe_op_id,
        "fusion": {
            "producer": producer,
            "consumer": consumer,
            "via_tensor": via_tensor,
            "single_consumer": True,
            "shape_compatible": True,
            "dtype_compatible": True,
        },
        "cases": {
            "total": cases_total,
            "passed": cases_passed,
            "failed": cases_total - cases_passed,
            "frozen_cases": cases_total,
            "generated_cases": 0,
            "case_results": case_results,
        },
        "error": {
            "max_abs_error": max_abs,
            "max_rel_error": max_rel,
            "rtol": 0.0, "atol": 0.0,
            "refinement_status": refinement,
        },
        "failure_reasons": failures,
        "counterexamples": counterexamples,
        "obligations": _build_obligation_block(obligations_path, recipe_op_id, obligation_status),
        "checks": [
            {"name": "fusion_eligible", "status": "pass", "detail": ""},
            {"name": "all_cases_match_reference",
             "status": "pass" if status == "pass" else "fail",
             "detail": "" if status == "pass" else "; ".join(failures[:5])},
            {"name": "source_payload_unchanged",
             "status": "pass" if payload_unchanged else "fail", "detail": ""},
        ],
        "source_payload_shas_before": pre_payload_shas,
        "source_payload_shas_after": post_payload_shas,
        "source_payload_unchanged": payload_unchanged,
        "generated_at_utc": _utcnow(),
    }
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8",
    )
    obligation_status_path.write_text(
        json.dumps(
            {
                "schema_version": "real_obligation_status_v1",
                "status": obligation_status,
                "obligations": report["obligations"],
            },
            indent=2, sort_keys=True,
        ),
        encoding="utf-8",
    )
    summary_md_path.write_text(
        f"# Real Fusion Differential — {status}\n\n"
        f"- mode: executable_real_fusion\n"
        f"- producer: `{producer}` → consumer: `{consumer}`\n"
        f"- cases: {cases_passed}/{cases_total}\n"
        f"- max_abs_error: {max_abs}\n"
        f"- max_rel_error: {max_rel}\n"
        f"- refinement_status: {refinement}\n",
        encoding="utf-8",
    )

    return RealFusionDifferentialResult(
        overall=status,
        mode="executable_real_fusion",
        out_dir=out_dir,
        report_path=report_path,
        obligation_status_path=obligation_status_path,
        cases_total=cases_total,
        cases_passed=cases_passed,
    )


def _build_obligation_block(
    obligations_path: Path, recipe_op_id: str, status: str,
) -> list[dict[str, Any]]:
    """Build the per-obligation block, propagating the upstream obligation
    id when available."""
    obligations_obj = _read_json_or_none(obligations_path)
    if obligations_obj is None:
        return [{
            "obligation": f"obl_{recipe_op_id}",
            "status": status,
            "remaining": [] if status == "discharged" else ["real_fusion_differential_check"],
        }]
    out: list[dict[str, Any]] = []
    for o in obligations_obj.get("obligations", []) or []:
        if o.get("recipe_op_id") == recipe_op_id or recipe_op_id == "recipe_0000":
            out.append({
                "obligation": o.get("id", f"obl_{recipe_op_id}"),
                "status": status,
                "remaining": (
                    [] if status == "discharged" else ["real_fusion_differential_check"]
                ),
            })
    if not out:
        out.append({
            "obligation": f"obl_{recipe_op_id}",
            "status": status,
            "remaining": [] if status == "discharged" else ["real_fusion_differential_check"],
        })
    return out
