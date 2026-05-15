"""Runtime plan assertions.

Phase C generate the body of ``assert_plan(io)`` from each
region's contract + the binding. Every assertion fires
with a named ``PLAN_VIOLATION_<KIND>`` typed error so a binary that
runs on inputs the plan didn't promise refuses to run with a
specific, auditable reason.

Generated kinds (one per contract field with a runtime-checkable
invariant):

::

    PLAN_VIOLATION_INPUT_COUNT     mismatched number of io tensors vs contract inputs
    PLAN_VIOLATION_INPUT_DTYPE     io tensor dtype not in contract.dtype_class
    PLAN_VIOLATION_INPUT_SHAPE     io tensor shape != contract dims
    PLAN_VIOLATION_INPUT_BYTES     io tensor numel * element_size != expected bytes
    PLAN_VIOLATION_LAYOUT          io tensor layout not row_major when contract says row_major
    PLAN_VIOLATION_BUFFER_SIZE allocated buffer size != planned size (wires)
    PLAN_VIOLATION_EVENT_WRITERS event has more than wait_count writers (wires)
    PLAN_VIOLATION_UNBOUND_REGION region declared in plan but no certified kernel (carryover)

The emitter renders straight-line Python that Section 4 Dream 4
describes:

::

    assert io["A"].numel() * io["A"].element_size() == 1024  # PLAN_VIOLATION_INPUT_BYTES

The IO order convention at is positional: the values of ``io``
in insertion order map to the contract's ``io.inputs[]`` in order.
wires named-binding (operator passes a dict that's resolved
against contract input names).

Each assertion path in the emitted source is reachable via a
negative-control test, so the invariant ("the runtime stops
loud, not silently") has real fault-injection coverage.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


_DTYPE_BYTES: dict[str, int] = {
    "f64": 8, "fp64": 8, "float64": 8,
    "f32": 4, "fp32": 4, "float32": 4,
    "f16": 2, "fp16": 2, "float16": 2,
    "bf16": 2, "bfloat16": 2,
    "i64": 8, "int64": 8,
    "i32": 4, "int32": 4,
    "i16": 2, "int16": 2,
    "i8": 1, "int8": 1,
    "u8": 1, "uint8": 1,
}


@dataclass(frozen=True)
class _RegionAssertions:
    region_id: str
    contract_hash: str
    inputs: list[dict[str, Any]]   # name, dims, dtype_class, expected_bytes, layout
    outputs: list[dict[str, Any]]  # same
    accumulator_dtype: str
    aliasing: list[dict[str, int]]  # input_idx, output_idx
    in_place_safe: bool
    event_decls: list[dict[str, Any]]  # name, wait_count
    # typed predicates lifted from the contract.
    preconditions: list[dict[str, Any]] = field(default_factory=list)
    postconditions: list[dict[str, Any]] = field(default_factory=list)


def _bytes_for(dims: list[int | None], dtype_class: list[str]) -> int:
    """Conservative estimate of expected tensor bytes — uses the
    first dtype in dtype_class that has a known width. Returns 0
    when shape contains None (dynamic) or dtype is unknown.
    """
    if not dims or any(d is None or d <= 0 for d in dims):
        return 0
    elem_bytes = 0
    for dt in dtype_class:
        if dt in _DTYPE_BYTES:
            elem_bytes = _DTYPE_BYTES[dt]
            break
    if elem_bytes == 0:
        return 0
    n = 1
    for d in dims:
        n *= int(d)
    return n * elem_bytes


def _region_assertions_from_contract(
    *, region_id: str, contract: dict[str, Any],
) -> _RegionAssertions:
    io = contract.get("io") or {}
    inputs = []
    for t in io.get("inputs") or []:
        dims = list((t.get("shape") or {}).get("dims") or [])
        dtype_class = list(t.get("dtype_class") or [])
        inputs.append({
            "name": t.get("name", ""),
            "dims": dims,
            "dtype_class": dtype_class,
            "layout": t.get("layout", "row_major"),
            "expected_bytes": _bytes_for(dims, dtype_class),
        })
    outputs = []
    for t in io.get("outputs") or []:
        dims = list((t.get("shape") or {}).get("dims") or [])
        dtype_class = list(t.get("dtype_class") or [])
        outputs.append({
            "name": t.get("name", ""),
            "dims": dims,
            "dtype_class": dtype_class,
            "layout": t.get("layout", "row_major"),
            "expected_bytes": _bytes_for(dims, dtype_class),
        })
    numerics = io.get("numerics") or {}
    orch = contract.get("orchestration") or {}
    sync = orch.get("sync") or {}
    memory = orch.get("memory") or {}
    aliasing = [
        {"input_idx": a.get("input_idx", 0), "output_idx": a.get("output_idx", 0)}
        for a in (sync.get("aliasing") or [])
    ]
    event_decls = [
        {"name": e.get("name", ""), "wait_count": int(e.get("wait_count", 1))}
        for e in (sync.get("event_decls") or [])
    ]
    preconditions = list(contract.get("preconditions") or [])
    postconditions = list(contract.get("postconditions") or [])
    return _RegionAssertions(
        region_id=region_id,
        contract_hash="",  # filled by caller
        inputs=inputs,
        outputs=outputs,
        accumulator_dtype=str(numerics.get("accumulator_dtype") or ""),
        aliasing=aliasing,
        in_place_safe=bool(memory.get("in_place_safe", False)),
        event_decls=event_decls,
        preconditions=preconditions,
        postconditions=postconditions,
    )


def collect_region_assertions(
    *,
    run_dir: Path,
    bindings: list[dict[str, Any]],
) -> list[_RegionAssertions]:
    """Read each binding's contract file and produce the assertion
    tuple. ``bindings`` is the list of dicts from
    ``05_execution_plan/region_kernel_bindings.json``.
    """
    rows: list[_RegionAssertions] = []
    for b in bindings:
        if b.get("status") != "bound":
            continue
        # Look up the contract file by walking
        # 04_kernel_codegen/contracts/<region_id>.<hash>.json.
        contracts_dir = run_dir / "04_kernel_codegen" / "contracts"
        contract_hash = b["contract_hash"]
        candidate = contracts_dir / f"{b['region_id']}.{contract_hash}.json"
        if not candidate.exists():
            continue
        body = json.loads(candidate.read_text(encoding="utf-8"))
        ra = _region_assertions_from_contract(
            region_id=b["region_id"], contract=body,
        )
        rows.append(_RegionAssertions(
            region_id=ra.region_id,
            contract_hash=contract_hash,
            inputs=ra.inputs,
            outputs=ra.outputs,
            accumulator_dtype=ra.accumulator_dtype,
            aliasing=ra.aliasing,
            in_place_safe=ra.in_place_safe,
            event_decls=ra.event_decls,
            preconditions=ra.preconditions,
            postconditions=ra.postconditions,
        ))
    return rows


def render_assert_plan_body(
    region_assertions: list[_RegionAssertions],
) -> str:
    """Render the executable Python body for ``assert_plan(io)``.

    The body raises typed ``PlanViolation`` subclasses defined in the
    emitted module. Every check has a corresponding negative-control
    test in tests/runtime/test_plan_assertions.py.
    """
    if not region_assertions:
        return "    # M-48: no bound regions; nothing to check beyond M-46 unbound check.\n    return\n"

    lines: list[str] = []
    lines.append("    # M-48 generated assertions per contract field.")
    lines.append("    # Each raises a typed PLAN_VIOLATION_<KIND> on failure.")
    lines.append("")

    # We embed ALL region assertions into a single positional pass.
    # IO dict values in insertion order map to inputs across regions
    # in placement order. will wire named binding; today's
    # assertions are positional.
    flat_inputs: list[tuple[str, dict[str, Any]]] = []
    for ra in region_assertions:
        for inp in ra.inputs:
            flat_inputs.append((ra.region_id, inp))

    lines.append("    # Total inputs across bound regions (positional order).")
    lines.append(f"    expected_input_count = {len(flat_inputs)}")
    lines.append("    actual_inputs = list(io.values()) if isinstance(io, dict) else list(io)")
    lines.append("    if len(actual_inputs) < expected_input_count:")
    lines.append("        raise PLAN_VIOLATION_INPUT_COUNT(")
    lines.append("            f\"expected at least {expected_input_count} inputs, \"")
    lines.append("            f\"got {len(actual_inputs)}\"")
    lines.append("        )")
    lines.append("")

    for idx, (region_id, inp) in enumerate(flat_inputs):
        lines.append(f"    # region={region_id!r} input[{idx}] name={inp['name']!r}")
        lines.append(f"    _t = actual_inputs[{idx}]")
        # Dtype check.
        dtypes = inp["dtype_class"]
        if dtypes:
            lines.append(
                f"    _allowed_dtypes = {dtypes!r}"
            )
            lines.append(
                "    _t_dtype = _normalise_dtype(getattr(_t, 'dtype', None))"
            )
            lines.append(
                "    if _t_dtype not in _allowed_dtypes:"
            )
            lines.append(
                f"        raise PLAN_VIOLATION_INPUT_DTYPE("
            )
            lines.append(
                f"            f\"region={region_id!r} input[{idx}]: dtype \""
            )
            lines.append(
                "            f\"{_t_dtype!r} not in {_allowed_dtypes!r}\""
            )
            lines.append(
                "        )"
            )
        # Shape check.
        dims = inp["dims"]
        if dims and all(d is not None and d > 0 for d in dims):
            lines.append(
                f"    _expected_shape = {tuple(dims)!r}"
            )
            lines.append(
                "    _actual_shape = tuple(getattr(_t, 'shape', ())) "
                "if hasattr(_t, 'shape') else None"
            )
            lines.append(
                "    if _actual_shape != _expected_shape:"
            )
            lines.append(
                f"        raise PLAN_VIOLATION_INPUT_SHAPE("
            )
            lines.append(
                f"            f\"region={region_id!r} input[{idx}]: shape \""
            )
            lines.append(
                "            f\"{_actual_shape!r} != {_expected_shape!r}\""
            )
            lines.append(
                "        )"
            )
        # Bytes check (only when both dims and dtype are concrete).
        if inp["expected_bytes"] > 0:
            lines.append(
                f"    _expected_bytes = {inp['expected_bytes']}"
            )
            lines.append(
                "    _actual_bytes = ("
                "_t.numel() * _t.element_size()"
                " if hasattr(_t, 'numel') and hasattr(_t, 'element_size')"
                " else None)"
            )
            lines.append(
                "    if _actual_bytes is not None and _actual_bytes != _expected_bytes:"
            )
            lines.append(
                f"        raise PLAN_VIOLATION_INPUT_BYTES("
            )
            lines.append(
                f"            f\"region={region_id!r} input[{idx}]: bytes \""
            )
            lines.append(
                "            f\"{_actual_bytes} != {_expected_bytes}\""
            )
            lines.append(
                "        )"
            )
        # Layout check.
        if inp.get("layout"):
            lines.append(
                f"    _expected_layout = {inp['layout']!r}"
            )
            lines.append(
                "    _is_contiguous = ("
                "_t.is_contiguous() if hasattr(_t, 'is_contiguous')"
                " else True)"
            )
            lines.append(
                "    if _expected_layout == 'row_major' and not _is_contiguous:"
            )
            lines.append(
                f"        raise PLAN_VIOLATION_LAYOUT("
            )
            lines.append(
                f"            f\"region={region_id!r} input[{idx}]: contract \""
            )
            lines.append(
                f"            f\"declares row_major but tensor is non-contiguous\""
            )
            lines.append(
                "        )"
            )
        lines.append("")

    # Event-writer check (static): every event_decl across regions
    # must be UNIQUE in name across regions; no event has multiple
    # producers.
    seen_events: dict[str, str] = {}
    for ra in region_assertions:
        for e in ra.event_decls:
            name = e["name"]
            if name in seen_events:
                lines.append(
                    f"    raise PLAN_VIOLATION_EVENT_WRITERS("
                )
                lines.append(
                    f"        \"event {name!r} declared by both regions \""
                )
                lines.append(
                    f"        \"{seen_events[name]!r} and {ra.region_id!r}; \""
                )
                lines.append(
                    f"        \"only one writer per event allowed\""
                )
                lines.append(
                    "    )"
                )
            seen_events[name] = ra.region_id

    # predicate-driven precondition checks. Currently emitted
    # for ModEq (the load-bearing one — block-size constraint) +
    # ByteSizeLe + DtypeIn. NoAlias and NumericalWithinEps are
    # postcondition-bound and surface in the verifier rather
    # than the runtime fast-path.
    flat_in_idx_for_region: dict[str, list[int]] = {}
    seen_idx = 0
    for ra in region_assertions:
        flat_in_idx_for_region[ra.region_id] = []
        for _ in ra.inputs:
            flat_in_idx_for_region[ra.region_id].append(seen_idx)
            seen_idx += 1

    for ra in region_assertions:
        for pred in ra.preconditions:
            kind = pred.get("kind", "")
            if kind == "mod_eq":
                arg_dim = pred.get("arg_dim", "")
                k = int(pred.get("k", 0))
                if k <= 0 or not arg_dim:
                    continue
                # Resolve arg_dim: convention is a single uppercase
                # letter naming the contract-level dim (e.g. "K"). For
                # matmul, K is dims[1] of inputs[0] AND dims[0] of
                # inputs[1]; we check the first input's K dim by
                # convention (the kernel's inner-loop bound).
                indices = flat_in_idx_for_region.get(ra.region_id, [])
                if not indices:
                    continue
                check_idx = indices[0]
                # K-dim is the LAST dim of input[0] for matmul.
                lines.append(
                    f"    # M-61 precondition: {arg_dim} mod {k} == 0"
                )
                lines.append(
                    f"    _t = actual_inputs[{check_idx}]"
                )
                lines.append(
                    "    _t_shape = tuple(getattr(_t, 'shape', ()))"
                )
                lines.append(
                    f"    if _t_shape and _t_shape[-1] % {k} != 0:"
                )
                lines.append(
                    f"        raise PLAN_VIOLATION_PRECONDITION_MOD_EQ("
                )
                lines.append(
                    f"            f\"region={ra.region_id!r}: precondition \""
                )
                lines.append(
                    f"            f\"{arg_dim} mod {k} == 0 violated; got \""
                )
                lines.append(
                    f"            f\"{arg_dim}={{_t_shape[-1]}}\""
                )
                lines.append(
                    "        )"
                )
                lines.append("")
            elif kind == "dtype_in":
                #  closure: emit a typed DtypeIn precondition
                # check independent of the structural INPUT_DTYPE
                # check. The structural check enforces dtype_class
                # membership; this predicate enforces a (possibly
                # narrower) explicit allowlist supplied by the
                # contract author.
                arg = pred.get("arg", "")
                dtype_set = list(pred.get("dtype_set") or [])
                if not arg or not dtype_set:
                    continue
                target_idx: int | None = None
                for j, inp in enumerate(ra.inputs):
                    if inp.get("name") == arg:
                        target_idx = flat_in_idx_for_region[ra.region_id][j]
                        break
                if target_idx is None:
                    continue
                lines.append(
                    f"    # M-61 precondition: {arg} dtype in {dtype_set!r}"
                )
                lines.append(
                    f"    _t = actual_inputs[{target_idx}]"
                )
                lines.append(
                    "    _t_dtype = _normalise_dtype(getattr(_t, 'dtype', None))"
                )
                # Normalise the allowlist entries the same way the
                # runtime's _normalise_dtype maps torch dtypes (fp32 →
                # f32, fp16 → f16, etc.).
                _dtype_alias = {
                    "fp32": "f32", "fp16": "f16",
                    "fp64": "f64", "float32": "f32",
                    "float16": "f16", "float64": "f64",
                    "bfloat16": "bf16",
                }
                normalised_set = sorted({
                    _dtype_alias.get(str(d), str(d)) for d in dtype_set
                })
                lines.append(
                    f"    _allowed_dtypes = {normalised_set!r}"
                )
                lines.append(
                    "    if _t_dtype not in _allowed_dtypes:"
                )
                lines.append(
                    f"        raise PLAN_VIOLATION_PRECONDITION_DTYPE_IN("
                )
                lines.append(
                    f"            f\"region={ra.region_id!r}: precondition \""
                )
                lines.append(
                    f"            f\"dtype_in({arg!r}, {dtype_set!r}) violated; \""
                )
                lines.append(
                    "            f\"got {_t_dtype!r}\""
                )
                lines.append(
                    "        )"
                )
                lines.append("")
            elif kind == "byte_size_le":
                arg = pred.get("arg", "")
                max_bytes = int(pred.get("max_bytes", 0))
                if not arg or max_bytes <= 0:
                    continue
                # Find the input index matching arg name.
                target_idx: int | None = None
                for j, inp in enumerate(ra.inputs):
                    if inp.get("name") == arg:
                        target_idx = flat_in_idx_for_region[ra.region_id][j]
                        break
                if target_idx is None:
                    continue
                lines.append(
                    f"    # M-61 precondition: {arg} numel*element_size <= {max_bytes}"
                )
                lines.append(
                    f"    _t = actual_inputs[{target_idx}]"
                )
                lines.append(
                    "    _t_bytes = _t.numel() * _t.element_size()"
                )
                lines.append(
                    f"    if _t_bytes > {max_bytes}:"
                )
                lines.append(
                    f"        raise PLAN_VIOLATION_PRECONDITION_BYTE_SIZE_LE("
                )
                lines.append(
                    f"            f\"region={ra.region_id!r}: precondition \""
                )
                lines.append(
                    f"            f\"{arg} bytes <= {max_bytes} violated; got \""
                )
                lines.append(
                    "            f\"bytes={_t_bytes}\""
                )
                lines.append(
                    "        )"
                )
                lines.append("")

    return "\n".join(lines) + "\n"


def render_assert_postconditions_body(
    region_assertions: list[_RegionAssertions],
) -> str:
    """ closure: render the body of
    ``assert_postconditions(outputs, refs)`` — fires postcondition
    predicates (NumericalWithinEps) against a caller-supplied
    reference dict.

    The caller passes ``outputs`` (the kernel's actual outputs) and
    ``refs`` (a dict from output-name → expected tensor or callable).
    For each :class:`NumericalWithinEps` postcondition on a region,
    we compute max relative error vs the reference and raise the
    typed ``PLAN_VIOLATION_POSTCONDITION_NUMERICAL_WITHIN_EPS`` when
    it exceeds ``eps``.

    When ``refs`` is None or doesn't contain a reference for an
    output, the postcondition is logged as skipped — the caller
    didn't provide ground truth to check against.
    """
    if not region_assertions or not any(
        ra.postconditions for ra in region_assertions
    ):
        return (
            "    # M-61 postconditions: none declared; no-op.\n"
            "    return\n"
        )

    lines: list[str] = [
        "    # M-61 postcondition checks (gap #11).",
        "    if refs is None:",
        "        return",
    ]
    for ra in region_assertions:
        for pc in ra.postconditions:
            kind = pc.get("kind", "")
            if kind != "numerical_within_eps":
                continue
            out = pc.get("out", "")
            ref = pc.get("ref", "reference")
            eps = float(pc.get("eps", 0.0))
            if not out:
                continue
            lines.append(
                f"    # region={ra.region_id!r}: "
                f"numerical_within_eps({out!r}, ref={ref!r}, eps={eps})"
            )
            lines.append(f"    if {out!r} in outputs and {out!r} in refs:")
            lines.append(f"        _y = outputs[{out!r}]")
            lines.append(f"        _r = refs[{out!r}]")
            lines.append("        # Best-effort numerical diff. Tensors with")
            lines.append("        # a .float() method ride through PyTorch's")
            lines.append("        # ops; plain numpy / floats fall back to")
            lines.append("        # max abs / max rel manually.")
            lines.append("        try:")
            lines.append("            _delta = (_y - _r).abs()")
            lines.append("            _max_rel = float(")
            lines.append("                (_delta / (_r.abs() + 1e-12)).max()")
            lines.append("            )")
            lines.append("        except Exception:")
            lines.append("            _max_rel = float('inf')")
            lines.append(f"        if _max_rel > {eps}:")
            lines.append(
                f"            raise PLAN_VIOLATION_POSTCONDITION_NUMERICAL_WITHIN_EPS("
            )
            lines.append(
                f"                f\"region={ra.region_id!r}: postcondition \""
            )
            lines.append(
                f"                f\"numerical_within_eps({out!r}, "
                f"ref={ref!r}, eps={eps}) \""
            )
            lines.append("                f\"violated; max_rel={_max_rel:.3e}\"")
            lines.append("            )")
    lines.append("    return")
    return "\n".join(lines) + "\n"


def render_plan_violation_classes() -> str:
    """Render the ``PLAN_VIOLATION_<KIND>`` typed-error classes
    embedded in the generated executor module."""
    kinds = [
        "INPUT_COUNT", "INPUT_DTYPE", "INPUT_SHAPE", "INPUT_BYTES",
        "LAYOUT", "BUFFER_SIZE", "EVENT_WRITERS", "UNBOUND_REGION",
        "IO_TYPE",
        # predicate-driven precondition / postcondition kinds.
        "PRECONDITION_MOD_EQ",
        "PRECONDITION_BYTE_SIZE_LE",
        "PRECONDITION_NO_ALIAS",
        "PRECONDITION_DTYPE_IN",
        "POSTCONDITION_NUMERICAL_WITHIN_EPS",
    ]
    lines: list[str] = [
        "class PlanViolation(RuntimeError):",
        "    \"\"\"Runtime plan-invariant violation. Subclasses are\n"
        "    typed by which check fired (M-48).\"\"\"",
        "",
    ]
    for k in kinds:
        lines.append(f"class PLAN_VIOLATION_{k}(PlanViolation):")
        lines.append(f"    pass")
        lines.append("")
    # Helper for normalising torch dtypes to contract dtype-class strings.
    lines.append("def _normalise_dtype(dt: Any) -> str:")
    lines.append("    \"\"\"Map a torch dtype (or any object) to a contract \"")
    lines.append("    dtype_class string the contract uses.\"\"\"")
    lines.append("    if dt is None:")
    lines.append("        return ''")
    lines.append("    s = str(dt)")
    lines.append("    if s.endswith('float32') or s.endswith('torch.float32'):")
    lines.append("        return 'f32'")
    lines.append("    if s.endswith('float16') or s.endswith('torch.float16'):")
    lines.append("        return 'f16'")
    lines.append("    if s.endswith('bfloat16') or s.endswith('torch.bfloat16'):")
    lines.append("        return 'bf16'")
    lines.append("    if s.endswith('float64') or s.endswith('torch.float64'):")
    lines.append("        return 'f64'")
    lines.append("    if s.endswith('int64') or s.endswith('torch.int64'):")
    lines.append("        return 'i64'")
    lines.append("    if s.endswith('int32') or s.endswith('torch.int32'):")
    lines.append("        return 'i32'")
    lines.append("    return s")
    return "\n".join(lines) + "\n"
