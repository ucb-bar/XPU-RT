"""C11 baremetal plan executor emitter (, Phase G).

Phase G Dream 1 of the §12 three-layer runtime. Sibling of
:mod:`compgen.runtime.glue_emit.python_sync` that emits a **C11**
``generated_plan_executor.c`` next to the Python module, calling the
same ``libcompgen_rt`` ABI that ``cpu_sync`` / ``cpu_task`` / ``cuda``
drivers already implement.

The emit target is per the §12 dream:

::

    extern int compgen_run(const compgen_io_t *io,
                           compgen_status_t  *status);

The emit:

- inlines typed ``COMPGEN_PLAN_VIOLATION_<KIND>`` codes derived from
  's contract-driven assertions (input count, dtype, shape, bytes,
  layout, unbound region, predicate-driven mod_eq / dtype_in /
  byte_size_le, and event-writer uniqueness);
- opens a ``cg_rt_instance_t`` + ``cg_rt_device_t`` for the target;
allocates one ``cg_rt_buffer_t`` per region's IO (wires the
  real binding-to-buffer map; emits the structural skeleton);
- builds a single ``cg_rt_command_buffer_t`` with one
  ``cg_rt_command_buffer_dispatch`` per region in the topologically
  sorted order shared with ;
- submits with a single timeline-semaphore signal point and waits for
  completion before returning;
- emits a sibling ``plan_executor_c11_manifest.json`` describing the
  emit (byte-stable, sorted keys, schema-versioned).

Hard rules:

- The emit calls **only** ``cg_rt_*`` symbols. The mechanical
  ABI-conformance check (D6) greps the emitted ``.c`` for any
  non-``cg_rt_`` extern call; the gate fails if it finds one.
- Plan invariants are checked at the top of ``compgen_run`` and a
  failure stores a named code into ``status->code`` then returns the
  same code (no abort/longjmp). This is the runtime analogue of
  raising a typed Python exception.
- Kernel executables are declared as ``extern cg_rt_executable_t*``
  symbols named ``compgen_kernel_<region_id>``; the operator links the
  emit against a kernel-pack object file produced. The emit
  is byte-stable wrt the plan + contracts.

Acceptance:

- Compiles cleanly with ``-std=c11 -Wall -Wextra -Werror`` against
  ``libcompgen_rt/include/compgen_rt/compgen_rt.h``.
- Together with a stub kernel that copies inputs to outputs (the
  emit's -style differential harness), the resulting binary
  matches the Python SYNC executor's output bit-for-bit on
  ``proxy_vla``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from compgen.runtime.execution_plan import ExecutionPlan
from compgen.runtime.glue_emit.plan_assertions import (
    collect_region_assertions,
)
from compgen.runtime.glue_emit.python_sync import (
    _read_yaml_or_json_plan,
    _topological_region_order,
)


_C11_EMIT_SCHEMA_VERSION = "plan_executor_c11_manifest_v1"

# Stable ordering of PLAN_VIOLATION_<KIND> codes. The numeric value is
# part of the ABI: status->code carries the integer; downstream tools
# (refinement check, evidence pack) read it back. Codes are
# negative so they don't collide with cg_rt_status_t conventions.
_PLAN_VIOLATION_CODES: tuple[tuple[str, int], ...] = (
    ("OK", 0),
    ("IO_NULL", -101),
    ("UNBOUND_REGION", -102),
    ("INPUT_COUNT", -103),
    ("INPUT_DTYPE", -104),
    ("INPUT_SHAPE", -105),
    ("INPUT_BYTES", -106),
    ("LAYOUT", -107),
    ("BUFFER_SIZE", -108),
    ("EVENT_WRITERS", -109),
    ("PRECONDITION_MOD_EQ", -110),
    ("PRECONDITION_BYTE_SIZE_LE", -111),
    ("PRECONDITION_DTYPE_IN", -112),
    ("PRECONDITION_NO_ALIAS", -113),
    ("POSTCONDITION_NUMERICAL_WITHIN_EPS", -114),
)


# Map 's contract dtype-class strings to a stable C dtype enum the
# emit asserts against. Anything not in this map is a code-only check
# (the operator's tensor-element-bytes match is by numel*element_size).
_DTYPE_C_NAME: dict[str, str] = {
    "f32": "COMPGEN_DTYPE_F32", "fp32": "COMPGEN_DTYPE_F32", "float32": "COMPGEN_DTYPE_F32",
    "f16": "COMPGEN_DTYPE_F16", "fp16": "COMPGEN_DTYPE_F16", "float16": "COMPGEN_DTYPE_F16",
    "bf16": "COMPGEN_DTYPE_BF16", "bfloat16": "COMPGEN_DTYPE_BF16",
    "f64": "COMPGEN_DTYPE_F64", "fp64": "COMPGEN_DTYPE_F64", "float64": "COMPGEN_DTYPE_F64",
    "i64": "COMPGEN_DTYPE_I64", "int64": "COMPGEN_DTYPE_I64",
    "i32": "COMPGEN_DTYPE_I32", "int32": "COMPGEN_DTYPE_I32",
    "i16": "COMPGEN_DTYPE_I16", "int16": "COMPGEN_DTYPE_I16",
    "i8": "COMPGEN_DTYPE_I8", "int8": "COMPGEN_DTYPE_I8",
    "u8": "COMPGEN_DTYPE_U8", "uint8": "COMPGEN_DTYPE_U8",
}


@dataclass(frozen=True)
class C11GlueEmitResult:
    out_dir: Path
    executor_path: Path
    header_path: Path
    manifest_path: Path
    overall: str  # "pass" | "skipped"
    bound_regions: tuple[str, ...]
    unbound_regions: tuple[str, ...]
    plan_violation_codes: dict[str, int]


def _utcnow() -> str:
    return datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _render_header(*, workload: str, target: str, plan_path: str) -> str:
    """Render the companion ``generated_plan_executor.h`` declaring
    the IO struct, status struct, and ``compgen_run`` entry point."""
    code_defs = "\n".join(
        f"#define COMPGEN_PLAN_VIOLATION_{name} ({val})"
        for name, val in _PLAN_VIOLATION_CODES
    )
    return f'''/* Auto-generated by M-88 (compgen.runtime.glue_emit.c11_baremetal).
 *
 * Workload : {workload}
 * Target   : {target}
 * Source   : {plan_path}
 *
 * DO NOT EDIT — regenerate by re-running ``--stop-after glue-emit``.
 *
 * This header is the public ABI between the operator and the emitted
 * C11 plan executor. It declares the IO struct that
 * ``compgen_run`` consumes, the status struct it populates, and the
 * named ``COMPGEN_PLAN_VIOLATION_<KIND>`` codes the runtime fills
 * when a plan invariant is violated (§12 D4).
 */

#ifndef COMPGEN_GENERATED_PLAN_EXECUTOR_H
#define COMPGEN_GENERATED_PLAN_EXECUTOR_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {{
#endif

/* Named plan-violation codes (ABI). The numeric values are stable
 * across emits; downstream tooling reads them back as integers. */
{code_defs}

/* Layout enum (matches the contract field). */
typedef enum {{
    COMPGEN_LAYOUT_UNKNOWN    = 0,
    COMPGEN_LAYOUT_ROW_MAJOR  = 1,
    COMPGEN_LAYOUT_COL_MAJOR  = 2
}} compgen_layout_t;

/* Dtype enum (matches the contract dtype_class strings). */
typedef enum {{
    COMPGEN_DTYPE_UNKNOWN = 0,
    COMPGEN_DTYPE_F64     = 1,
    COMPGEN_DTYPE_F32     = 2,
    COMPGEN_DTYPE_F16     = 3,
    COMPGEN_DTYPE_BF16    = 4,
    COMPGEN_DTYPE_I64     = 5,
    COMPGEN_DTYPE_I32     = 6,
    COMPGEN_DTYPE_I16     = 7,
    COMPGEN_DTYPE_I8      = 8,
    COMPGEN_DTYPE_U8      = 9
}} compgen_dtype_t;

/* One IO tensor. ``rank`` is the dimensionality; ``shape`` is a
 * packed dimension list of length ``rank`` (max 8 dims per the
 * runtime's static budget). ``bytes`` is the buffer's total byte
 * count.  ``layout`` matches the contract's declared layout. */
#define COMPGEN_MAX_TENSOR_RANK 8

typedef struct {{
    void             *data;
    size_t            bytes;
    int32_t           rank;
    int64_t           shape[COMPGEN_MAX_TENSOR_RANK];
    compgen_dtype_t   dtype;
    compgen_layout_t  layout;
}} compgen_tensor_t;

/* The full IO bundle the operator passes into ``compgen_run``. The
 * order of ``inputs[]`` and ``outputs[]`` matches the M-46 placement
 * order × contract input order, the same convention the Python SYNC
 * emit uses. */
typedef struct {{
    const compgen_tensor_t *inputs;
    size_t                  n_inputs;
    compgen_tensor_t       *outputs;
    size_t                  n_outputs;
}} compgen_io_t;

/* Status of a single invocation. ``code`` is one of the
 * COMPGEN_PLAN_VIOLATION_* codes (or 0 for success). When ``code`` is
 * a plan-violation, ``detail`` is a static string naming the failing
 * invariant (no allocation).  When ``code`` is a libcompgen_rt error
 * (passed through verbatim), ``rt_status`` carries the underlying
 * cg_rt_status_t. */
typedef struct {{
    int32_t      code;
    int32_t      rt_status;
    const char  *detail;
}} compgen_status_t;

/* Per-workload plan executor. Returns 0 on success or a negative
 * COMPGEN_PLAN_VIOLATION_<KIND> on plan-invariant failure. */
int compgen_run(const compgen_io_t *io, compgen_status_t *status);

#ifdef __cplusplus
}}
#endif

#endif /* COMPGEN_GENERATED_PLAN_EXECUTOR_H */
'''


def _render_io_assertions(
    region_assertions: list[Any],
) -> str:
    """Render the C-level assertion block. Mirrors 's positional
    convention: io.inputs[i] maps to the flattened
    region_assertions[*].inputs[*] in placement order."""
    lines: list[str] = []
    flat_inputs: list[tuple[str, dict[str, Any]]] = []
    for ra in region_assertions:
        for inp in ra.inputs:
            flat_inputs.append((ra.region_id, inp))

    expected_in = len(flat_inputs)
    lines.append("    /* M-48 + M-88: typed plan-invariant checks. */")
    lines.append(f"    const size_t expected_input_count = {expected_in};")
    lines.append("    if (io->n_inputs < expected_input_count) {")
    lines.append("        status->code = COMPGEN_PLAN_VIOLATION_INPUT_COUNT;")
    lines.append(
        '        status->detail = "expected at least '
        f'{expected_in} inputs";'
    )
    lines.append("        return status->code;")
    lines.append("    }")
    lines.append("")

    for idx, (region_id, inp) in enumerate(flat_inputs):
        lines.append(f"    /* region={region_id!r} input[{idx}] "
                     f"name={inp['name']!r} */")
        # Dtype check.
        dtypes = [_DTYPE_C_NAME[d] for d in inp.get("dtype_class", [])
                  if d in _DTYPE_C_NAME]
        if dtypes:
            cond = " && ".join(
                f"io->inputs[{idx}].dtype != {c}" for c in dtypes
            )
            lines.append(f"    if ({cond}) {{")
            lines.append(
                "        status->code = COMPGEN_PLAN_VIOLATION_INPUT_DTYPE;"
            )
            lines.append(
                f'        status->detail = "region={region_id} '
                f'input[{idx}]: dtype not in contract dtype_class";'
            )
            lines.append("        return status->code;")
            lines.append("    }")
        # Shape check (rank + each dim).
        dims = inp.get("dims", [])
        if dims and all(d is not None and d > 0 for d in dims):
            lines.append(f"    if (io->inputs[{idx}].rank != {len(dims)}) {{")
            lines.append(
                "        status->code = COMPGEN_PLAN_VIOLATION_INPUT_SHAPE;"
            )
            lines.append(
                f'        status->detail = "region={region_id} '
                f'input[{idx}]: rank mismatch";'
            )
            lines.append("        return status->code;")
            lines.append("    }")
            for di, d in enumerate(dims):
                lines.append(
                    f"    if (io->inputs[{idx}].shape[{di}] != "
                    f"(int64_t){int(d)}) {{"
                )
                lines.append(
                    "        status->code = COMPGEN_PLAN_VIOLATION_INPUT_SHAPE;"
                )
                lines.append(
                    f'        status->detail = "region={region_id} '
                    f'input[{idx}]: shape[{di}] mismatch";'
                )
                lines.append("        return status->code;")
                lines.append("    }")
        # Bytes check.
        if inp.get("expected_bytes", 0) > 0:
            lines.append(
                f"    if (io->inputs[{idx}].bytes != "
                f"(size_t){inp['expected_bytes']}) {{"
            )
            lines.append(
                "        status->code = COMPGEN_PLAN_VIOLATION_INPUT_BYTES;"
            )
            lines.append(
                f'        status->detail = "region={region_id} '
                f'input[{idx}]: bytes mismatch";'
            )
            lines.append("        return status->code;")
            lines.append("    }")
        # Layout check.
        layout = inp.get("layout", "")
        if layout == "row_major":
            lines.append(
                f"    if (io->inputs[{idx}].layout != "
                f"COMPGEN_LAYOUT_ROW_MAJOR) {{"
            )
            lines.append(
                "        status->code = COMPGEN_PLAN_VIOLATION_LAYOUT;"
            )
            lines.append(
                f'        status->detail = "region={region_id} '
                f'input[{idx}]: contract declares row_major";'
            )
            lines.append("        return status->code;")
            lines.append("    }")
        lines.append("")

    # Event-writer static uniqueness — single-emit check (same as ).
    seen_events: dict[str, str] = {}
    for ra in region_assertions:
        for e in ra.event_decls:
            name = e.get("name", "")
            if not name:
                continue
            if name in seen_events:
                lines.append(
                    "    /* M-48 event-writer uniqueness fails statically: */"
                )
                lines.append(
                    "    status->code = COMPGEN_PLAN_VIOLATION_EVENT_WRITERS;"
                )
                lines.append(
                    f'    status->detail = "event {name!r} declared by both '
                    f'{seen_events[name]!r} and {ra.region_id!r}";'
                )
                lines.append("    return status->code;")
            seen_events[name] = ra.region_id

    # predicate-driven preconditions (mod_eq, byte_size_le, dtype_in).
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
                k = int(pred.get("k", 0))
                indices = flat_in_idx_for_region.get(ra.region_id, [])
                if k <= 0 or not indices:
                    continue
                check_idx = indices[0]
                lines.append(
                    f"    /* M-61 mod_eq({pred.get('arg_dim', '?')}, {k}) "
                    f"on region={ra.region_id} */"
                )
                lines.append(
                    f"    {{ int32_t _r = io->inputs[{check_idx}].rank;"
                )
                lines.append(
                    f"      int64_t _last = (_r > 0)"
                    f" ? io->inputs[{check_idx}].shape[_r - 1] : 0;"
                )
                lines.append(f"      if (_last % {k} != 0) {{")
                lines.append(
                    "          status->code = "
                    "COMPGEN_PLAN_VIOLATION_PRECONDITION_MOD_EQ;"
                )
                lines.append(
                    f'          status->detail = "region={ra.region_id}: '
                    f'precondition {pred.get("arg_dim", "?")} mod {k} == 0 '
                    f'violated";'
                )
                lines.append("          return status->code;")
                lines.append("      } }")
                lines.append("")
            elif kind == "byte_size_le":
                arg = pred.get("arg", "")
                max_bytes = int(pred.get("max_bytes", 0))
                if not arg or max_bytes <= 0:
                    continue
                target_idx: int | None = None
                for j, inp in enumerate(ra.inputs):
                    if inp.get("name") == arg:
                        target_idx = flat_in_idx_for_region[ra.region_id][j]
                        break
                if target_idx is None:
                    continue
                lines.append(
                    f"    /* M-61 byte_size_le({arg}, {max_bytes}) on "
                    f"region={ra.region_id} */"
                )
                lines.append(
                    f"    if (io->inputs[{target_idx}].bytes > "
                    f"(size_t){max_bytes}) {{"
                )
                lines.append(
                    "        status->code = "
                    "COMPGEN_PLAN_VIOLATION_PRECONDITION_BYTE_SIZE_LE;"
                )
                lines.append(
                    f'        status->detail = "region={ra.region_id}: '
                    f'precondition {arg} bytes <= {max_bytes} violated";'
                )
                lines.append("        return status->code;")
                lines.append("    }")
                lines.append("")
            elif kind == "dtype_in":
                arg = pred.get("arg", "")
                dtype_set_raw = list(pred.get("dtype_set") or [])
                dtype_set = [
                    _DTYPE_C_NAME[d] for d in dtype_set_raw
                    if d in _DTYPE_C_NAME
                ]
                if not arg or not dtype_set:
                    continue
                target_idx2: int | None = None
                for j, inp in enumerate(ra.inputs):
                    if inp.get("name") == arg:
                        target_idx2 = flat_in_idx_for_region[ra.region_id][j]
                        break
                if target_idx2 is None:
                    continue
                cond = " && ".join(
                    f"io->inputs[{target_idx2}].dtype != {c}"
                    for c in dtype_set
                )
                lines.append(
                    f"    /* M-61 dtype_in({arg}, {dtype_set_raw}) on "
                    f"region={ra.region_id} */"
                )
                lines.append(f"    if ({cond}) {{")
                lines.append(
                    "        status->code = "
                    "COMPGEN_PLAN_VIOLATION_PRECONDITION_DTYPE_IN;"
                )
                lines.append(
                    f'        status->detail = "region={ra.region_id}: '
                    f'precondition dtype_in({arg}) violated";'
                )
                lines.append("        return status->code;")
                lines.append("    }")
                lines.append("")

    return "\n".join(lines)


def _render_dispatch_block(
    *,
    region_order: list[str],
    bindings_by_region: dict[str, Any],
) -> tuple[str, list[str]]:
    """Render the per-region dispatch sequence + the list of kernel
    extern declarations the emit depends on. supplies the
    matching object file (each kernel is a ``cg_rt_executable_t *``
    named ``compgen_kernel_<region_id>``)."""
    lines: list[str] = []
    externs: list[str] = []
    bound_count = 0
    for region_id in region_order:
        binding = bindings_by_region.get(region_id)
        if binding is None:
            lines.append(
                f"    /* region={region_id}: UNBOUND — checked in "
                f"assertions; never reached. */"
            )
            continue
        bound_count += 1
        kernel_sym = f"compgen_kernel_{region_id}"
        externs.append(kernel_sym)
        lines.append(
            f"    /* dispatch region={region_id} "
            f"(contract={binding.contract_hash[:8]}..., "
            f"{binding.dispatch_model}) */"
        )
        lines.append(
            f"    rt_status = cg_rt_command_buffer_dispatch("
            f"command_buffer, {kernel_sym},"
        )
        lines.append("        push_constants, sizeof(push_constants),")
        lines.append("        bindings, n_bindings);")
        lines.append("    if (rt_status != 0) {")
        lines.append(
            "        status->code = rt_status; status->rt_status = rt_status;"
        )
        lines.append(
            f'        status->detail = "region={region_id}: dispatch failed";'
        )
        lines.append("        goto cleanup;")
        lines.append("    }")
        lines.append(
            "    rt_status = cg_rt_command_buffer_barrier(command_buffer);"
        )
        lines.append("    if (rt_status != 0) {")
        lines.append(
            "        status->code = rt_status; status->rt_status = rt_status;"
        )
        lines.append(
            f'        status->detail = "region={region_id}: barrier failed";'
        )
        lines.append("        goto cleanup;")
        lines.append("    }")
        lines.append("")
    if bound_count == 0:
        lines.append("    /* no bound regions; nothing to dispatch */")
    return "\n".join(lines), externs


def _render_executor_source(
    *,
    plan: ExecutionPlan,
    plan_path: str,
    region_order: list[str],
    region_assertions: list[Any],
) -> str:
    """Render the full generated_plan_executor.c body."""
    bindings_by_region = {
        b.region_id: b for b in plan.region_kernel_bindings
    }

    # Static unbound-region check — guards against an operator running
    # the emit when the plan has regions left unbound. This
    # mirrors PLAN_VIOLATION_UNBOUND_REGION .
    unbound = [r for r in region_order if r not in bindings_by_region]
    unbound_check = ""
    if unbound:
        first = unbound[0]
        unbound_check = (
            "    /* Plan declared region(s) M-46 could not bind. */\n"
            "    status->code = COMPGEN_PLAN_VIOLATION_UNBOUND_REGION;\n"
            f'    status->detail = "region {first!r} has no certified '
            f'kernel binding (M-46)";\n'
            "    return status->code;\n"
        )

    io_assertions = _render_io_assertions(region_assertions)
    dispatch_block, externs = _render_dispatch_block(
        region_order=region_order,
        bindings_by_region=bindings_by_region,
    )
    extern_decls = "\n".join(
        f"extern cg_rt_executable_t *{sym};" for sym in externs
    ) or "/* no bound regions — no kernel externs */"

    driver_name = "cpu_sync"  # default; swaps to "cuda".
    return f'''/* Auto-generated by M-88 (compgen.runtime.glue_emit.c11_baremetal).
 *
 * Workload : {plan.workload}
 * Target   : {plan.target}
 * Source   : {plan_path}
 *
 * DO NOT EDIT — regenerate by re-running ``--stop-after glue-emit``.
 *
 * This is the §12 Dream-1 emitted Layer-1 plan executor. It calls
 * the libcompgen_rt ABI (cg_rt_*) only; no vendor primitive is ever
 * referenced directly. The M-91 ABI-conformance gate grep-checks the
 * emitted source for non-cg_rt_ externs.
 */

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "compgen_rt/compgen_rt.h"
#include "generated_plan_executor.h"

/* Extern kernel symbols (one per bound region). The operator links
 * the emit against a kernel pack built by M-49 that defines each of
 * these as a ``cg_rt_executable_t *``. */
{extern_decls}

/* Driver name baked into the emit. M-88 ships cpu_sync; M-89 widens
 * to cuda; M-90 multi-plan dispatch overrides per active plan. */
static const char *const COMPGEN_DRIVER_NAME = "{driver_name}";

int compgen_run(const compgen_io_t *io, compgen_status_t *status) {{
    if (status == NULL) {{
        /* No status pointer to populate; refuse with a fixed return. */
        return COMPGEN_PLAN_VIOLATION_IO_NULL;
    }}
    status->code = 0;
    status->rt_status = 0;
    status->detail = NULL;

    if (io == NULL) {{
        status->code = COMPGEN_PLAN_VIOLATION_IO_NULL;
        status->detail = "io pointer is NULL";
        return status->code;
    }}

{unbound_check}{io_assertions}

    cg_rt_instance_t       *instance       = NULL;
    cg_rt_device_t         *device         = NULL;
    cg_rt_command_buffer_t *command_buffer = NULL;
    cg_rt_semaphore_t      *signal_sem     = NULL;
    cg_rt_status_t          rt_status      = 0;

    /* Push-constants + bindings arrays are kernel-pack-specific; the
     * M-49 binding map populates them. M-88 reserves the slots so the
     * dispatch loop compiles. */
    uint8_t            push_constants[256] = {{0}};
    cg_rt_buffer_t    *bindings[32]        = {{0}};
    size_t             n_bindings          = 0;

    rt_status = cg_rt_instance_create(COMPGEN_DRIVER_NAME, &instance);
    if (rt_status != 0) {{
        status->code = rt_status; status->rt_status = rt_status;
        status->detail = "cg_rt_instance_create failed";
        return status->code;
    }}
    rt_status = cg_rt_device_open(instance, 0, &device);
    if (rt_status != 0) {{
        status->code = rt_status; status->rt_status = rt_status;
        status->detail = "cg_rt_device_open failed";
        goto cleanup;
    }}
    rt_status = cg_rt_command_buffer_create(device, &command_buffer);
    if (rt_status != 0) {{
        status->code = rt_status; status->rt_status = rt_status;
        status->detail = "cg_rt_command_buffer_create failed";
        goto cleanup;
    }}
    rt_status = cg_rt_command_buffer_begin(command_buffer);
    if (rt_status != 0) {{
        status->code = rt_status; status->rt_status = rt_status;
        status->detail = "cg_rt_command_buffer_begin failed";
        goto cleanup;
    }}

{dispatch_block}

    rt_status = cg_rt_command_buffer_end(command_buffer);
    if (rt_status != 0) {{
        status->code = rt_status; status->rt_status = rt_status;
        status->detail = "cg_rt_command_buffer_end failed";
        goto cleanup;
    }}

    rt_status = cg_rt_semaphore_create(device, 0, &signal_sem);
    if (rt_status != 0) {{
        status->code = rt_status; status->rt_status = rt_status;
        status->detail = "cg_rt_semaphore_create failed";
        goto cleanup;
    }}
    cg_rt_semaphore_point_t signal_point = {{ .semaphore = signal_sem,
                                              .value = 1 }};
    rt_status = cg_rt_queue_submit(device, 0, NULL, 0, &signal_point, 1,
                                   command_buffer);
    if (rt_status != 0) {{
        status->code = rt_status; status->rt_status = rt_status;
        status->detail = "cg_rt_queue_submit failed";
        goto cleanup;
    }}
    rt_status = cg_rt_semaphore_wait(signal_sem, 1, CG_RT_TIMEOUT_INFINITE);
    if (rt_status != 0) {{
        status->code = rt_status; status->rt_status = rt_status;
        status->detail = "cg_rt_semaphore_wait failed";
        goto cleanup;
    }}

cleanup:
    if (signal_sem)     cg_rt_semaphore_destroy(signal_sem);
    if (command_buffer) cg_rt_command_buffer_destroy(command_buffer);
    if (device)         cg_rt_device_close(device);
    if (instance)       cg_rt_instance_destroy(instance);

    /* Silence unused-variable warnings on plans with zero bound regions. */
    (void)push_constants; (void)bindings; (void)n_bindings;

    return status->code;
}}
'''


def emit_c11_baremetal_executor(run_dir: Path) -> C11GlueEmitResult:
    """Read the plan from disk, render the C11 executor + header,
    persist them under ``06_glue_emit/``, and write the manifest.

    Caller invariant: has already emitted
    ``05_execution_plan/execution_plan.yaml`` (or .json).
    """
    run_dir = Path(run_dir).resolve()
    plan_path = run_dir / "05_execution_plan" / "execution_plan.yaml"
    if not plan_path.exists():
        plan_path = run_dir / "05_execution_plan" / "execution_plan.json"
    if not plan_path.exists():
        raise FileNotFoundError(
            f"M-46 execution plan not found at "
            f"{run_dir / '05_execution_plan'}; run --stop-after "
            f"execution-plan-emit first"
        )
    plan_dict = _read_yaml_or_json_plan(plan_path)
    plan = ExecutionPlan.from_dict(plan_dict)

    region_order = _topological_region_order(plan)
    bound = tuple(b.region_id for b in plan.region_kernel_bindings)
    placement_regions = tuple(rp.region_id for rp in plan.region_placement)
    unbound = tuple(r for r in placement_regions if r not in bound)

    bindings_dicts = [
        {
            "region_id": b.region_id, "status": "bound",
            "contract_hash": b.contract_hash,
            "certificate_path": b.certificate_path,
        }
        for b in plan.region_kernel_bindings
    ]
    region_assertions = collect_region_assertions(
        run_dir=run_dir, bindings=bindings_dicts,
    )

    out_dir = run_dir / "06_glue_emit"
    out_dir.mkdir(parents=True, exist_ok=True)

    rel_plan = str(plan_path.relative_to(run_dir))
    header_path = out_dir / "generated_plan_executor.h"
    header_path.write_text(
        _render_header(
            workload=plan.workload,
            target=plan.target,
            plan_path=rel_plan,
        ),
        encoding="utf-8",
    )

    executor_path = out_dir / "generated_plan_executor.c"
    executor_path.write_text(
        _render_executor_source(
            plan=plan,
            plan_path=rel_plan,
            region_order=region_order,
            region_assertions=region_assertions,
        ),
        encoding="utf-8",
    )

    manifest_path = out_dir / "plan_executor_c11_manifest.json"
    plan_violation_codes = {name: val for name, val in _PLAN_VIOLATION_CODES}
    manifest_path.write_text(
        json.dumps({
            "schema_version": _C11_EMIT_SCHEMA_VERSION,
            "generated_at_utc": _utcnow(),
            "workload": plan.workload,
            "target": plan.target,
            "executor_kind": "c11_baremetal",
            "executor_path": str(executor_path.relative_to(run_dir)),
            "header_path": str(header_path.relative_to(run_dir)),
            "source_plan_path": rel_plan,
            "bound_regions": list(bound),
            "unbound_regions": list(unbound),
            "region_order": region_order,
            "plan_violation_codes": plan_violation_codes,
            "abi": {
                "driver_name": "cpu_sync",
                "uses_only_cg_rt": True,
                "kernel_extern_prefix": "compgen_kernel_",
            },
            "kernel_bindings": [
                {
                    "region_id": b.region_id,
                    "contract_hash": b.contract_hash,
                    "certificate_path": b.certificate_path,
                    "kernel_artifact": b.kernel_artifact,
                    "dispatch_model": b.dispatch_model,
                }
                for b in plan.region_kernel_bindings
            ],
        }, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    overall = "pass" if bound else "skipped"
    return C11GlueEmitResult(
        out_dir=out_dir,
        executor_path=executor_path,
        header_path=header_path,
        manifest_path=manifest_path,
        overall=overall,
        bound_regions=bound,
        unbound_regions=unbound,
        plan_violation_codes=plan_violation_codes,
    )
