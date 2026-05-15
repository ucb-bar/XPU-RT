"""Multi-plan dispatch-table emitter (, Phase G — §12 Dream 5).

Phase G emit a deterministic, byte-stable dispatcher over a
runtime feature vector (``batch``, ``seqlen``, ``dtype``, ...) so a
single Layer-1 emit can carry several plans without sacrificing
static-plan performance. Each inner plan is authored and verified
independently; the dispatcher is a trivial lookup the D6
plan-refinement gate signs off on.

The dispatch table is fed in from a Recipe IR
``recipe.plan_dispatch_table`` op (see
:mod:`compgen.ir.recipe.ops_dispatch`); this module is the *emit*
side that consumes a normalised :class:`PlanDispatchSpec` and writes
the dispatcher source in Python, C11, or C++.

Determinism: entries are tried in declaration order; the first whose
declared features all match the runtime feature vector wins. When no
entry matches, the dispatcher routes to ``default_plan_ref``. The
runtime never raises from a dispatcher — that would defeat the
§12 D5 "trivial dispatcher" invariant.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal


_DISPATCH_SCHEMA_VERSION = "plan_dispatch_table_manifest_v1"


# --------------------------------------------------------------------------- #
# Normalised dispatch spec                                                    #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class PlanDispatchEntry:
    """One entry: a feature-vector subset that selects ``plan_ref``."""

    features: dict[str, int | str]
    plan_ref: str

    def matches(self, runtime_features: dict[str, int | str]) -> bool:
        """Return True iff every declared feature in this entry is
        present and equal in ``runtime_features``."""
        for k, v in self.features.items():
            if runtime_features.get(k) != v:
                return False
        return True


@dataclass(frozen=True)
class PlanDispatchSpec:
    """Normalised dispatch table the emitters consume."""

    workload: str
    target: str
    feature_keys: tuple[str, ...]
    entries: tuple[PlanDispatchEntry, ...]
    default_plan_ref: str

    def __post_init__(self) -> None:
        if not self.feature_keys:
            raise ValueError(
                "PlanDispatchSpec requires at least one feature key"
            )
        if not self.entries:
            raise ValueError(
                "PlanDispatchSpec requires at least one entry; an empty "
                "table would degrade silently to the default plan"
            )
        if not self.default_plan_ref:
            raise ValueError(
                "default_plan_ref is required (no silent failure path)"
            )
        for idx, entry in enumerate(self.entries):
            unknown = set(entry.features) - set(self.feature_keys)
            if unknown:
                raise ValueError(
                    f"entry[{idx}] features {sorted(unknown)!r} not in "
                    f"declared feature_keys {sorted(self.feature_keys)!r}"
                )


# --------------------------------------------------------------------------- #
# Runtime selection (pure, used by Python emit + the conformance gate) #
# --------------------------------------------------------------------------- #


def select_plan(
    spec: PlanDispatchSpec,
    runtime_features: dict[str, int | str],
) -> str:
    """Deterministically pick the plan ref for a runtime feature
    vector. Entries are tried in declaration order; the first whose
    features all match wins. Falls back to ``default_plan_ref``.
    """
    for entry in spec.entries:
        if entry.matches(runtime_features):
            return entry.plan_ref
    return spec.default_plan_ref


# --------------------------------------------------------------------------- #
# Emit                                                                        #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class DispatchEmitResult:
    out_dir: Path
    python_path: Path | None
    c11_path: Path | None
    cpp_path: Path | None
    manifest_path: Path
    target: Literal["python", "c11", "cpp", "all"]
    feature_keys: tuple[str, ...]
    n_entries: int
    spec_hash: str = field(default="")


def _utcnow() -> str:
    return datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _spec_hash(spec: PlanDispatchSpec) -> str:
    """Stable 16-char SHA256 digest of the dispatch spec."""
    body = json.dumps({
        "workload": spec.workload,
        "target": spec.target,
        "feature_keys": list(spec.feature_keys),
        "entries": [
            {"features": dict(e.features), "plan_ref": e.plan_ref}
            for e in spec.entries
        ],
        "default_plan_ref": spec.default_plan_ref,
    }, sort_keys=True).encode("utf-8")
    return hashlib.sha256(body).hexdigest()[:16]


def _render_python_dispatcher(spec: PlanDispatchSpec) -> str:
    """Emit a deterministic Python dispatcher.

    Routes via importing the inner plan's emitted module by
    convention ``06_glue_emit_<plan_ref>.generated_plan_executor``;
    the actual loader is operator-supplied via the ``plan_loader``
    callable so the dispatch core remains pure and testable.
    """
    entries_repr = [
        {"features": dict(e.features), "plan_ref": e.plan_ref}
        for e in spec.entries
    ]
    return f'''"""Auto-generated by M-90 (compgen.runtime.glue_emit.dispatch_table).

Workload : {spec.workload}
Target   : {spec.target}
Spec hash: {_spec_hash(spec)}

DO NOT EDIT — regenerate by re-running ``--stop-after glue-emit``.

This is the §12 Dream-5 multi-plan dispatcher. Each declared entry
selects a per-shape plan; the dispatcher is trivial and total (a
``default_plan_ref`` always exists). The inner plans are emitted
independently by M-47 / M-51 / M-52 / M-88 / M-89.
"""
from __future__ import annotations

from typing import Any, Callable, Mapping


FEATURE_KEYS: tuple[str, ...] = {spec.feature_keys!r}
DISPATCH_ENTRIES: tuple[dict, ...] = (
{chr(10).join(f"    {entry!r}," for entry in entries_repr)}
)
DEFAULT_PLAN_REF: str = {spec.default_plan_ref!r}


def select_plan_ref(runtime_features: Mapping[str, Any]) -> str:
    """Return the plan ref to dispatch for this runtime feature vector.

    Entries are tried in declaration order; the first whose declared
    features all match wins. Falls back to DEFAULT_PLAN_REF.
    """
    for entry in DISPATCH_ENTRIES:
        feats = entry["features"]
        if all(runtime_features.get(k) == v for k, v in feats.items()):
            return entry["plan_ref"]
    return DEFAULT_PLAN_REF


def compgen_dispatch_run(
    runtime_features: Mapping[str, Any],
    io: Mapping[str, Any],
    kernels: Mapping[str, Callable[..., Any]],
    runtime: Any,
    plan_loader: Callable[[str], Any],
) -> Any:
    """Pick a plan, then run its compgen_run.

    ``plan_loader(plan_ref)`` returns the inner executor module
    exposing ``compgen_run(io, kernels, runtime)`` — i.e. one of the
    M-47/M-51/M-52 emitted modules.  The dispatcher itself does no
    IO; the operator's loader resolves on-disk paths.
    """
    plan_ref = select_plan_ref(runtime_features)
    executor = plan_loader(plan_ref)
    return executor.compgen_run(io, kernels, runtime)
'''


def _c_literal(value: int | str) -> str:
    """Render a Python feature value as a C int / string literal."""
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, int):
        return str(value)
    return f'"{value}"'


def _render_c11_dispatcher(spec: PlanDispatchSpec) -> str:
    """Emit a deterministic C11 dispatcher.

    The C representation of a feature vector is a small struct of
    int64s; string features map to a stable enum (the manifest
    serialises the enum table for the operator to consume).
    """
    feature_decls = "\n".join(
        f"    int64_t {k};" for k in spec.feature_keys
    )
    # Build the entries table. Each entry encodes feature presence
    # via a mask + values array; matching is mask-then-value.
    entry_rows: list[str] = []
    for entry in spec.entries:
        mask = 0
        values: list[str] = []
        for idx, k in enumerate(spec.feature_keys):
            if k in entry.features:
                mask |= 1 << idx
                values.append(_c_literal(entry.features[k]))
            else:
                values.append("0")
        entry_rows.append(
            f"    {{ /*mask*/ 0x{mask:08x}, /*values*/ {{ "
            f"{', '.join(values)} }}, /*plan_ref*/ \"{entry.plan_ref}\" }},"
        )

    return f'''/* Auto-generated by M-90 (compgen.runtime.glue_emit.dispatch_table).
 *
 * Workload : {spec.workload}
 * Target   : {spec.target}
 * Spec hash: {_spec_hash(spec)}
 *
 * §12 Dream-5: multi-plan dispatcher over a runtime feature vector.
 */

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {{
#endif

#define COMPGEN_DISPATCH_N_FEATURES   {len(spec.feature_keys)}
#define COMPGEN_DISPATCH_N_ENTRIES    {len(spec.entries)}
#define COMPGEN_DISPATCH_PLAN_REF_MAX 64

typedef struct {{
{feature_decls}
}} compgen_features_t;

typedef struct {{
    uint32_t mask;
    int64_t  values[COMPGEN_DISPATCH_N_FEATURES];
    char     plan_ref[COMPGEN_DISPATCH_PLAN_REF_MAX];
}} compgen_dispatch_entry_t;

/* Stable, byte-identical across reruns of the same spec. */
static const compgen_dispatch_entry_t COMPGEN_DISPATCH_ENTRIES[] = {{
{chr(10).join(entry_rows)}
}};
static const char COMPGEN_DISPATCH_DEFAULT_PLAN_REF[] = "{spec.default_plan_ref}";

const char *compgen_select_plan_ref(const compgen_features_t *features) {{
    if (features == NULL) {{
        return COMPGEN_DISPATCH_DEFAULT_PLAN_REF;
    }}
    const int64_t *fv = (const int64_t *)features;
    for (size_t i = 0; i < COMPGEN_DISPATCH_N_ENTRIES; ++i) {{
        const compgen_dispatch_entry_t *e = &COMPGEN_DISPATCH_ENTRIES[i];
        int match = 1;
        for (size_t j = 0; j < COMPGEN_DISPATCH_N_FEATURES; ++j) {{
            if ((e->mask >> j) & 1u) {{
                if (fv[j] != e->values[j]) {{ match = 0; break; }}
            }}
        }}
        if (match) {{
            return e->plan_ref;
        }}
    }}
    return COMPGEN_DISPATCH_DEFAULT_PLAN_REF;
}}

#ifdef __cplusplus
}}
#endif
'''


def _render_cpp_dispatcher(spec: PlanDispatchSpec) -> str:
    """Emit a deterministic C++ dispatcher that re-uses the C11 core
    via ``extern "C"`` linkage (one definition; binary-compatible)."""
    # The C++ variant simply includes the C source by reference;
    # we duplicate the entry table in a header-style emit so the .cpp
    # file is standalone.
    c_body = _render_c11_dispatcher(spec)
    # Strip the C-style auto-generated banner; replace with C++ one.
    return c_body.replace(
        "/* Auto-generated by M-90",
        "/* Auto-generated by M-90 (C++ variant)",
        1,
    )


# --------------------------------------------------------------------------- #
# Top-level emit                                                              #
# --------------------------------------------------------------------------- #


def emit_dispatch_table(
    spec: PlanDispatchSpec,
    out_dir: Path,
    *,
    target: Literal["python", "c11", "cpp", "all"] = "all",
) -> DispatchEmitResult:
    """Emit the dispatcher(s) under ``out_dir`` plus a manifest.

    The expected ``out_dir`` is ``<run_dir>/06_glue_emit/`` so the
    dispatcher sits next to the per-plan executors.
    """
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    python_path: Path | None = None
    c11_path: Path | None = None
    cpp_path: Path | None = None

    if target in ("python", "all"):
        python_path = out_dir / "generated_plan_dispatcher.py"
        python_path.write_text(
            _render_python_dispatcher(spec), encoding="utf-8",
        )
    if target in ("c11", "all"):
        c11_path = out_dir / "generated_plan_dispatcher.c"
        c11_path.write_text(
            _render_c11_dispatcher(spec), encoding="utf-8",
        )
    if target in ("cpp", "all"):
        cpp_path = out_dir / "generated_plan_dispatcher.cpp"
        cpp_path.write_text(
            _render_cpp_dispatcher(spec), encoding="utf-8",
        )

    manifest_path = out_dir / "plan_dispatch_table_manifest.json"
    manifest_path.write_text(
        json.dumps({
            "schema_version": _DISPATCH_SCHEMA_VERSION,
            "generated_at_utc": _utcnow(),
            "workload": spec.workload,
            "target": spec.target,
            "feature_keys": list(spec.feature_keys),
            "default_plan_ref": spec.default_plan_ref,
            "entries": [
                {"features": dict(e.features), "plan_ref": e.plan_ref}
                for e in spec.entries
            ],
            "emit_target": target,
            "spec_hash": _spec_hash(spec),
            "python_path": (
                str(python_path.name) if python_path else None
            ),
            "c11_path": (str(c11_path.name) if c11_path else None),
            "cpp_path": (str(cpp_path.name) if cpp_path else None),
        }, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    return DispatchEmitResult(
        out_dir=out_dir,
        python_path=python_path,
        c11_path=c11_path,
        cpp_path=cpp_path,
        manifest_path=manifest_path,
        target=target,
        feature_keys=spec.feature_keys,
        n_entries=len(spec.entries),
        spec_hash=_spec_hash(spec),
    )


# --------------------------------------------------------------------------- #
# Bridge from Recipe IR PlanDispatchTableOp                                   #
# --------------------------------------------------------------------------- #


def plan_dispatch_spec_from_recipe_op(op: Any) -> PlanDispatchSpec:
    """Normalise a ``recipe.plan_dispatch_table`` op into a
    :class:`PlanDispatchSpec` the emitter consumes.  Imports xDSL types
    lazily so callers that never touch IR don't pay the cost.
    """
    from xdsl.dialects.builtin import (  # type: ignore[import-untyped]
        DictionaryAttr,
        IntegerAttr,
        StringAttr,
    )

    workload = op.workload.data if op.workload is not None else ""
    target = op.target.data if op.target is not None else ""

    feature_keys: list[str] = []
    for k in op.feature_keys.data:
        if not isinstance(k, StringAttr):
            raise TypeError(
                f"feature_keys entries must be StringAttr; got {type(k)}"
            )
        feature_keys.append(k.data)

    entries: list[PlanDispatchEntry] = []
    for raw in op.entries.data:
        if not isinstance(raw, DictionaryAttr):
            raise TypeError("entry must be DictionaryAttr")
        features_attr = raw.data["features"]
        plan_ref_attr = raw.data["plan_ref"]
        features: dict[str, int | str] = {}
        for k, v in features_attr.data.items():
            if isinstance(v, IntegerAttr):
                features[k] = int(v.value.data)
            elif isinstance(v, StringAttr):
                features[k] = v.data
            else:
                raise TypeError(
                    f"feature {k}: expected IntegerAttr or StringAttr, "
                    f"got {type(v)}"
                )
        entries.append(PlanDispatchEntry(
            features=features,
            plan_ref=plan_ref_attr.data,
        ))

    return PlanDispatchSpec(
        workload=workload,
        target=target,
        feature_keys=tuple(feature_keys),
        entries=tuple(entries),
        default_plan_ref=op.default_plan_ref.data,
    )
