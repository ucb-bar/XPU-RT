"""M-63 — Coverage-first scheduling.

The Phase C pipeline emits exactly one kernel-codegen task per run
(for the recipe-planning-selected region). Other regions sharing the
same shape class — and there can be many; merlin_mlp_wide has three
matmul regions all on host_cpu with the same dtype + layout — go
unbound. M-63 closes this with a *coverage-first* analysis pass:

1. **Coverage pass** (``first-pass-coverage`` and ``both`` modes):
   walk every region dossier under
   ``02_graph_analysis/region_dossiers/``; for each compute-tiled
   region, derive a canonical contract hash from its shape facts;
   look up a matching certificate via M-58's
   :func:`find_certificate_by_canonical_hash`. When the hash matches,
   emit an additional coverage binding so M-46's
   ``region_kernel_bindings.json`` reflects the reuse. The
   ``coverage_report.json`` records per-canonical-hash group sizes
   so a downstream tactician can see that one verified kernel
   covered N regions.

2. **Specialization pass** (``specialize`` and ``both`` modes):
   list every region NOT covered by the coverage pass (or not
   matching an existing cert), sorted by analytical-cost descending.
   The ``specialization_report.json`` is an advisory: a future
   iteration of the pipeline can pick the top-N entries for a
   second auction round with shape-specialized contracts. M-63
   itself does not iterate the pipeline — it produces the report.

Modes:

* ``both`` (default) — coverage + specialization.
* ``first-pass-coverage`` — coverage report only.
* ``specialize`` — specialization report only (assumes a prior
  coverage pass has already run).
* ``disabled`` — no-op.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import structlog

log = structlog.get_logger()


_COVERAGE_SCHEMA = "coverage_report_v1"
_SPECIALIZATION_SCHEMA = "specialization_report_v1"


# Gap #10: archetypes the coverage pass considers. matmul is the
# original M-63 target; pointwise + reduce families ride this list
# so the coverage analysis surfaces non-matmul reuse opportunities
# too.
_COVERED_KINDS: frozenset[str] = frozenset({
    "matmul", "bias_add",
    "elementwise_relu", "elementwise_gelu",
    "elementwise_add", "elementwise_mul", "elementwise_sub",
    "softmax", "layer_norm", "batch_norm",
    "reduce_sum", "reduce_mean", "reduce_max", "argmax",
})


@dataclass(frozen=True)
class CoverageGroup:
    """One ``(canonical_contract_hash → list[region_id])`` group."""

    canonical_contract_hash: str
    region_ids: tuple[str, ...]
    cert_present: bool
    cert_path: str = ""
    cert_winner_provider: str = ""
    coverage_inflation: int = 0  # extra bindings added beyond the original cert's region

    def to_dict(self) -> dict[str, Any]:
        return {
            "canonical_contract_hash": self.canonical_contract_hash,
            "region_ids": list(self.region_ids),
            "size": len(self.region_ids),
            "cert_present": self.cert_present,
            "cert_path": self.cert_path,
            "cert_winner_provider": self.cert_winner_provider,
            "coverage_inflation": self.coverage_inflation,
        }


@dataclass(frozen=True)
class CoverageResult:
    overall: str  # "pass" | "skipped" | "no_dossiers" | "no_certs"
    mode: str
    groups: tuple[CoverageGroup, ...] = ()
    coverage_inflation_total: int = 0
    specialization_top_n: tuple[dict[str, Any], ...] = ()
    coverage_report_path: str = ""
    specialization_report_path: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "overall": self.overall,
            "mode": self.mode,
            "groups": [g.to_dict() for g in self.groups],
            "coverage_inflation_total": self.coverage_inflation_total,
            "specialization_top_n": list(self.specialization_top_n),
            "coverage_report_path": self.coverage_report_path,
            "specialization_report_path": self.specialization_report_path,
        }


def _utcnow() -> str:
    return datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_json_or_none(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return _read_json(path)
    except json.JSONDecodeError:
        return None


def _coverage_signature(
    *,
    dossier: dict[str, Any],
    target_name: str,
) -> str:
    """Build a coverage-matching signature from a region dossier.

    The signature is a stable string ``"<target>|<archetype>|<dtype>|
    <layout>|<input_shapes>|<output_shapes>"`` that two regions share
    iff they could reuse the same kernel under matching tile choices.
    Tile choice itself is a candidate-selection artifact, NOT a
    kernel-shape fact, so it's deliberately excluded — coverage
    matching is on region-level IO equivalence.

    Returns an empty string when the dossier doesn't supply enough
    shape information for a stable signature.
    """
    region_shape = dossier.get("region_shape") or {}
    input_shapes = region_shape.get("input_shapes") or []
    output_shapes = region_shape.get("output_shapes") or []
    dtype = (region_shape.get("dtype") or "f32").lower()
    kind = (dossier.get("kind") or "").lower()

    # Gap #10 closure: per-archetype arity. matmul = 2 inputs, 1 output.
    # pointwise = 1+ inputs, 1 output. reduce = 1 input, 1 output.
    if kind in ("matmul",):
        if (
            len(input_shapes) < 2
            or len(input_shapes[0]) != 2
            or len(input_shapes[1]) != 2
        ):
            return ""
        if not output_shapes or len(output_shapes[0]) != 2:
            return ""
        archetype = "compute_tiled"
        sig_inputs = input_shapes[:2]
    elif kind.startswith("elementwise_") or kind in ("bias_add",):
        archetype = "pointwise"
        # Pointwise: take first 1-2 inputs (output shape = input shape).
        if not input_shapes:
            return ""
        sig_inputs = input_shapes[: max(1, min(2, len(input_shapes)))]
        # Output may be omitted for unary; fall back to input shape.
        if not output_shapes and sig_inputs:
            output_shapes = [sig_inputs[0]]
    elif kind.startswith("reduce_") or kind in ("softmax", "layer_norm", "batch_norm", "argmax"):
        archetype = "reduce"
        if not input_shapes:
            return ""
        sig_inputs = input_shapes[:1]
        if not output_shapes:
            return ""
    else:
        return ""

    in_repr = ";".join(",".join(str(d) for d in s) for s in sig_inputs)
    out_repr = ";".join(",".join(str(d) for d in s) for s in output_shapes[:1])
    layout = "row_major"  # M-60 default
    return (
        f"{target_name}|{archetype}|{dtype}|{layout}|{in_repr}|{out_repr}"
    )


def _signature_from_certificate(*, run_dir: Path, cert: Any) -> str:
    """Derive the coverage signature for an existing cert by reading
    its source contract file from disk.

    Gap #10 closure: archetype is read from the contract body so
    pointwise + reduce certs sign correctly (was previously
    locked to compute_tiled).
    """
    contract_rel = getattr(cert, "contract_path", "") or ""
    if not contract_rel:
        return ""
    contract_path = run_dir / contract_rel
    body = _read_json_or_none(contract_path)
    if body is None:
        return ""
    try:
        target = (
            (body.get("orchestration") or {})
            .get("execution", {})
            .get("hardware", {})
            .get("target_name", "")
        )
        archetype = str(body.get("archetype") or "compute_tiled").lower()
        # Reconstruct input/output shape strings.
        io_body = body.get("io") or {}
        in_dims = [
            tuple(t["shape"]["dims"]) for t in (io_body.get("inputs") or [])
        ]
        out_dims = [
            tuple(t["shape"]["dims"]) for t in (io_body.get("outputs") or [])
        ]
        dtypes = [
            (t.get("dtype_class") or ["f32"])[0]
            for t in (io_body.get("inputs") or [])
        ]
        dtype = (dtypes[0] if dtypes else "f32").lower()
        # Slice to per-archetype arity (matches dossier-side signature).
        if archetype == "compute_tiled":
            in_dims = in_dims[:2]
        elif archetype == "pointwise":
            in_dims = in_dims[: max(1, min(2, len(in_dims)))]
        elif archetype == "reduce":
            in_dims = in_dims[:1]
        in_repr = ";".join(",".join(str(d) for d in s) for s in in_dims)
        out_repr = ";".join(",".join(str(d) for d in s) for s in out_dims[:1])
        layout = "row_major"
        return f"{target}|{archetype}|{dtype}|{layout}|{in_repr}|{out_repr}"
    except Exception:  # noqa: BLE001
        return ""


def _load_certificates(run_dir: Path) -> list[Any]:
    """Load every cert under ``04_kernel_codegen/certificates/``."""
    cert_dir = run_dir / "04_kernel_codegen" / "certificates"
    if not cert_dir.exists():
        return []
    out: list[Any] = []
    try:
        from compgen.kernels.kernel_certificate import KernelCertificate
    except Exception:  # noqa: BLE001
        return []
    for path in sorted(cert_dir.glob("*.json")):
        try:
            body = json.loads(path.read_text(encoding="utf-8"))
            out.append(KernelCertificate.from_dict(body))
        except Exception:  # noqa: BLE001
            continue
    return out


def _load_target_profile(run_dir: Path) -> dict[str, Any]:
    """Best-effort target-profile load for canonical-hash derivation.

    Reads the manifest's recorded target_config_path; falls back to
    ``configs/targets/<target_id>.yaml`` rooted at the repo. Returns
    empty dict when nothing matches — from_recipe still works with
    defaults but the M-60 hardware envelope fields stay unpopulated
    (so the hash drifts from the materialised one).
    """
    manifest = _read_json_or_none(run_dir / "run_manifest.json")
    target_section = (manifest or {}).get("target") or {}
    target_path = target_section.get("config_path") or ""
    if target_path:
        path = Path(target_path)
        if path.exists():
            try:
                import yaml  # type: ignore[import-untyped]

                return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            except ImportError:
                return _read_json_or_none(path) or {}
    return {}


def _resolve_target_name(run_dir: Path) -> str:
    summary = _read_json_or_none(run_dir / "03_recipe_planning" / "recipe_summary.json")
    if summary and summary.get("target_id"):
        return str(summary["target_id"])
    manifest = _read_json_or_none(run_dir / "run_manifest.json")
    if manifest:
        return str((manifest.get("target") or {}).get("target_id", "") or "host_cpu")
    return "host_cpu"


def _walk_dossiers(run_dir: Path) -> list[tuple[str, dict[str, Any]]]:
    """Return ``(region_id, dossier_body)`` for every dossier under
    ``02_graph_analysis/region_dossiers/``."""
    dossier_dir = run_dir / "02_graph_analysis" / "region_dossiers"
    if not dossier_dir.exists():
        return []
    out: list[tuple[str, dict[str, Any]]] = []
    for p in sorted(dossier_dir.glob("*.json")):
        body = _read_json_or_none(p)
        if body is None:
            continue
        rid = str(body.get("region_id") or p.stem.split("__")[0])
        out.append((rid, body))
    return out


def run_coverage_first(
    *,
    run_dir: Path,
    mode: str = "both",
    specialization_top_n: int = 5,
) -> CoverageResult:
    """Run the M-63 coverage-first pass.

    Reads the run's region dossiers + on-disk certificates; produces
    ``coverage_report.json`` and/or ``specialization_report.json``
    under ``04_kernel_codegen/`` according to ``mode``. When the
    coverage pass finds a region whose canonical hash matches an
    existing cert, it appends a binding row to
    ``05_execution_plan/region_kernel_bindings.json`` so M-47's
    emitter sees the reuse without a second M-43 commit cycle.
    """
    if mode not in ("both", "first-pass-coverage", "specialize", "disabled"):
        raise ValueError(
            f"unknown coverage mode {mode!r}; expected one of "
            f"['both', 'first-pass-coverage', 'specialize', 'disabled']"
        )
    run_dir = Path(run_dir).resolve()
    out_dir = run_dir / "04_kernel_codegen"
    out_dir.mkdir(parents=True, exist_ok=True)

    if mode == "disabled":
        return CoverageResult(overall="skipped", mode=mode)

    dossiers = _walk_dossiers(run_dir)
    if not dossiers:
        return CoverageResult(overall="no_dossiers", mode=mode)

    target_name = _resolve_target_name(run_dir)
    target_profile = _load_target_profile(run_dir)  # noqa: F841 — reserved for M-65

    # Gap #10: pointwise + reduce coverage in addition to matmul.
    # Signature arity is enforced inside _coverage_signature.
    # Build coverage_signature → [(region_id, dossier)] groups.
    groups_dict: dict[str, list[tuple[str, dict[str, Any]]]] = {}
    for region_id, dossier in dossiers:
        kind = (dossier.get("kind") or "").lower()
        if kind not in _COVERED_KINDS:
            continue
        signature = _coverage_signature(
            dossier=dossier, target_name=target_name,
        )
        if not signature:
            continue
        groups_dict.setdefault(signature, []).append((region_id, dossier))

    # Build a sig → cert lookup from the on-disk certs.
    sig_to_cert: dict[str, Any] = {}
    for cert in _load_certificates(run_dir):
        sig = _signature_from_certificate(run_dir=run_dir, cert=cert)
        if sig:
            sig_to_cert[sig] = cert

    coverage_groups: list[CoverageGroup] = []
    coverage_bindings_added: list[dict[str, Any]] = []

    if mode in ("both", "first-pass-coverage"):
        existing_bindings = _existing_bindings(run_dir)
        existing_region_ids = {b.get("region_id") for b in existing_bindings}

        for signature, members in groups_dict.items():
            cert = sig_to_cert.get(signature)
            cert_path = ""
            cert_winner = ""
            cert_present = False
            inflation = 0
            canonical = ""
            if cert is not None:
                cert_present = True
                cert_path = (
                    f"04_kernel_codegen/certificates/{cert.contract_hash}.json"
                )
                cert_winner = str(
                    (cert.claims or {}).get("auction_provider", "")
                )
                canonical = getattr(cert, "canonical_contract_hash", "")
                # Coverage inflation: every region in the group that
                # isn't already bound becomes a coverage binding.
                for region_id, dossier in members:
                    if region_id in existing_region_ids:
                        continue
                    inflation += 1
                    coverage_bindings_added.append(
                        _build_coverage_binding(
                            region_id=region_id,
                            cert=cert,
                            cert_path_rel=cert_path,
                        )
                    )

            coverage_groups.append(
                CoverageGroup(
                    canonical_contract_hash=canonical or signature,
                    region_ids=tuple(rid for rid, _ in members),
                    cert_present=cert_present,
                    cert_path=cert_path,
                    cert_winner_provider=cert_winner,
                    coverage_inflation=inflation,
                )
            )

        # Append coverage bindings to the execution plan's bindings file.
        if coverage_bindings_added:
            _append_coverage_bindings(
                run_dir=run_dir, new_bindings=coverage_bindings_added,
            )

    coverage_inflation_total = sum(g.coverage_inflation for g in coverage_groups)

    coverage_report_path = ""
    if mode in ("both", "first-pass-coverage"):
        report_body = {
            "schema_version": _COVERAGE_SCHEMA,
            "generated_at_utc": _utcnow(),
            "mode": mode,
            "target_name": target_name,
            "groups": [g.to_dict() for g in coverage_groups],
            "coverage_inflation_total": coverage_inflation_total,
            "summary": {
                "n_groups": len(coverage_groups),
                "n_groups_with_cert": sum(
                    1 for g in coverage_groups if g.cert_present
                ),
                "max_group_size": max(
                    (len(g.region_ids) for g in coverage_groups), default=0,
                ),
            },
        }
        cp = out_dir / "coverage_report.json"
        cp.write_text(
            json.dumps(report_body, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        coverage_report_path = str(cp.relative_to(run_dir))

    # Specialization pass.
    specialization_top: list[dict[str, Any]] = []
    specialization_report_path = ""
    if mode in ("both", "specialize"):
        # Rank EVERY compute_tiled region by analytical_cost (latency)
        # descending. Regions covered by an existing cert are tagged
        # ``coverage_status: covered``; uncovered regions are
        # candidates for shape-specialization.
        spec_rows: list[dict[str, Any]] = []
        covered_region_ids: set[str] = set()
        for g in coverage_groups:
            if g.cert_present:
                covered_region_ids.update(g.region_ids)

        for region_id, dossier in dossiers:
            if (dossier.get("kind") or "").lower() not in _COVERED_KINDS:
                continue
            cost = dossier.get("cost") or {}
            latency_us = float(cost.get("latency_us") or 0.0)
            spec_rows.append(
                {
                    "region_id": region_id,
                    "analytical_cost_us": latency_us,
                    "coverage_status": (
                        "covered" if region_id in covered_region_ids else "uncovered"
                    ),
                    "region_shape": dossier.get("region_shape") or {},
                }
            )
        spec_rows.sort(
            key=lambda r: (-float(r["analytical_cost_us"]), r["region_id"]),
        )
        # Recommendations: top-N uncovered regions are the
        # specialization candidates.
        uncovered_top = [r for r in spec_rows if r["coverage_status"] == "uncovered"]
        specialization_top = tuple(uncovered_top[:specialization_top_n])

        spec_body = {
            "schema_version": _SPECIALIZATION_SCHEMA,
            "generated_at_utc": _utcnow(),
            "mode": mode,
            "ranked_regions": spec_rows,
            "recommended_specialization_targets": list(specialization_top),
            "summary": {
                "n_regions_total": len(spec_rows),
                "n_covered": sum(
                    1 for r in spec_rows if r["coverage_status"] == "covered"
                ),
                "n_uncovered": sum(
                    1 for r in spec_rows if r["coverage_status"] == "uncovered"
                ),
            },
        }
        sp = out_dir / "specialization_report.json"
        sp.write_text(
            json.dumps(spec_body, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        specialization_report_path = str(sp.relative_to(run_dir))

    return CoverageResult(
        overall="pass",
        mode=mode,
        groups=tuple(coverage_groups),
        coverage_inflation_total=coverage_inflation_total,
        specialization_top_n=tuple(specialization_top),
        coverage_report_path=coverage_report_path,
        specialization_report_path=specialization_report_path,
    )


def _existing_bindings(run_dir: Path) -> list[dict[str, Any]]:
    """Read the M-46 bindings file. Returns empty list when absent."""
    bp = run_dir / "05_execution_plan" / "region_kernel_bindings.json"
    body = _read_json_or_none(bp)
    if body is None:
        return []
    return list(body.get("bindings") or [])


def _build_coverage_binding(
    *,
    region_id: str,
    cert: Any,
    cert_path_rel: str,
) -> dict[str, Any]:
    """Construct a binding row for a coverage-inflated region.

    The binding points at the existing cert (no second M-43 cycle);
    its kernel_artifact comes from the cert's recorded artifact_paths;
    its dispatch_model defaults to 'sync' (M-50 widens later).
    """
    artifact_paths = cert.artifact_paths if hasattr(cert, "artifact_paths") else {}
    kernel_artifact = artifact_paths.get("kernel_source", "")
    return {
        "region_id": region_id,
        "status": "bound",
        "contract_hash": cert.contract_hash,
        "canonical_contract_hash": getattr(cert, "canonical_contract_hash", ""),
        "certificate_path": cert_path_rel,
        "kernel_artifact": kernel_artifact,
        "dispatch_model": "sync",
        "unbound_reason": "",
        "coverage_source": True,
    }


def _append_coverage_bindings(
    *,
    run_dir: Path,
    new_bindings: list[dict[str, Any]],
) -> None:
    """Append coverage-inflated rows to ``region_kernel_bindings.json``.

    The original bindings emitted by M-46 stay untouched; the coverage
    rows extend the list. ``bound_count`` and ``unbound_count`` are
    recomputed; ``coverage_inflated_count`` records the M-63 addition
    so the M-47 emitter can audit it.
    """
    bp = run_dir / "05_execution_plan" / "region_kernel_bindings.json"
    body = _read_json_or_none(bp)
    if body is None:
        # No prior bindings file — M-63 doesn't bootstrap one.
        return
    existing = list(body.get("bindings") or [])
    merged = existing + new_bindings
    body["bindings"] = merged
    body["bound_count"] = sum(1 for b in merged if b.get("status") == "bound")
    body["unbound_count"] = sum(1 for b in merged if b.get("status") == "unbound")
    body["coverage_inflated_count"] = len(new_bindings)
    body["coverage_inflated_at_utc"] = _utcnow()
    bp.write_text(
        json.dumps(body, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


__all__ = [
    "CoverageGroup",
    "CoverageResult",
    "run_coverage_first",
]
