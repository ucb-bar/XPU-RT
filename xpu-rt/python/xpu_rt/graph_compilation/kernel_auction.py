"""M-57 — Multi-bidder kernel auction.

Bridges M-55 (registry applicability) + M-56 (bid/fulfill protocol) into
a single orchestrator that runs after M-42 emits the kernel-codegen
task and before M-43's commit-driven path produces a single response.

Flow:

::

    [M-42 request] → applicable() → bid() per provider
                                   → top-K by score → fulfill() per top-K
                                   → translate ProviderResult to artifacts on disk
                                   → M-44 verifier per fulfilled set
                                   → M-45 certificate per verified set
                                   → pick winner by perf_estimate_us
                                   → promote winner → standard M-43 response location
                                   → M-46 binds normally

Modes:

* ``multi-bidder`` (default): every applicable provider bids; top-K
  fulfill; first verified bid sets the winner shape, but the auction
  still runs the remaining top-K so the report shows real comparative
  data. Selector picks the verified bid with the lowest
  ``claims.perf_estimate_us``.
* ``first-fit``: stop at the first verified bid in priority order.
* ``disabled``: no-op; today's M-43 commit path remains canonical.

Hard contract:

* The auction never mutates the materialised ``KernelContractV3``.
* Every fulfilled bid's artifacts are sandboxed under
  ``04_kernel_codegen/auction/<task_id>/fulfilled/<provider_name>/``.
* The winner is *copied* (not moved) to the standard M-43 response
  location so the runner-up trail stays inspectable.
* Every bid (including losers) lands in ``auction_report.json`` with
  the verifier's verdict so a future tactician analysis can pick up
  signal even without rerunning the auction.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import structlog

log = structlog.get_logger()


_AUCTION_REPORT_SCHEMA = "auction_report_v1"


# --------------------------------------------------------------------------- #
# Result types
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class _BidRecord:
    provider_name: str
    bid_dict: dict[str, Any]
    score: float
    rank: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider_name": self.provider_name,
            "rank": self.rank,
            "score": self.score,
            "bid": self.bid_dict,
        }


@dataclass(frozen=True)
class _FulfillRecord:
    provider_name: str
    found: bool
    artifact_dir_rel: str  # under run_dir
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider_name": self.provider_name,
            "found": self.found,
            "artifact_dir": self.artifact_dir_rel,
            "error": self.error,
        }


@dataclass(frozen=True)
class _VerifyRecord:
    provider_name: str
    overall: str  # "pass" | "fail" | "skipped"
    failure_kind: str = ""
    failure_summary: str = ""
    validation_report_rel: str = ""
    certificate_rel: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider_name": self.provider_name,
            "overall": self.overall,
            "failure_kind": self.failure_kind,
            "failure_summary": self.failure_summary,
            "validation_report_path": self.validation_report_rel,
            "certificate_path": self.certificate_rel,
        }


@dataclass(frozen=True)
class AuctionResult:
    overall: str  # "pass" | "skipped" | "no_winner" | "no_applicable_providers" | "error"
    mode: str
    task_id: str
    contract_hash: str
    bids: list[_BidRecord] = field(default_factory=list)
    fulfilled: list[_FulfillRecord] = field(default_factory=list)
    verified: list[_VerifyRecord] = field(default_factory=list)
    winner_provider: str = ""
    auction_report_path: str = ""
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": _AUCTION_REPORT_SCHEMA,
            "overall": self.overall,
            "mode": self.mode,
            "task_id": self.task_id,
            "contract_hash": self.contract_hash,
            "bids": [b.to_dict() for b in self.bids],
            "fulfilled": [f.to_dict() for f in self.fulfilled],
            "verified": [v.to_dict() for v in self.verified],
            "winner_provider": self.winner_provider,
            "auction_report_path": self.auction_report_path,
            "error": self.error,
        }


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _utcnow() -> str:
    return datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_json_or_none(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return _read_json(path)


def _bid_score(bid_dict: dict[str, Any]) -> float:
    """Lower is better.

    Score = perf_us / max(confidence, 1e-3) — a high-confidence cheap
    bid wins; a high-confidence cache hit (perf_estimate=+inf, but
    fulfilled in 1s) is special-cased to score 0.0 since the perf
    measurement itself comes after fulfill().
    """
    import math

    confidence = float(bid_dict.get("confidence", 0.0))
    if confidence <= 0.0:
        return float("inf")
    if bid_dict.get("cache_hit", False):
        # A real cache hit means the provider already has a measured
        # kernel; treat as a minimal-cost winner candidate. Real perf
        # ranking lands once we dispatch.
        return 0.0
    perf_us = float(bid_dict.get("perf_estimate_us", float("inf")))
    if math.isinf(perf_us):
        return float("inf")
    return perf_us / confidence


def _v3_to_legacy_contract(contract_v3: Any) -> Any:
    """Bridge a :class:`KernelContractV3` to a legacy ``KernelContract``
    so that providers' existing ``search()`` paths work unchanged.

    Lossy on orchestration (sync/wait_on/lifetimes/fusion not surfaced)
    — providers that need the full V3 should consume it directly via
    M-56's ``bid()``.
    """
    from compgen.kernels.provider import KernelContract

    target = ""
    try:
        target = contract_v3.orchestration.execution.hardware.target_name
    except AttributeError:
        pass

    inputs = []
    outputs = []
    dtypes: list[str] = []
    layout = "row_major"
    try:
        for t in contract_v3.io.inputs:
            inputs.append(tuple(d for d in t.shape.dims if d is not None))
            dtypes.extend(t.dtype_class)
            layout = t.layout.value
        for t in contract_v3.io.outputs:
            outputs.append(tuple(d for d in t.shape.dims if d is not None))
    except AttributeError:
        pass

    op_name = ""
    try:
        op_name = contract_v3.op_name
    except AttributeError:
        pass
    op_family = op_name.split(".")[-1] if "." in op_name else op_name
    if op_family.startswith("aten_"):
        op_family = op_family.removeprefix("aten_")
        for suffix in ("_default", "_Tensor", "_Scalar", "_self_int"):
            if op_family.endswith(suffix):
                op_family = op_family[: -len(suffix)]
                break
    if "matmul" in op_family:
        op_family = "matmul"

    return KernelContract(
        region_id="auction",
        op_family=op_family,
        input_shapes=tuple(inputs),
        output_shapes=tuple(outputs),
        dtypes=tuple(dict.fromkeys(dtypes)) or ("f32",),
        layout=layout,
        target_name=target,
        hardware_key=target,
        objective="latency",
    )


def _ext_for_language(language: str) -> str:
    return {
        "triton": "py",
        "python": "py",
        "cuda": "cu",
        "cpp": "cpp",
        "c++": "cpp",
        "c": "c",
        "asm": "S",
        "ptx": "ptx",
    }.get((language or "").lower(), "txt")


def _translate_provider_result_to_artifacts(
    *,
    result: Any,  # ProviderResult
    artifact_dir: Path,
    contract_v3: Any,
    bid_dict: dict[str, Any],
    provider_name: str,
) -> dict[str, str]:
    """Write a provider's :class:`ProviderResult` to the on-disk artifact
    layout the M-43 commit + M-44 verifier consume.

    Returns the ``artifacts`` map (relative-to-run-dir paths) suitable
    for inclusion in a synthesized ``provider_response_v1`` body.
    """
    artifact_dir.mkdir(parents=True, exist_ok=True)

    # Kernel source.
    ext = _ext_for_language(getattr(result, "language", ""))
    src_name = f"kernel.{ext}"
    src_path = artifact_dir / src_name
    src_path.write_text(getattr(result, "kernel_code", "") or "", encoding="utf-8")

    # Kernel metadata — derived from the V3 contract, not from the
    # provider's free-form metadata. The verifier (M-44) cross-checks
    # these against the contract.
    inputs_meta: list[dict[str, Any]] = []
    outputs_meta: list[dict[str, Any]] = []
    accumulator = "f32"
    try:
        for t in contract_v3.io.inputs:
            inputs_meta.append(
                {
                    "name": t.name,
                    "dims": list(t.shape.dims),
                    "dtype": t.dtype_class[0] if t.dtype_class else "f32",
                    "layout": t.layout.value,
                }
            )
        for t in contract_v3.io.outputs:
            outputs_meta.append(
                {
                    "name": t.name,
                    "dims": list(t.shape.dims),
                    "dtype": t.dtype_class[0] if t.dtype_class else "f32",
                    "layout": t.layout.value,
                }
            )
        accumulator = contract_v3.io.numerics.accumulator_dtype
    except AttributeError:
        pass

    metadata_body = {
        "schema_version": "kernel_metadata_v1",
        "symbol": "compgen_matmul_f32",  # auction default; provider may override
        "inputs": inputs_meta,
        "outputs": outputs_meta,
        "accumulator_dtype": accumulator,
    }
    # Allow the provider to override symbol/etc. via result.metadata.
    pmeta = getattr(result, "metadata", None) or {}
    if "symbol" in pmeta:
        metadata_body["symbol"] = pmeta["symbol"]

    metadata_path = artifact_dir / "kernel_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata_body, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    # Launch config.
    geom = getattr(result, "dispatch_geometry", None)
    launch_body = {
        "schema_version": "launch_config_v1",
        "num_warps": getattr(geom, "num_warps", 1),
        "threadblock_shape": list(getattr(geom, "threadblock_shape", (1,))),
        "grid_shape": list(getattr(geom, "grid_shape", (1,))),
    }
    launch_path = artifact_dir / "launch_config.json"
    launch_path.write_text(
        json.dumps(launch_body, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    # Provider claims — anchored on the BidPreview the auction collected.
    backend = "c_reference"
    if (getattr(result, "language", "") or "").lower() in ("triton", "python", "cuda"):
        backend = "triton"
    claims_body = {
        "schema_version": "provider_claims_v1",
        "backend": backend,
        "supports_dispatch": ["sync"],
        "estimated_registers": int(bid_dict.get("registers_used", 0)),
        "estimated_smem_bytes": int(bid_dict.get("smem_bytes", 0)),
        "expected_numerics": "tolerance_eps",
        "auction_provider": provider_name,
        "auction_perf_estimate_us": bid_dict.get("perf_estimate_us"),
    }
    # Bit-equality refinement when contract numerics demand it.
    try:
        if contract_v3.io.numerics.max_relative_error == 0.0:
            claims_body["expected_numerics"] = "bit_equality"
    except AttributeError:
        pass
    claims_path = artifact_dir / "provider_claims.json"
    claims_path.write_text(
        json.dumps(claims_body, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    return {
        "kernel_source": str(src_path),
        "kernel_metadata": str(metadata_path),
        "launch_config": str(launch_path),
        "provider_claims": str(claims_path),
    }


def _build_response_body(
    *,
    request_body: dict[str, Any],
    artifacts: dict[str, str],
    provider_name: str,
) -> dict[str, Any]:
    """Synthesize a ``provider_response_v1`` body from auction artifacts.

    M-43's verifier path consumes this same shape; the auction's
    fulfill path is just an in-process producer instead of an
    external operator.
    """
    return {
        "schema_version": "provider_response_v1",
        "task_id": request_body["task_id"],
        "contract_hash": request_body["contract_hash"],
        "artifacts": artifacts,
        "claims": {
            "backend": "c_reference"
            if "kernel.c" in (artifacts.get("kernel_source", "") or "")
            else "triton",
            "supports_dispatch": ["sync"],
            "estimated_registers": 0,
            "estimated_smem_bytes": 0,
            "expected_numerics": "tolerance_eps",
        },
        "provider": {
            "kind": f"auction_{provider_name}",
            "model": "",
            "session_id": "",
            "started_at": _utcnow(),
            "finished_at": _utcnow(),
        },
        "contract_feedback": [],
        "notes": f"Auction-fulfilled artifact from provider {provider_name!r}",
    }


# --------------------------------------------------------------------------- #
# Main orchestrator
# --------------------------------------------------------------------------- #


def run_kernel_auction(
    *,
    run_dir: Path,
    mode: str = "multi-bidder",
    bid_cutoff: int = 3,
    registry: Any | None = None,  # ProviderRegistry override (tests)
) -> AuctionResult:
    """Run the M-57 auction and write ``auction/<task_id>/auction_report.json``.

    A no-op when:
    * ``mode == "disabled"``
    * The M-42 request is missing or ``request_kind != "kernel_codegen"``
    * No applicable providers are registered

    On success, the winner's artifact set is also written to the
    standard M-43 response location so M-46/M-47 work unchanged.
    """
    run_dir = Path(run_dir).resolve()

    if mode not in ("multi-bidder", "first-fit", "disabled"):
        raise ValueError(
            f"unknown auction mode {mode!r}; must be one of "
            f"['multi-bidder', 'first-fit', 'disabled']"
        )

    # Locate the request — the auction is keyed off whatever M-42 emitted.
    requests_dir = run_dir / "04_kernel_codegen" / "requests"
    request_files = sorted(requests_dir.glob("*.request.json")) if requests_dir.exists() else []
    if not request_files:
        return AuctionResult(
            overall="skipped",
            mode=mode,
            task_id="",
            contract_hash="",
            error="no_m42_request",
        )

    request_body = _read_json(request_files[0])
    task_id = request_body["task_id"]
    contract_hash = request_body.get("contract_hash", "") or ""

    if request_body.get("request_kind") != "kernel_codegen":
        return AuctionResult(
            overall="skipped",
            mode=mode,
            task_id=task_id,
            contract_hash=contract_hash,
            error=f"request_kind={request_body.get('request_kind')!r}_not_kernel_codegen",
        )

    if mode == "disabled":
        return AuctionResult(
            overall="skipped",
            mode=mode,
            task_id=task_id,
            contract_hash=contract_hash,
            error="auction_disabled_by_caller",
        )

    # Reconstruct the V3 contract.
    contract_path = run_dir / request_body["contract_paths"]["full"]
    if not contract_path.exists():
        return AuctionResult(
            overall="skipped",
            mode=mode,
            task_id=task_id,
            contract_hash=contract_hash,
            error="contract_path_missing",
        )

    from compgen.graph_compilation.kernel_codegen_response import (
        _reconstruct_contract_from_dict,
    )

    contract_v3 = _reconstruct_contract_from_dict(_read_json(contract_path))

    # Build registry + applicable list.
    if registry is None:
        from compgen.kernels.registry import default_registry

        registry = default_registry()

    applicability = registry.applicable(contract_v3)
    applicable_providers = [
        p
        for row, p in zip(applicability, registry._providers, strict=False)
        if row.applicable
    ]

    if not applicable_providers:
        return _write_empty_report(
            run_dir=run_dir,
            mode=mode,
            task_id=task_id,
            contract_hash=contract_hash,
            error="no_applicable_providers",
        )

    # Collect bids.
    from compgen.kernels.registry import collect_bids

    bid_previews = collect_bids(applicable_providers, contract_v3)
    bid_records: list[_BidRecord] = []
    for rank, (provider, bid) in enumerate(
        sorted(
            zip(applicable_providers, bid_previews, strict=False),
            key=lambda pair: _bid_score(pair[1].to_dict()),
        ),
        start=1,
    ):
        bid_records.append(
            _BidRecord(
                provider_name=provider.name,
                bid_dict=bid.to_dict(),
                score=_bid_score(bid.to_dict()),
                rank=rank,
            )
        )

    # Pick top-K by score.
    if mode == "first-fit":
        # First-fit: evaluate one bid at a time in score order.
        ranked_pairs = list(zip(applicable_providers, bid_previews, strict=False))
        ranked_pairs.sort(key=lambda pair: _bid_score(pair[1].to_dict()))
        top_pairs = ranked_pairs  # walk in order, stop on first verified pass
    else:
        ranked_pairs = list(zip(applicable_providers, bid_previews, strict=False))
        ranked_pairs.sort(key=lambda pair: _bid_score(pair[1].to_dict()))
        top_pairs = ranked_pairs[: max(0, int(bid_cutoff))]

    # Fulfill + verify per top-K bid.
    auction_root = run_dir / "04_kernel_codegen" / "auction" / task_id
    auction_root.mkdir(parents=True, exist_ok=True)
    fulfilled_root = auction_root / "fulfilled"
    fulfilled_root.mkdir(parents=True, exist_ok=True)

    fulfilled_records: list[_FulfillRecord] = []
    verified_records: list[_VerifyRecord] = []
    # M-59: aggregate per-provider contract_feedback from each
    # fulfilled bid. Routed into write_auction_feedback_artifacts at
    # the end of the auction.
    per_provider_feedback: list[tuple[str, list[Any]]] = []

    legacy_contract = _v3_to_legacy_contract(contract_v3)
    from compgen.kernels.provider import SearchBudget

    budget = SearchBudget()

    winner_provider: str | None = None
    winner_artifacts: dict[str, str] | None = None

    for provider, bid in top_pairs:
        provider_dir = fulfilled_root / provider.name
        try:
            result = provider.search(legacy_contract, budget)
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "auction.fulfill.error",
                provider=provider.name,
                error=f"{type(exc).__name__}: {exc}",
            )
            fulfilled_records.append(
                _FulfillRecord(
                    provider_name=provider.name,
                    found=False,
                    artifact_dir_rel=str(provider_dir.relative_to(run_dir)),
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
            continue

        if not getattr(result, "found", False) or not getattr(result, "kernel_code", ""):
            fulfilled_records.append(
                _FulfillRecord(
                    provider_name=provider.name,
                    found=False,
                    artifact_dir_rel=str(provider_dir.relative_to(run_dir)),
                    error="provider_returned_not_found",
                )
            )
            continue

        # Translate to artifacts.
        try:
            artifacts = _translate_provider_result_to_artifacts(
                result=result,
                artifact_dir=provider_dir,
                contract_v3=contract_v3,
                bid_dict=bid.to_dict(),
                provider_name=provider.name,
            )
        except Exception as exc:  # noqa: BLE001
            fulfilled_records.append(
                _FulfillRecord(
                    provider_name=provider.name,
                    found=False,
                    artifact_dir_rel=str(provider_dir.relative_to(run_dir)),
                    error=f"translate_failed:{type(exc).__name__}:{exc}",
                )
            )
            continue

        # Make artifact paths relative to run_dir for the response body.
        rel_artifacts = {
            k: str(Path(v).relative_to(run_dir)) for k, v in artifacts.items()
        }
        fulfilled_records.append(
            _FulfillRecord(
                provider_name=provider.name,
                found=True,
                artifact_dir_rel=str(provider_dir.relative_to(run_dir)),
            )
        )

        # M-59: capture this bid's contract_feedback for routing.
        per_provider_feedback.append(
            (provider.name, list(getattr(result, "contract_feedback", []) or []))
        )

        # Verify via M-44 (re-using the existing verifier path).
        response_body = _build_response_body(
            request_body=request_body,
            artifacts=rel_artifacts,
            provider_name=provider.name,
        )
        from compgen.graph_compilation.kernel_codegen_response import _run_m44_verifier

        # M-44 wants the response to live on disk under a per-provider
        # validation directory so the verifier's report path doesn't
        # collide across bidders.
        verify_dir = auction_root / "verified" / provider.name
        verify_dir.mkdir(parents=True, exist_ok=True)
        # The verifier reads validation/<task_id>.validation.json — to
        # keep per-provider reports separate, we stamp the provider
        # name into the task_id used for the verifier call.
        per_bid_task_id = f"{task_id}__{provider.name}"
        m44_result = _run_m44_verifier(
            run_dir=run_dir,
            request_body=request_body,
            task_id=per_bid_task_id,
            response_body=response_body,
        )

        overall = (m44_result or {}).get("overall", "skipped")
        validation_rel = (m44_result or {}).get("validation_report_path", "")
        cert_rel = ""

        if overall == "pass":
            try:
                from compgen.kernels.kernel_certificate import emit_certificate

                cert_path = emit_certificate(
                    run_dir=run_dir,
                    request_body={**request_body, "task_id": per_bid_task_id},
                    response_body=response_body,
                    verifier_report_path=run_dir / validation_rel,
                    fallback_used=False,
                    fallback_reason="",
                )
                # Move/copy the cert into the per-provider auction tree
                # so the standard certificates/<contract_hash>.json slot
                # only carries the winner cert (the M-43 commit path
                # owns that filename).
                aux_cert = verify_dir / f"{provider.name}.certificate.json"
                shutil.copy2(cert_path, aux_cert)
                cert_rel = str(aux_cert.relative_to(run_dir))
                # Remove the global cert until the winner is chosen; the
                # winner step re-emits via emit_certificate to set the
                # canonical path.
                cert_path.unlink(missing_ok=True)
            except Exception as exc:  # noqa: BLE001
                cert_rel = f"cert_emit_failed:{type(exc).__name__}:{exc}"

        verified_records.append(
            _VerifyRecord(
                provider_name=provider.name,
                overall=overall,
                failure_kind=(m44_result or {}).get("failure_kind", ""),
                failure_summary=(m44_result or {}).get("failure_summary", ""),
                validation_report_rel=validation_rel,
                certificate_rel=cert_rel,
            )
        )

        if overall == "pass" and winner_provider is None:
            winner_provider = provider.name
            winner_artifacts = rel_artifacts
            if mode == "first-fit":
                break

    # If multi-bidder and we have a winner, re-evaluate by perf_estimate
    # among ALL passing bids — first-verified-passes-by-score is fine
    # for first-fit, but multi-bidder picks by perf.
    if mode == "multi-bidder" and verified_records:
        verified_pass_names = {v.provider_name for v in verified_records if v.overall == "pass"}
        candidates: list[tuple[str, float]] = []
        for record in bid_records:
            if record.provider_name not in verified_pass_names:
                continue
            perf = record.bid_dict.get("perf_estimate_us")
            if isinstance(perf, str):
                # +inf placeholder — treat as worst.
                p = float("inf")
            else:
                p = float(perf if perf is not None else float("inf"))
            candidates.append((record.provider_name, p))
        if candidates:
            candidates.sort(key=lambda x: (x[1], x[0]))
            winner_provider = candidates[0][0]
            # Re-derive winner_artifacts.
            winner_dir = fulfilled_root / winner_provider
            winner_artifacts = {
                "kernel_source": str((winner_dir / _find_kernel_source(winner_dir)).relative_to(run_dir)),
                "kernel_metadata": str((winner_dir / "kernel_metadata.json").relative_to(run_dir)),
                "launch_config": str((winner_dir / "launch_config.json").relative_to(run_dir)),
                "provider_claims": str((winner_dir / "provider_claims.json").relative_to(run_dir)),
            }

    # Promote winner to the standard M-43 response location.
    if winner_provider and winner_artifacts:
        _promote_winner(
            run_dir=run_dir,
            request_body=request_body,
            winner_artifacts=winner_artifacts,
            winner_provider=winner_provider,
        )

    overall = "pass" if winner_provider else "no_winner"

    result = AuctionResult(
        overall=overall,
        mode=mode,
        task_id=task_id,
        contract_hash=contract_hash,
        bids=bid_records,
        fulfilled=fulfilled_records,
        verified=verified_records,
        winner_provider=winner_provider or "",
    )

    # M-59: write the contract_feedback artifacts (auction-local + run-wide
    # aggregate). Always called, even when feedback is empty — the
    # downstream agent_decision_request emit can rely on the file's
    # existence to decide whether to surface advisory rows.
    try:
        from compgen.graph_compilation.contract_feedback_apply import (
            write_auction_feedback_artifacts,
        )

        write_auction_feedback_artifacts(
            run_dir=run_dir,
            task_id=task_id,
            contract_hash=contract_hash,
            per_provider_feedback=per_provider_feedback,
        )
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "auction.m59_feedback_persist_failed",
            error=f"{type(exc).__name__}: {exc}",
        )

    # Persist auction_report.json + winner.json + runners_up.json.
    report_path = auction_root / "auction_report.json"
    report_body = result.to_dict()
    report_body["generated_at_utc"] = _utcnow()
    report_path.write_text(
        json.dumps(report_body, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    object.__setattr__(result, "auction_report_path", str(report_path.relative_to(run_dir)))

    if winner_provider:
        (auction_root / "winner.json").write_text(
            json.dumps(
                {
                    "provider_name": winner_provider,
                    "task_id": task_id,
                    "contract_hash": contract_hash,
                    "promoted_to_response": True,
                    "generated_at_utc": _utcnow(),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    runners_up = [v.to_dict() for v in verified_records if v.provider_name != winner_provider]
    (auction_root / "runners_up.json").write_text(
        json.dumps(
            {"runners_up": runners_up, "generated_at_utc": _utcnow()},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    return result


def _find_kernel_source(directory: Path) -> str:
    for p in directory.iterdir():
        if p.name.startswith("kernel.") and p.suffix in {".c", ".cpp", ".cu", ".py"}:
            return p.name
    return "kernel.txt"


def _write_empty_report(
    *,
    run_dir: Path,
    mode: str,
    task_id: str,
    contract_hash: str,
    error: str,
) -> AuctionResult:
    auction_root = run_dir / "04_kernel_codegen" / "auction" / (task_id or "no_task")
    auction_root.mkdir(parents=True, exist_ok=True)
    result = AuctionResult(
        overall="skipped" if error == "no_applicable_providers" else "error",
        mode=mode,
        task_id=task_id,
        contract_hash=contract_hash,
        error=error,
    )
    report_path = auction_root / "auction_report.json"
    body = result.to_dict()
    body["generated_at_utc"] = _utcnow()
    report_path.write_text(
        json.dumps(body, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    object.__setattr__(result, "auction_report_path", str(report_path.relative_to(run_dir)))
    return result


def _promote_winner(
    *,
    run_dir: Path,
    request_body: dict[str, Any],
    winner_artifacts: dict[str, str],
    winner_provider: str,
) -> None:
    """Copy the winner's artifacts into the standard M-43 response
    location and re-emit the M-45 cert under the canonical contract
    hash so M-46 binds normally."""
    out_dir = run_dir / "04_kernel_codegen"
    task_id = request_body["task_id"]
    artifact_sandbox = out_dir / "artifacts" / task_id
    artifact_sandbox.mkdir(parents=True, exist_ok=True)

    # Copy each winner artifact into the standard sandbox.
    promoted: dict[str, str] = {}
    for name, rel in winner_artifacts.items():
        src = run_dir / rel
        dst = artifact_sandbox / Path(rel).name
        shutil.copy2(src, dst)
        promoted[name] = str(dst.relative_to(run_dir))

    # Write a canonical response file at the M-43 location.
    response_dir = out_dir / "responses"
    response_dir.mkdir(parents=True, exist_ok=True)
    response_path = response_dir / f"{task_id}.response.json"
    response_body = _build_response_body(
        request_body=request_body,
        artifacts=promoted,
        provider_name=winner_provider,
    )
    response_path.write_text(
        json.dumps(response_body, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    # Re-run M-44 against the promoted location to produce the canonical
    # validation report under the original task_id, then emit the cert.
    from compgen.graph_compilation.kernel_codegen_response import _run_m44_verifier
    from compgen.kernels.kernel_certificate import emit_certificate

    m44 = _run_m44_verifier(
        run_dir=run_dir,
        request_body=request_body,
        task_id=task_id,
        response_body=response_body,
    )
    if (m44 or {}).get("overall") == "pass":
        emit_certificate(
            run_dir=run_dir,
            request_body=request_body,
            response_body=response_body,
            verifier_report_path=run_dir / m44["validation_report_path"],
            fallback_used=False,
            fallback_reason="",
        )


__all__ = [
    "AuctionResult",
    "run_kernel_auction",
]
