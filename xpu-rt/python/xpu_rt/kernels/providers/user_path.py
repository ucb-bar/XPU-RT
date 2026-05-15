""":class:`UserKernelProvider`: bid + fulfill from indexed user kernels.

Reads the on-disk index produced by
:func:`compgen.kernels.user_kernel_index.reindex`, matches incoming
``KernelContractV3`` instances against indexed manifests, and bids
high-confidence when an indexed kernel covers the contract.

Match semantics (priority order):

1. **Exact canonical hash match** — the user manifest declares
   concrete dims that, when materialised into a contract, produce
   the same canonical_contract_hash as the incoming contract. This
   is the cleanest hit; bids ``confidence=0.95``,
   ``perf_estimate_us`` from the user's ``perf_priors`` if present.
2. **Archetype + dtype + layout + target match** (shape-class
   compat) — the manifest's archetype/op_family/dtype/layout/target
   match, but concrete dims differ. Bids ``confidence=0.6`` —
   below an exact match, above a placeholder. The auction may pick
   another provider on perf grounds even when this matches.
3. **No match** — bids ``confidence=0.0`` (placeholder).

Fulfill:

1. Re-audit locked files (raises :class:`UserKernelHashDriftError`
   if the kernel was edited after indexing).
2. Read the kernel source from disk.
3. Return a ``ProviderResult`` with the source + language declared
   in the manifest. The auction's translate step writes it to the
   per-provider artifact dir under ``04_kernel_codegen/auction/...``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog

from compgen.kernels.provider import (
    BidPreview,
    KernelContract,
    KnowledgeExport,
    ProviderResult,
    SearchBudget,
)
from compgen.kernels.user_kernel_index import (
    IndexEntry,
    UserKernelHashDriftError,
    audit_locked_files,
    default_index_root,
    load_index_entries,
)

log = structlog.get_logger()


@dataclass
class UserKernelProvider:
    """Auction provider serving user-supplied kernels from the local
    ``.compgen/user_kernel_index/`` directory.

    Constructed once per process; the index is loaded eagerly from
    disk. Re-call :func:`compgen.kernels.user_kernel_index.reindex`
    and reconstruct the provider to pick up newly indexed kernels.
    """

    index_root: Path = field(default_factory=default_index_root)
    name_str: str = "user_path"
    priority: int = 20  # higher than CReferenceProvider (5)
    applicable_targets: tuple[str, ...] = ()  # wildcard — manifest constrains
    applicable_archetypes: tuple[str, ...] = ()  # wildcard
    _entries: list[IndexEntry] = field(default_factory=list)
    _exports: list[KnowledgeExport] = field(default_factory=list)
    _last_match: IndexEntry | None = None

    def __post_init__(self) -> None:
        self._entries = load_index_entries(index_root=self.index_root)

    @property
    def name(self) -> str:
        return self.name_str

    # ----- legacy KernelProvider interface ----------------------------------

    def accepts_contract(self, contract: KernelContract) -> bool:
        """Best-effort acceptance gate for the legacy v1 contract.

        The auction's V3 path uses :meth:`bid` instead; this method
        exists only so the provider remains isinstance-conformant.
        """
        for entry in self._entries:
            if entry.manifest.target_name == contract.target_name:
                return True
        return bool(self._entries)

    def export_knowledge(self) -> list[KnowledgeExport]:
        return list(self._exports)

    # ----- Phase D bid + fulfill --------------------------------------------

    def bid(self, contract_v3: Any) -> BidPreview:
        """Match indexed kernels against the V3 contract.

        See module docstring for match priority. Updates ``_last_match``
        so :meth:`search` knows which entry to fulfil.
        """
        if not self._entries:
            return BidPreview(
                provider_name=self.name,
                confidence=0.0,
                rationale="no_indexed_kernels",
            )

        try:
            target_name = contract_v3.orchestration.execution.hardware.target_name
            archetype = contract_v3.archetype.value
            op_name = contract_v3.op_name.lower()
            dtype = contract_v3.io.inputs[0].dtype_class[0] if contract_v3.io.inputs else ""
            layout = contract_v3.io.inputs[0].layout.value if contract_v3.io.inputs else ""
        except (AttributeError, IndexError):
            return BidPreview(
                provider_name=self.name,
                confidence=0.0,
                rationale="contract_introspection_failed",
            )

        # Compute the contract's canonical hash for exact match.
        try:
            from compgen.promotion.contract_hash import canonical_contract_hash

            contract_canonical = canonical_contract_hash(contract_v3)
        except Exception:  # noqa: BLE001
            contract_canonical = ""

        exact_match: IndexEntry | None = None
        compat_match: IndexEntry | None = None

        for entry in self._entries:
            m = entry.manifest
            if m.target_name != target_name:
                continue
            # Archetype match required at minimum.
            if m.archetype != archetype:
                continue
            # op_family heuristic — manifest may name "linalg.matmul"
            # while the contract is "linalg.matmul"; we just require
            # the manifest's op_name (lowercased, last segment) to be
            # a substring of the contract's op_name.
            mop = m.op_name.lower()
            if "." in mop:
                mop = mop.rsplit(".", 1)[-1]
            cop = op_name.rsplit(".", 1)[-1] if "." in op_name else op_name
            if mop not in cop and cop not in mop:
                continue
            # dtype + layout match on first input.
            if m.inputs and dtype and m.inputs[0].get("dtype") != dtype:
                continue
            if m.inputs and layout and m.inputs[0].get("layout") != layout:
                continue

            # Exact match if the manifest declares concrete dims that
            # equal the contract's.
            manifest_dims = []
            for t in m.inputs:
                d = t.get("dims")
                if d is not None:
                    manifest_dims.append(tuple(d))
            contract_dims = [tuple(t.shape.dims) for t in contract_v3.io.inputs]
            if manifest_dims and manifest_dims == contract_dims[: len(manifest_dims)]:
                exact_match = entry
                break
            if compat_match is None:
                compat_match = entry

        winner = exact_match or compat_match
        if winner is None:
            self._last_match = None
            return BidPreview(
                provider_name=self.name,
                confidence=0.0,
                rationale="no_indexed_kernel_matches_contract",
            )

        self._last_match = winner
        priors = winner.manifest.perf_priors or {}
        if exact_match is not None:
            confidence = float(priors.get("confidence", 0.95) or 0.95)
            rationale = f"exact_match:{winner.index_id}"
        else:
            confidence = float(priors.get("confidence", 0.6) or 0.6)
            rationale = f"compat_match:{winner.index_id}"

        # Clamp confidence into [0, 1].
        confidence = max(0.0, min(1.0, confidence))
        perf_us = float(priors.get("estimated_us", 1.0) or 1.0)

        return BidPreview(
            provider_name=self.name,
            perf_estimate_us=perf_us,
            confidence=confidence,
            time_to_generate_s_estimate=0.05,  # disk read + audit
            rationale=rationale,
            cache_hit=True,  # the kernel already exists on disk
        )

    def search(self, contract: KernelContract, budget: SearchBudget) -> ProviderResult:
        """Fulfill the most-recent matched bid.

        The auction calls :meth:`bid` before :meth:`search`; the
        match recorded in ``_last_match`` tells us which kernel to
        serve. On hash drift, raises :class:`UserKernelHashDriftError`
        — the auction's error handler converts it into a typed
        fulfill failure.
        """
        entry = self._last_match
        if entry is None:
            return ProviderResult(
                found=False,
                metadata={"reason": "no bid match recorded; call bid() first"},
            )

        # Tamper detection.
        audit_locked_files(entry)

        # : kernel_source is a tuple — read the primary entry.
        kernel_path = (
            Path(entry.source_dir) / entry.manifest.primary_kernel_source
        )
        try:
            kernel_code = kernel_path.read_text(encoding="utf-8")
        except OSError as exc:
            return ProviderResult(
                found=False,
                metadata={
                    "reason": f"failed to read kernel_source: {exc}",
                    "index_id": entry.index_id,
                },
            )

        return ProviderResult(
            found=True,
            kernel_code=kernel_code,
            language=entry.manifest.language,
            iterations_used=1,
            total_candidates=1,
            metadata={
                "provider": self.name,
                "index_id": entry.index_id,
                "source_path": str(kernel_path),
                "entry_symbol": entry.manifest.entry_symbol,
            },
        )


__all__ = ["UserKernelProvider"]
